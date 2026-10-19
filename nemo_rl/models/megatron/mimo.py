# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NeMo-RL adapters for the Megatron-Bridge MIMO configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from nemo_rl.distributed.named_sharding import NamedSharding

if TYPE_CHECKING:
    from megatron.bridge.models.megatron_mimo import (
        MegatronMIMOParallelismConfig,
    )


class MIMOSFTConfig(BaseModel):
    """SFT-v2 adapter around Bridge's native heterogeneous-parallelism config."""

    model_config = ConfigDict(extra="allow")

    # Opaque pass-through to Bridge, which owns the schema and its defaults.
    # Keep plain data here: the driver/controller do not have backend dependencies.
    parallelism: dict[str, Any]
    # True shares the encoder batch via TransferQueue; False uses local role loaders.
    single_owner_loader: bool = True
    # Queue each worker's optimizer finish behind its own train call, so a module
    # can update before the other module's ranks finish training. This saves time
    # only if no collective spans modules after the backward pass (Core
    # mimo_module_local_grad_clip, Bridge broadcast_num_tokens_to_encoders_only).
    async_finish: bool = False
    # When encoder DP > language DP, the encoder ranks of a lane split each pack's
    # images and video tubelets into contiguous ranges of equal cost. Cost of one
    # unit = its patches per frame + this weight per frame.
    encoder_media_split_frame_weight: int = Field(default=165, ge=0)
    # Frames per video tubelet, the smallest piece that split keeps whole. Match
    # the task encoder's temporal_patch_size; it only affects video.
    encoder_media_split_temporal_patch_size: int = Field(default=2, gt=0)


def resolve_mimo_sft_config(policy_config: Mapping[str, Any]) -> MIMOSFTConfig | None:
    """Parse the opt-in MIMO block without importing the training backend."""
    megatron_config = policy_config.get("megatron_cfg") or {}
    raw_config = megatron_config.get("mimo")
    if raw_config is None:
        return None
    return MIMOSFTConfig.model_validate(raw_config)


def validate_mimo_parallelism(
    mimo_config: MIMOSFTConfig,
    *,
    world_size: int,
    megatron_config: Mapping[str, Any],
) -> "MegatronMIMOParallelismConfig":
    """Worker-only Bridge validation and legacy language-settings check."""
    from megatron.bridge.models.megatron_mimo import (
        MegatronMIMOParallelismConfig,
    )
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

    parallelism = TypeAdapter(MegatronMIMOParallelismConfig).validate_python(
        mimo_config.parallelism
    )
    parallelism.finalize(world_size)

    # Existing NeMo-RL setup helpers still read these first-class fields before
    # the standard provider is wrapped by MegatronMIMOProvider. Keep that legacy
    # view consistent with Bridge's authoritative language-module config.
    language = parallelism.get_parallelism(MIMO_LANGUAGE_MODULE_KEY)
    encoders = [
        config
        for name, config in parallelism.module_parallelisms.items()
        if name != MIMO_LANGUAGE_MODULE_KEY
    ]
    if len(encoders) != 1:
        raise ValueError(
            "The initial MIMO SFT v2 data path requires exactly one encoder component."
        )
    if encoders[0].data_parallel_size < language.data_parallel_size:
        raise NotImplementedError(
            "MIMO SFT v2 currently requires encoder DP >= language DP; "
            "multi-lane encoder owners are not implemented."
        )
    if encoders[0].data_parallel_size % language.data_parallel_size:
        raise ValueError("MIMO encoder DP must be a multiple of language DP.")
    if encoders[0].context_parallel_size != 1:
        raise NotImplementedError("MIMO encoder data delivery currently requires CP=1.")
    for name, config in parallelism.module_parallelisms.items():
        if config.pipeline_model_parallel_size != 1:
            raise NotImplementedError(
                f"MIMO SFT v2 data delivery currently requires {name} PP=1."
            )
    expected_language_parallelism = {
        "tensor_model_parallel_size": megatron_config["tensor_model_parallel_size"],
        "pipeline_model_parallel_size": megatron_config["pipeline_model_parallel_size"],
        "context_parallel_size": megatron_config["context_parallel_size"],
        "expert_model_parallel_size": megatron_config["expert_model_parallel_size"],
        "expert_tensor_parallel_size": megatron_config["expert_tensor_parallel_size"],
    }
    mismatches = {
        name: (getattr(language, name), expected)
        for name, expected in expected_language_parallelism.items()
        if getattr(language, name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}: mimo={actual}, megatron_cfg={expected}"
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(
            "MIMO language parallelism must match the existing first-class Megatron settings; "
            + details
        )
    return parallelism


def mimo_rank_to_lane(
    parallelism: Mapping[str, Any], *, world_size: int
) -> list[int]:
    """Map every rank to the language lane whose batch it has to receive.

    ``build_mimo_data_loader_placement`` answers a neighbouring question --
    which rank *owns a loader* for a lane -- and under single-owner loading not
    every rank appears there. GRPO fans data out from the controller, so it
    needs the lane every rank must *see*. Ranks are contiguous ``tp*cp*pp``
    replicas from each module's ``rank_offset``, the same layout that function
    uses; expert parallelism folds into DP and does not widen a replica.
    """
    modules = parallelism["module_parallelisms"]
    language = modules["language"]
    encoders = [config for name, config in modules.items() if name != "language"]
    if len(encoders) != 1:
        raise ValueError("MIMO GRPO requires exactly one encoder component.")
    encoder = encoders[0]

    language_dp = language["data_parallel_size"]
    encoder_dp = encoder["data_parallel_size"]
    if language_dp <= 0 or encoder_dp <= 0 or encoder_dp % language_dp:
        raise ValueError(
            f"MIMO encoder DP {encoder_dp} must be a positive multiple of "
            f"language DP {language_dp}."
        )
    stride = encoder_dp // language_dp

    lanes: list[Optional[int]] = [None] * world_size

    def assign(offset: int, replica_size: int, replicas: int, lane_of) -> None:
        for replica in range(replicas):
            start = offset + replica * replica_size
            for rank in range(start, start + replica_size):
                if not 0 <= rank < world_size:
                    raise ValueError(
                        f"MIMO rank {rank} is outside a world of {world_size}."
                    )
                if lanes[rank] is not None:
                    raise ValueError(f"MIMO rank {rank} is claimed by two modules.")
                lanes[rank] = lane_of(replica)

    assign(
        language["rank_offset"],
        _replica_size(language),
        language_dp,
        lambda replica: replica,
    )
    assign(
        encoder["rank_offset"],
        _replica_size(encoder),
        encoder_dp,
        lambda replica: replica // stride,
    )

    unassigned = [rank for rank, lane in enumerate(lanes) if lane is None]
    if unassigned:
        raise ValueError(
            f"MIMO leaves ranks {unassigned} without a module; the parallelism "
            "block does not cover the world."
        )
    return cast(list[int], lanes)


def _replica_size(module: Mapping[str, Any]) -> int:
    return (
        module["tensor_model_parallel_size"]
        * module["context_parallel_size"]
        * module["pipeline_model_parallel_size"]
    )


def mimo_worker_sharding(world_size: int) -> NamedSharding:
    """Describe physical Ray workers without inventing a homogeneous MIMO grid."""
    return NamedSharding(np.arange(world_size), names=["worker"])
