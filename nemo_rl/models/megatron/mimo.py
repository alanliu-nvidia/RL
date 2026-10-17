# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NeMo-RL adapters for the Megatron-Bridge MIMO configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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


def mimo_worker_sharding(world_size: int) -> NamedSharding:
    """Describe physical Ray workers without inventing a homogeneous MIMO grid."""
    return NamedSharding(np.arange(world_size), names=["worker"])
