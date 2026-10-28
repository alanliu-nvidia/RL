# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NeMo-RL adapters for the Megatron-Bridge MIMO configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from nemo_rl.distributed.named_sharding import NamedSharding

if TYPE_CHECKING:
    from nemo_rl.data_plane import KVBatchMeta
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
    # Pack complete language/encoder data lanes into nodes. This spreads CPU
    # offload memory across hosts instead of putting language ranks on a subset.
    balance_lane_placement: bool = False
    # Optional permutation of each node's bundle positions after lane packing.
    # On four-GPU, dual-NUMA hosts, [0, 2, 1, 3] spreads language CP ranks
    # over both memory domains instead of concentrating optimizer offload.
    node_bundle_order: list[int] | None = None
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


def mimo_rank_to_lane(parallelism: Mapping[str, Any], *, world_size: int) -> list[int]:
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


def mimo_worker_bundle_indices(
    mimo_config: MIMOSFTConfig | None,
    bundle_indices: list[int] | None,
    gpus_per_node: int,
    *,
    bundle_counts: list[int] | None = None,
) -> list[tuple[int, list[int]]]:
    """Assign logical ranks to physical bundles, optionally colocating each lane.

    Preserve logical module/CP/DP ranks and each bundle's physical node index.
    Whole lanes must tile nodes, so replicas never straddle a node boundary.
    """
    if gpus_per_node <= 0:
        raise ValueError("MIMO placement requires positive GPUs per node.")
    if bundle_indices is None:
        if bundle_counts is None:
            raise ValueError(
                "MIMO placement requires bundle indices or per-node counts."
            )
        physical = [
            (pg_idx, [bundle])
            for pg_idx, count in enumerate(bundle_counts)
            for bundle in range(count)
        ]
    else:
        physical = [(i // gpus_per_node, [b]) for i, b in enumerate(bundle_indices)]
    if mimo_config is None or not mimo_config.balance_lane_placement:
        if mimo_config is not None and mimo_config.node_bundle_order is not None:
            raise ValueError("node_bundle_order requires balanced MIMO placement.")
        return physical
    if len(physical) % gpus_per_node or (
        bundle_counts is not None and any(n != gpus_per_node for n in bundle_counts)
    ):
        raise ValueError("Balanced MIMO placement requires complete GPU nodes.")
    if len({(pg_idx, b[0]) for pg_idx, b in physical}) != len(physical):
        raise ValueError("Balanced MIMO placement requires unique GPU bundles.")
    if mimo_config.node_bundle_order is not None:
        order = mimo_config.node_bundle_order
        if sorted(order) != list(range(gpus_per_node)):
            raise ValueError("node_bundle_order must permute every bundle on a node.")
        physical = [
            physical[start + position]
            for start in range(0, len(physical), gpus_per_node)
            for position in order
        ]
    lanes = mimo_rank_to_lane(mimo_config.parallelism, world_size=len(physical))
    lane_ranks: dict[int, list[int]] = {}
    for rank, lane in enumerate(lanes):
        lane_ranks.setdefault(lane, []).append(rank)
    widths = {len(ranks) for ranks in lane_ranks.values()}
    if len(widths) != 1 or gpus_per_node % next(iter(widths)):
        raise ValueError("Complete MIMO lanes must evenly tile GPUs per node.")
    rank_order = [rank for lane in sorted(lane_ranks) for rank in lane_ranks[lane]]
    rank_to_physical = dict(zip(rank_order, physical))
    return [rank_to_physical[rank] for rank in range(len(physical))]


class MIMODriverMixin:
    """Shared policy/value dispatch over heterogeneous language and vision ranks."""

    mimo_config: MIMOSFTConfig | None
    worker_group: Any
    use_sequence_packing: bool
    use_dynamic_batches: bool
    data_parallel_size: int

    def _mimo_per_rank_shards(self, sharded_data: list[Any]) -> list[Any]:
        """Give every MIMO rank the shard of the lane it belongs to.

        Encoder ranks of a lane each receive that lane's whole batch and reduce
        it to their own media range in the worker, the way the Energon task
        encoder does for SFT.
        """
        assert self.mimo_config is not None
        if len(sharded_data) != self.data_parallel_size:
            raise ValueError(
                "MIMO expects one shard per language lane: lanes="
                f"{self.data_parallel_size}, shards={len(sharded_data)}."
            )
        lanes = mimo_rank_to_lane(
            self.mimo_config.parallelism,
            world_size=len(self.worker_group.workers),
        )
        return [sharded_data[lane] for lane in lanes]

    def _mimo_lane_microbatches(
        self,
        dp_metas: list[KVBatchMeta],
        micro_batch_size: int,
    ) -> int:
        """How many microbatches every MIMO rank will step through.

        The heterogeneous schedule only runs when all of its roles enter it
        the same number of times, so this has to be one number for every
        lane. Unpacked, a lane's row count decides it. Packed, the driver's
        plan does: ``shard_meta_for_dp`` packs the whole batch with the bin
        count forced to a multiple of the lane count, so the bins divide
        evenly and every lane is handed the same number of them. That same
        plan rides each lane's ``extra_info`` to the encoder ranks, which is
        what lets them land on the boundaries the language rank will use
        without seeing its tokens.

        Dynamic batching has no equivalent guarantee -- it agrees microbatch
        *boundaries* across shards but not how many there are -- so it stays
        refused rather than hanging the schedule when two lanes disagree.
        """
        if self.use_dynamic_batches:
            raise NotImplementedError(
                "MIMO GRPO cannot use dynamic batching: it lets each lane "
                "choose its own microbatch count, and the heterogeneous "
                "schedule needs one count for every role."
            )
        # Import only the backend-independent metadata key.
        from nemo_rl.data_plane.schema import MICRO_BATCH_INDICES

        counts: set[int] = set()
        for meta in dp_metas:
            plan = (meta.extra_info or {}).get(MICRO_BATCH_INDICES)
            if plan is not None:
                counts.add(len(plan[0]))
                continue
            if self.use_sequence_packing:
                raise RuntimeError(
                    "MIMO needs the driver's microbatch plan when sequence "
                    "packing is on, but shard_meta_for_dp did not stamp one."
                )
            lane_size = len(meta.sample_ids)
            if lane_size % micro_batch_size:
                raise ValueError(
                    f"MIMO lane batch {lane_size} is not a whole number of "
                    f"microbatches of {micro_batch_size}."
                )
            counts.add(lane_size // micro_batch_size)
        if len(counts) != 1:
            raise ValueError(
                "MIMO lanes must agree on their microbatch count so every "
                f"rank steps through the same schedule; got {sorted(counts)}."
            )
        return counts.pop()

    def _mimo_language_lane_ranks(self) -> list[int]:
        """Rank of one language replica per data lane.

        The MIMO stand-in for the ``is_replica_leader`` filter, which a flat
        worker axis cannot answer: it reports every rank a leader, encoder
        ranks included. Each module's ranks are contiguous
        tensor x context x pipeline replicas from its offset, so striding by
        that width picks exactly one rank per lane -- the same disjoint cover
        of the batch the named-axis filter gives every other backend.
        """
        assert self.mimo_config is not None
        language = self.mimo_config.parallelism["module_parallelisms"]["language"]
        replica_width = (
            language["tensor_model_parallel_size"]
            * language["context_parallel_size"]
            * language["pipeline_model_parallel_size"]
        )
        offset = language["rank_offset"]
        return [
            offset + lane * replica_width
            for lane in range(language["data_parallel_size"])
        ]
