# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map policy parallel ranks to logical Energon loader copies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

from nemo_rl.distributed.named_sharding import (
    GTP_WEIGHT_REMAT_AXIS,
    REPLICATED_AXES,
    NamedSharding,
)


@dataclass(frozen=True)
class LoaderCopy:
    """One physical loader view of a logical data lane and its consumers."""

    logical_rank: int
    logical_world_size: int
    owner_rank: int
    delivery_ranks: tuple[int, ...]
    role: Literal["language", "encoder"] | None = None
    # Additional role consumers of this physical loader's canonical batch.
    role_delivery_ranks: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # Encoder copies only: this copy's media range among the lane's encoder ranks.
    encoder_shard: int | None = None


@dataclass(frozen=True)
class DataLoaderPlacementPlan:
    """Stable mapping from logical data shards to policy ranks."""

    copies: tuple[LoaderCopy, ...]
    placement_hash: str

    @property
    def logical_world_size(self) -> int:
        return len({copy.logical_rank for copy in self.copies})

    def copy_for_logical_rank(self, logical_rank: int) -> LoaderCopy:
        """Return the copy for ``logical_rank``."""
        matches = [copy for copy in self.copies if copy.logical_rank == logical_rank]
        if not matches:
            raise ValueError(f"Logical data rank {logical_rank} has no loader copy.")
        if len(matches) != 1:
            raise ValueError(
                f"Logical data rank {logical_rank} has {len(matches)} role copies."
            )
        return matches[0]


class DataLoaderTopologyMapper(Protocol):
    """Build a logical loader placement for a policy sharding layout."""

    def map(self, sharding: NamedSharding) -> DataLoaderPlacementPlan: ...


class DefaultDataLoaderTopologyMapper:
    """Create one loader per DP×GTP data lane on TP0/PP0/CP0."""

    def map(self, sharding: NamedSharding) -> DataLoaderPlacementPlan:
        """Map each data-lane coordinate to one owner and its TP/PP/CP consumers.

        GTP peers hold distinct microbatches, so they are data-shard ranks, not
        replicas. Logical rank is ``dp * gtp_size + gtp``, matching MCore's
        full data-distribution group. When ``gtp_weight_remat`` is absent, GTP
        size is 1 and the plan is one copy per DP replica.
        """
        if "data_parallel" not in sharding.names:
            raise ValueError("Policy sharding must define a data_parallel axis.")

        dp_size = sharding.get_axis_size("data_parallel")
        has_gtp_axis = GTP_WEIGHT_REMAT_AXIS in sharding.names
        gtp_size = sharding.get_axis_size(GTP_WEIGHT_REMAT_AXIS) if has_gtp_axis else 1
        logical_world_size = dp_size * gtp_size
        copies: list[LoaderCopy] = []
        for dp_rank in range(dp_size):
            for gtp_rank in range(gtp_size):
                logical_rank = dp_rank * gtp_size + gtp_rank
                shard_coords = {"data_parallel": dp_rank}
                if has_gtp_axis:
                    shard_coords[GTP_WEIGHT_REMAT_AXIS] = gtp_rank
                delivery_ranks = tuple(sharding.get_ranks_by_coord(**shard_coords))
                owner_coords = dict(shard_coords)
                owner_coords.update(
                    {axis: 0 for axis in REPLICATED_AXES if axis in sharding.names}
                )
                owner_candidates = sharding.get_ranks_by_coord(**owner_coords)
                if len(owner_candidates) != 1:
                    raise ValueError(
                        "Default loader placement requires exactly one TP0/PP0/CP0 "
                        f"owner for data lane dp={dp_rank} gtp={gtp_rank}; "
                        f"found {owner_candidates}."
                    )
                copies.append(
                    LoaderCopy(
                        logical_rank=logical_rank,
                        logical_world_size=logical_world_size,
                        owner_rank=owner_candidates[0],
                        delivery_ranks=delivery_ranks,
                    )
                )

        payload = {
            "parallel_shape": sharding.shape,
            "parallel_axes": sharding.names,
            "copies": [
                {
                    "logical_rank": copy.logical_rank,
                    "logical_world_size": copy.logical_world_size,
                    "owner_rank": copy.owner_rank,
                    "delivery_ranks": list(copy.delivery_ranks),
                }
                for copy in copies
            ],
        }
        placement_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return DataLoaderPlacementPlan(
            copies=tuple(copies), placement_hash=placement_hash
        )


def resolve_topology_mapper(name: str) -> DataLoaderTopologyMapper:
    """Resolve the Stage 1 topology mapper."""
    if name != "default":
        raise ValueError(f"Unknown data-loader topology mapper {name!r}.")
    return DefaultDataLoaderTopologyMapper()


def build_mimo_data_loader_placement(
    parallelism: Mapping[str, Any], *, world_size: int, single_owner: bool
) -> DataLoaderPlacementPlan:
    """Map explicit YAML rank spans to loader owners; workers validate topology.

    Every encoder DP replica runs a loader. The ``stride = encoder DP /
    language DP`` encoder replicas of one lane read the same stream and each
    keeps one media range of every pack (``encoder_shard``). Copies are ordered
    by lane: the lane's encoder shards, then its language copy (role loaders).
    With single-owner loading, encoder shard 0 also delivers the language batch.
    """
    modules = {
        name: {
            "tp": config["tensor_model_parallel_size"],
            "cp": config["context_parallel_size"],
            "pp": config["pipeline_model_parallel_size"],
            "dp": config["data_parallel_size"],
            "rank_offset": config["rank_offset"],
        }
        for name, config in parallelism["module_parallelisms"].items()
    }
    language = modules["language"]
    (encoder,) = (config for name, config in modules.items() if name != "language")
    if encoder["dp"] < language["dp"] or encoder["dp"] % language["dp"]:
        raise ValueError(
            "MIMO encoder DP must be a positive multiple of language DP: "
            f"encoder={encoder['dp']}, language={language['dp']}."
        )
    encoder_stride = encoder["dp"] // language["dp"]
    payload = {
        "kind": "mimo-native-v2-role-loaders-v2",
        "world_size": world_size,
        "logical_world_size": language["dp"],
        "modules": modules,
        "owners": {
            "encoder": encoder["rank_offset"],
            "language": language["rank_offset"],
        },
    }
    if single_owner:
        payload["kind"] = "mimo-native-v2-single-owner-v1"
    if encoder_stride > 1:
        # Earlier layouts loaded one encoder copy per lane. Their loader and save
        # states must not restore into the split layout.
        payload["encoder_media_split"] = encoder_stride
    placement_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    encoder_replica_size = encoder["tp"] * encoder["cp"] * encoder["pp"]
    language_replica_size = language["tp"] * language["cp"] * language["pp"]
    copies = []
    for lane in range(language["dp"]):
        language_owner = language["rank_offset"] + lane * language_replica_size
        language_ranks = tuple(
            range(language_owner, language_owner + language_replica_size)
        )
        for shard in range(encoder_stride):
            encoder_owner = (
                encoder["rank_offset"]
                + (lane * encoder_stride + shard) * encoder_replica_size
            )
            copies.append(
                LoaderCopy(
                    role="encoder",
                    logical_rank=lane,
                    logical_world_size=language["dp"],
                    owner_rank=encoder_owner,
                    delivery_ranks=tuple(
                        range(encoder_owner, encoder_owner + encoder_replica_size)
                    ),
                    role_delivery_ranks=(
                        {"language": language_ranks}
                        if single_owner and shard == 0
                        else {}
                    ),
                    encoder_shard=shard,
                )
            )
        if not single_owner:
            copies.append(
                LoaderCopy(
                    role="language",
                    logical_rank=lane,
                    logical_world_size=language["dp"],
                    owner_rank=language_owner,
                    delivery_ranks=language_ranks,
                )
            )
    return DataLoaderPlacementPlan(copies=tuple(copies), placement_hash=placement_hash)
