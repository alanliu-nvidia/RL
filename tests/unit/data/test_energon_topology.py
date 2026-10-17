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

from __future__ import annotations

import numpy as np
import pytest

from nemo_rl.data.energon.topology import (
    DefaultDataLoaderTopologyMapper,
    build_mimo_data_loader_placement,
    resolve_topology_mapper,
)
from nemo_rl.distributed.named_sharding import GTP_WEIGHT_REMAT_AXIS, NamedSharding


def _sharding(*, dp: int, pp: int = 1, cp: int = 1, tp: int = 1) -> NamedSharding:
    world_size = dp * pp * cp * tp
    return NamedSharding(
        np.arange(world_size).reshape(pp, dp, cp, tp),
        [
            "pipeline_parallel",
            "data_parallel",
            "context_parallel",
            "tensor_parallel",
        ],
    )


@pytest.mark.parametrize("dp", [1, 2, 4])
def test_default_mapper_creates_one_copy_per_dp_replica(dp: int) -> None:
    plan = DefaultDataLoaderTopologyMapper().map(_sharding(dp=dp))

    assert plan.logical_world_size == dp
    assert [copy.logical_rank for copy in plan.copies] == list(range(dp))
    assert [copy.owner_rank for copy in plan.copies] == list(range(dp))


def test_default_mapper_delivers_to_all_tp_pp_cp_ranks() -> None:
    sharding = _sharding(dp=2, pp=2, cp=2, tp=2)
    plan = DefaultDataLoaderTopologyMapper().map(sharding)

    for copy in plan.copies:
        expected = tuple(sharding.get_ranks_by_coord(data_parallel=copy.logical_rank))
        assert copy.delivery_ranks == expected
        owner_coords = sharding.get_worker_coords(copy.owner_rank)
        assert owner_coords == {
            "pipeline_parallel": 0,
            "data_parallel": copy.logical_rank,
            "context_parallel": 0,
            "tensor_parallel": 0,
        }


def test_placement_hash_is_stable_and_layout_sensitive() -> None:
    mapper = DefaultDataLoaderTopologyMapper()
    first = mapper.map(_sharding(dp=2, cp=2))
    second = mapper.map(_sharding(dp=2, cp=2))
    different = mapper.map(_sharding(dp=2, tp=2))

    assert first.placement_hash == second.placement_hash
    assert first.placement_hash != different.placement_hash


def test_resolver_rejects_unknown_mapper() -> None:
    with pytest.raises(ValueError, match="Unknown data-loader topology mapper"):
        resolve_topology_mapper("mimo")


def _gtp_sharding(
    *, dp: int, gtp: int, pp: int = 1, cp: int = 1, tp: int = 1
) -> NamedSharding:
    world_size = dp * gtp * pp * cp * tp
    return NamedSharding(
        np.arange(world_size).reshape(pp, dp, gtp, cp, tp),
        [
            "pipeline_parallel",
            "data_parallel",
            GTP_WEIGHT_REMAT_AXIS,
            "context_parallel",
            "tensor_parallel",
        ],
    )


@pytest.mark.parametrize("pp", [1, 2])
def test_default_mapper_creates_one_copy_per_dp_gtp_lane(pp: int) -> None:
    sharding = _gtp_sharding(dp=2, gtp=2, pp=pp, cp=2, tp=2)
    plan = DefaultDataLoaderTopologyMapper().map(sharding)

    assert plan.logical_world_size == 4
    assert [copy.logical_rank for copy in plan.copies] == [0, 1, 2, 3]
    for copy in plan.copies:
        dp_rank, gtp_rank = divmod(copy.logical_rank, 2)
        expected = tuple(
            sharding.get_ranks_by_coord(
                data_parallel=dp_rank, **{GTP_WEIGHT_REMAT_AXIS: gtp_rank}
            )
        )
        assert copy.delivery_ranks == expected
        owner_coords = sharding.get_worker_coords(copy.owner_rank)
        assert owner_coords == {
            "pipeline_parallel": 0,
            "data_parallel": dp_rank,
            GTP_WEIGHT_REMAT_AXIS: gtp_rank,
            "context_parallel": 0,
            "tensor_parallel": 0,
        }
        other_gtp = tuple(
            sharding.get_ranks_by_coord(
                data_parallel=dp_rank, **{GTP_WEIGHT_REMAT_AXIS: 1 - gtp_rank}
            )
        )
        assert not set(copy.delivery_ranks) & set(other_gtp)


def test_gtp_loader_delivery_covers_each_rank_once() -> None:
    sharding = _gtp_sharding(dp=2, gtp=2, pp=2, cp=2, tp=2)
    plan = DefaultDataLoaderTopologyMapper().map(sharding)
    delivered = [rank for copy in plan.copies for rank in copy.delivery_ranks]
    assert sorted(delivered) == list(range(32))


@pytest.mark.parametrize("language_dp", [1, 2])
@pytest.mark.parametrize("single_owner", [False, True])
def test_mimo_placement_owners_and_deliveries(language_dp, single_owner) -> None:
    parallelism = {
        "module_parallelisms": {
            "images": {
                "tensor_model_parallel_size": 1,
                "context_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": 4,
                "rank_offset": 0,
            },
            "language": {
                "tensor_model_parallel_size": 2 // language_dp,
                "context_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": language_dp,
                "rank_offset": 4,
            },
        }
    }

    plan = build_mimo_data_loader_placement(
        parallelism, world_size=8, single_owner=single_owner
    )

    # Each of the lane's encoder DP replicas loads the lane and keeps one media
    # range of every pack (its encoder shard).
    stride = 4 // language_dp
    assert plan.logical_world_size == language_dp
    assert len(plan.copies) == (stride + (0 if single_owner else 1)) * language_dp
    for lane in range(language_dp):
        lane_copies = [copy for copy in plan.copies if copy.logical_rank == lane]
        encoders = [copy for copy in lane_copies if copy.role == "encoder"]
        language_owner = 4 + lane * (4 // language_dp)
        language_ranks = tuple(range(language_owner, language_owner + 4 // language_dp))
        assert [copy.encoder_shard for copy in encoders] == list(range(stride))
        assert [copy.owner_rank for copy in encoders] == [
            lane * stride + shard for shard in range(stride)
        ]
        assert [copy.delivery_ranks for copy in encoders] == [
            (lane * stride + shard,) for shard in range(stride)
        ]
        if single_owner:
            assert encoders[0].role_delivery_ranks == {"language": language_ranks}
            assert all(copy.role_delivery_ranks == {} for copy in encoders[1:])
            assert len(lane_copies) == stride
            continue
        (language,) = [copy for copy in lane_copies if copy.role == "language"]
        assert language.owner_rank == language_owner
        assert language.delivery_ranks == language_ranks
        assert language.encoder_shard is None
    if single_owner:
        independent = build_mimo_data_loader_placement(
            parallelism, world_size=8, single_owner=False
        )
        assert plan.placement_hash != independent.placement_hash
    elif language_dp == 2:
        assert [copy.owner_rank for copy in plan.copies] == [0, 1, 4, 2, 3, 6]
        # Language CP1 replicas never create loaders.
        assert not {5, 7} & {copy.owner_rank for copy in plan.copies}
    with pytest.raises(ValueError, match=f"has {len(lane_copies)} role copies"):
        plan.copy_for_logical_rank(0)
