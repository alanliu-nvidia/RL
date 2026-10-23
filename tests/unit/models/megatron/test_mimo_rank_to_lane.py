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

import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nemo_rl.models.megatron.mimo import (
    MIMOSFTConfig,
    mimo_rank_to_lane,
    mimo_worker_bundle_indices,
)


def _parallelism(
    *,
    language_dp,
    encoder_dp,
    language_tp=1,
    language_cp=1,
    language_offset=0,
    encoder_offset=None,
):
    language_replica = language_tp * language_cp
    if encoder_offset is None:
        encoder_offset = language_offset + language_dp * language_replica
    return {
        "module_parallelisms": {
            "language": {
                "tensor_model_parallel_size": language_tp,
                "context_parallel_size": language_cp,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": language_dp,
                "rank_offset": language_offset,
            },
            "images": {
                "tensor_model_parallel_size": 1,
                "context_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": encoder_dp,
                "rank_offset": encoder_offset,
            },
        }
    }


def test_language_ranks_come_first_then_strided_encoder_lanes():
    lanes = mimo_rank_to_lane(_parallelism(language_dp=2, encoder_dp=4), world_size=6)
    # Ranks 0-1 are the two language lanes; 2-5 are two encoder ranks per lane.
    assert lanes == [0, 1, 0, 0, 1, 1]


def test_a_language_replica_spans_its_whole_tp_cp_group():
    lanes = mimo_rank_to_lane(
        _parallelism(language_dp=2, encoder_dp=2, language_tp=2, language_cp=2),
        world_size=10,
    )
    assert lanes[:4] == [0, 0, 0, 0]
    assert lanes[4:8] == [1, 1, 1, 1]
    assert lanes[8:] == [0, 1]


def test_equal_dp_gives_each_encoder_rank_its_own_lane():
    lanes = mimo_rank_to_lane(_parallelism(language_dp=4, encoder_dp=4), world_size=8)
    assert lanes == [0, 1, 2, 3, 0, 1, 2, 3]


def test_rejects_encoder_dp_that_is_not_a_multiple():
    with pytest.raises(ValueError, match="multiple of"):
        mimo_rank_to_lane(_parallelism(language_dp=4, encoder_dp=6), world_size=10)


def test_rejects_a_world_the_modules_do_not_cover():
    with pytest.raises(ValueError, match="without a module"):
        mimo_rank_to_lane(_parallelism(language_dp=2, encoder_dp=2), world_size=5)


def test_rejects_overlapping_module_rank_spans():
    with pytest.raises(ValueError, match="claimed by two modules"):
        mimo_rank_to_lane(
            _parallelism(language_dp=2, encoder_dp=2, encoder_offset=1),
            world_size=4,
        )


@pytest.mark.parametrize("language_dp", [4, 32])
def test_balanced_bundles_spread_language_and_keep_complete_lanes_on_nodes(language_dp):
    parallelism = _parallelism(
        language_dp=language_dp, encoder_dp=2 * language_dp, language_cp=2
    )
    config = MIMOSFTConfig(parallelism=parallelism, balance_lane_placement=True)
    world = 4 * language_dp
    bundles = [7 + 3 * i for i in range(world)]
    placement = mimo_worker_bundle_indices(config, bundles, 4)
    lanes = mimo_rank_to_lane(parallelism, world_size=world)
    assert sorted(b[0] for _, b in placement) == sorted(bundles)
    for node in range(language_dp):
        ranks = [rank for rank, (n, _) in enumerate(placement) if n == node]
        assert len(ranks) == 4
        assert sum(rank < 2 * language_dp for rank in ranks) == 2
        assert {lanes[rank] for rank in ranks} == {node}
        assert {b[0] for n, b in placement if n == node} == set(
            bundles[4 * node : 4 * node + 4]
        )


def test_default_bundle_order_is_preserved_for_policy_and_value():
    bundles = [8, 10, 12, 14]
    expected = [(0, [8]), (0, [10]), (1, [12]), (1, [14])]
    assert mimo_worker_bundle_indices(None, bundles, 2) == expected
    config = MIMOSFTConfig(parallelism={})
    assert mimo_worker_bundle_indices(config, bundles, 2) == expected


def test_balanced_lane_that_cannot_fit_node_is_rejected():
    config = MIMOSFTConfig(
        parallelism=_parallelism(language_dp=2, encoder_dp=4),
        balance_lane_placement=True,
    )
    with pytest.raises(ValueError, match="evenly tile"):
        mimo_worker_bundle_indices(config, list(range(6)), 2)


def test_balanced_placement_rejects_partial_nodes_and_duplicate_bundles():
    config = MIMOSFTConfig(
        parallelism=_parallelism(language_dp=2, encoder_dp=4, language_cp=2),
        balance_lane_placement=True,
    )
    with pytest.raises(ValueError, match="complete GPU nodes"):
        mimo_worker_bundle_indices(config, list(range(8)), 3)
    with pytest.raises(ValueError, match="unique GPU bundles"):
        mimo_worker_bundle_indices(config, [0] * 8, 4)


def test_balanced_per_node_groups_preserve_group_and_local_bundle_ids():
    config = MIMOSFTConfig(
        parallelism=_parallelism(language_dp=4, encoder_dp=8, language_cp=2),
        balance_lane_placement=True,
    )
    placement = mimo_worker_bundle_indices(config, None, 4, bundle_counts=[4] * 4)
    for node in range(4):
        ranks = [r for r, (pg, _) in enumerate(placement) if pg == node]
        assert len(ranks) == 4
        assert sum(r < 8 for r in ranks) == 2
        assert {b[0] for pg, b in placement if pg == node} == set(range(4))


@pytest.mark.parametrize("language_dp", [4, 32])
@pytest.mark.parametrize("unified", [False, True])
def test_node_bundle_order_spreads_language_across_numa_domains(language_dp, unified):
    world = 4 * language_dp
    config = MIMOSFTConfig(
        parallelism=_parallelism(
            language_dp=language_dp, encoder_dp=2 * language_dp, language_cp=2
        ),
        balance_lane_placement=True,
        node_bundle_order=[0, 2, 1, 3],
    )
    placement = mimo_worker_bundle_indices(
        config,
        list(range(world)) if unified else None,
        4,
        bundle_counts=None if unified else [4] * language_dp,
    )
    for node in range(language_dp):
        base = 4 * node if unified else 0
        assert placement[2 * node : 2 * node + 2] == [
            (node, [base]),
            (node, [base + 2]),
        ]
        start = 2 * language_dp + 2 * node
        assert placement[start : start + 2] == [(node, [base + 1]), (node, [base + 3])]


@pytest.mark.parametrize("order", [[0, 2, 1], [0, 2, 2, 3], [0, 2, 1, 4]])
def test_invalid_node_bundle_order_is_rejected(order):
    config = MIMOSFTConfig(
        parallelism=_parallelism(language_dp=4, encoder_dp=8, language_cp=2),
        balance_lane_placement=True,
        node_bundle_order=order,
    )
    with pytest.raises(ValueError, match="permute every bundle"):
        mimo_worker_bundle_indices(config, None, 4, bundle_counts=[4] * 4)


@pytest.mark.parametrize("kind", ["policy", "value"])
@pytest.mark.parametrize("unified", [False, True])
def test_driver_constructor_places_lanes_for_both_group_layouts(
    monkeypatch, kind, unified
):
    # Driver imports are heavier than the backend-independent placement helpers.
    from omegaconf import OmegaConf

    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.models.value.lm_value import Value
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    for name in ("NT4_NANO_MODEL_DIR", "NT4_NANO_PROCESSOR", "NT4_NANO_MIMO_CKPT"):
        monkeypatch.setenv(name, "/test/asset")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "10.0")
    register_omegaconf_resolvers()
    recipe = (
        Path(__file__).parents[4]
        / "examples/configs/recipes/vlm/vlm_ppo-nt4nano-clevr-mimo-8n4g-smoke.yaml"
    )
    config = OmegaConf.to_container(load_config(recipe), resolve=True)[kind]
    cluster = SimpleNamespace(
        _sorted_bundle_indices=list(range(16)) if unified else None,
        num_gpus_per_node=4,
        world_size=lambda: 16,
        get_placement_groups=lambda: [SimpleNamespace(bundle_count=4)] * 4,
    )
    driver = Policy if kind == "policy" else Value
    module = (
        "nemo_rl.models.policy.lm_policy"
        if kind == "policy"
        else "nemo_rl.models.value.lm_value"
    )

    class WorkersRequested(Exception):
        pass

    with (
        patch("nemo_rl.models.policy.lm_policy.RayQueue"),
        patch("ray.util.queue.Queue"),
        patch(module + ".RayWorkerBuilder"),
        patch(module + ".RayWorkerGroup", side_effect=WorkersRequested) as workers,
        patch.object(driver, "__del__"),
    ):
        kwargs = (
            {"allow_mimo_sft": True, "init_reference_model": False}
            if kind == "policy"
            else {}
        )
        with pytest.raises(WorkersRequested):
            driver(cluster, config, tokenizer=MagicMock(), **kwargs)
    placement = workers.call_args.kwargs["bundle_indices_list"]
    assert len(placement) == 16
    for node in range(4):
        ranks = [r for r, (pg, _) in enumerate(placement) if pg == node]
        assert len(ranks) == 4
        assert sum(r < 8 for r in ranks) == 2
        base = 4 * node if unified else 0
        assert {
            b[0] for r, (pg, b) in enumerate(placement) if pg == node and r < 8
        } == {base, base + 2}
