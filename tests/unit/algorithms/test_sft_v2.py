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

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from nemo_rl.algorithms.sft_v2 import (
    SFT_V2_TEED_METRICS,
    SFTSingleControllerActor,
    SFTV2SaveState,
    _max_train_steps,
    _validate_mimo_topology_mapper,
)
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data.energon.topology import LoaderCopy
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.interfaces import LocalDataPlaneConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.telemetry.instrumentation import TRACE_CARRIER_KWARG

_ACTOR_CLS = SFTSingleControllerActor.__ray_metadata__.modified_class


def _envelope(rank: int, *, source_count: int = 1) -> StepEnvelope:
    return StepEnvelope(
        meta=KVBatchMeta(
            partition_id=f"p{rank}",
            task_name="train",
            sample_ids=[f"s{rank}"],
            fields=["input_ids"],
            sequence_lengths=[8],
        ),
        logical_rank=rank,
        logical_world_size=2,
        source_ids=tuple(
            f"source-{rank}-{source_index}" for source_index in range(source_count)
        ),
        field_names=("input_ids",),
        sequence_lengths=(8,),
        load_seconds=0.1 + rank * 0.1,
        valid_tokens=4,
        load_phase_seconds={
            "iter": 0.01 + rank * 0.01,
            "prepare": 0.02,
            "post-prepare": 0.03,
            "tensordict": 0.04,
            "publish": 0.05,
        },
    )


def _controller() -> object:
    controller = object.__new__(_ACTOR_CLS)
    controller._tracer = None
    controller._trainer = MagicMock(mimo_config=None)
    controller._trainer.mimo_config = None
    controller._trainer.finish_train_step.return_value = {
        "loss": 1.0,
        "grad_norm": 0.5,
        "all_mb_metrics": {},
        "mtp_metrics": {"mtp_1_loss": 0.25},
        "step_phases": {"fetch": 0.1, "fwd_bwd": 0.2},
    }
    controller._trainer.train_placed_microbatches.return_value = {
        "stamp_pad": 0.01,
        "dispatch": 0.02,
    }
    controller._master_config = SimpleNamespace(
        sft=SimpleNamespace(val_at_start=False, val_period=0, val_at_end=False)
    )
    controller._placement_plan = SimpleNamespace(logical_world_size=2, copies=[0, 1])
    controller._save_state = SFTV2SaveState(0, 0, 0, "hash")
    controller._load_envelopes = MagicMock(return_value=[_envelope(0), _envelope(1)])
    controller._loader_call = MagicMock(return_value=[None, None])
    controller._loss_fn = object()
    return controller


def _valid_setup_config(
    *,
    sft_overrides: dict[str, Any] | None = None,
    data_overrides: dict[str, Any] | None = None,
    policy_overrides: dict[str, dict[str, Any]] | None = None,
    metric_name: str | None = None,
) -> SimpleNamespace:
    sft = {
        "seed": 0,
        "val_period": 0,
        "val_at_start": False,
        "val_at_end": False,
    }
    sft.update(sft_overrides or {})
    data = {
        "backend": "energon",
        "validation": None,
        "max_input_seq_length": 128,
        "energon": SimpleNamespace(packing_buffer_size=None),
    }
    data.update(data_overrides or {})
    policy = {
        "megatron_cfg": {
            "enabled": True,
            "context_parallel_size": 1,
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
        },
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "make_sequence_length_divisible_by": 1,
    }
    for section, values in (policy_overrides or {}).items():
        policy[section].update(values)
    packing = None
    if data["energon"].packing_buffer_size is not None:
        packing = SimpleNamespace(
            name=policy["sequence_packing"].get("algorithm"),
            options=SimpleNamespace(
                max_sequence_length=data["max_input_seq_length"],
                sequence_length_pad_multiple=policy[
                    "make_sequence_length_divisible_by"
                ],
            ),
        )
    data["energon"].task_encoder = SimpleNamespace(packing=packing)
    return SimpleNamespace(
        sft=SimpleNamespace(**sft),
        data=data,
        data_plane=LocalDataPlaneConfig(),
        policy=policy,
        checkpointing={"metric_name": metric_name},
    )


def test_train_step_orders_split_policy_lifecycle_and_commit() -> None:
    controller = _controller()
    controller._load_envelopes.return_value = [
        _envelope(0, source_count=2),
        _envelope(1),
    ]

    metrics = controller._run_train_step()

    controller._trainer.begin_train_step.assert_called_once_with(controller._loss_fn)
    controller._trainer.train_placed_microbatches.assert_called_once()
    controller._trainer.finish_train_step.assert_called_once_with()
    controller._loader_call.assert_called_once_with("commit_sft_batch")
    assert controller._save_state.total_steps == 1
    assert controller._save_state.consumed_samples == 3
    assert metrics["valid_tokens"] == 8
    assert metrics["source_samples"] == 3
    assert metrics["physical_packs"] == 2
    assert {row.logger_key for row in SFT_V2_TEED_METRICS} <= set(metrics)


def test_mimo_train_step_counts_canonical_batch_once() -> None:
    controller = _controller()
    controller._trainer.mimo_config = SimpleNamespace(single_owner_loader=True)
    controller._placement_plan = SimpleNamespace(
        logical_world_size=1,
        copies=(
            LoaderCopy(
                logical_rank=0,
                logical_world_size=1,
                owner_rank=0,
                delivery_ranks=(0,),
                role="encoder",
                role_delivery_ranks={"language": (4, 5, 6, 7)},
                encoder_shard=0,
            ),
        ),
    )
    encoder = replace(
        _envelope(0, source_count=2), logical_world_size=1, mimo_role="encoder"
    )
    language_meta = replace(
        encoder.meta, partition_id="language_partition", fields=["input_ids"]
    )
    controller._load_envelopes.return_value = [
        replace(encoder, role_metas={"language": language_meta})
    ]
    metrics = controller._run_train_step()
    assert controller._save_state.consumed_samples == 2
    assert controller._save_state.total_valid_tokens == 4
    assert metrics["valid_tokens"] == 4
    assert metrics["source_samples"] == 2
    assert metrics["physical_packs"] == 1
    call = controller._trainer.train_placed_microbatches.call_args
    encoder_meta, language_meta = call.args[0]
    assert language_meta.partition_id == "language_partition"
    assert encoder_meta.partition_id != language_meta.partition_id
    assert encoder_meta.sample_ids == language_meta.sample_ids
    assert call.kwargs["delivery_ranks"] == [(0,), (4, 5, 6, 7)]
    controller._loader_call.assert_called_once_with("commit_sft_batch")


def test_mimo_train_step_with_encoder_shards_counts_each_lane_once() -> None:
    controller = _controller()
    controller._trainer.mimo_config = SimpleNamespace(single_owner_loader=True)
    controller._placement_plan = SimpleNamespace(
        logical_world_size=1,
        copies=tuple(
            LoaderCopy(
                logical_rank=0,
                logical_world_size=1,
                owner_rank=shard,
                delivery_ranks=(shard,),
                role="encoder",
                role_delivery_ranks={"language": (4, 5, 6, 7)} if shard == 0 else {},
                encoder_shard=shard,
            )
            for shard in range(2)
        ),
    )
    encoder = replace(
        _envelope(0, source_count=2), logical_world_size=1, mimo_role="encoder"
    )
    language_meta = replace(
        encoder.meta, partition_id="language_partition", fields=["input_ids"]
    )
    shard0 = replace(encoder, role_metas={"language": language_meta})
    shard1 = replace(encoder, meta=replace(encoder.meta, partition_id="shard1"))
    controller._load_envelopes.return_value = [shard0, shard1]
    metrics = controller._run_train_step()
    assert controller._save_state.consumed_samples == 2
    assert metrics["valid_tokens"] == 4
    assert metrics["physical_packs"] == 1
    call = controller._trainer.train_placed_microbatches.call_args
    assert [meta.partition_id for meta in call.args[0]] == [
        encoder.meta.partition_id,
        "shard1",
        "language_partition",
    ]
    assert call.kwargs["delivery_ranks"] == [(0,), (1,), (4, 5, 6, 7)]


def test_train_step_aborts_policy_and_loader_on_training_failure() -> None:
    controller = _controller()
    controller._trainer.train_placed_microbatches.side_effect = RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        controller._run_train_step()

    controller._trainer.abort_train_step.assert_called_once_with()
    controller._loader_call.assert_any_call("abort_sft_batch")
    assert controller._save_state.total_steps == 0


def test_validation_counts_source_samples_before_packing(monkeypatch: Any) -> None:
    controller = _controller()
    controller._master_config.sft = SimpleNamespace(
        val_batches=1,
        only_unmask_final=True,
        val_micro_batch_size=1,
    )
    controller._master_config.policy = {"make_sequence_length_divisible_by": 1}
    envelopes = [_envelope(0, source_count=3), _envelope(1, source_count=2)]
    controller._trainer.worker_group.run_all_workers_single_data.return_value = object()
    controller._trainer.evaluate_placed_microbatches.return_value = {
        "loss": 2.0,
        "grad_norm": 0.0,
        "all_mb_metrics": {
            "num_unmasked_tokens": [8],
            # Physical packed rows, deliberately smaller than the five sources.
            "num_valid_samples": [2],
            "answer_nll_sum": [3.0],
            "answer_exact_match_count": [1],
            "num_answer_tokens": [2],
            "num_answer_sequences": [2],
        },
    }
    monkeypatch.setattr("nemo_rl.algorithms.sft_v2.ray.get", lambda _: envelopes)

    metrics = controller._run_validation()

    assert metrics["num_valid_samples"] == 5
    assert metrics["loss"] == 2.0
    assert metrics["answer_token_ce"] == 1.5
    assert metrics["answer_exact_match_accuracy"] == 0.5


def _save_controller(**checkpointing: Any) -> object:
    controller = _controller()
    controller._master_config.checkpointing = {
        "enabled": True,
        "save_period": 10,
        "metric_name": None,
        **checkpointing,
    }
    controller._max_steps = 25
    return controller


def test_should_save_honors_save_period_ft_period_and_timeout() -> None:
    controller = _save_controller(ft_save_period=4)

    controller._save_state.total_steps = 3
    assert not controller._should_save(save_by_timeout=False)
    # A timeout save has to land between save_period boundaries, or a preempted
    # run loses everything since the last periodic save.
    assert controller._should_save(save_by_timeout=True)

    for step in (4, 10, 25):  # ft_save_period, save_period, final step
        controller._save_state.total_steps = step
        assert controller._should_save(save_by_timeout=False)


def test_should_save_is_disabled_by_the_checkpointing_flag() -> None:
    controller = _save_controller(enabled=False)
    controller._save_state.total_steps = 10

    assert not controller._should_save(save_by_timeout=True)


def test_run_stops_after_a_timeout_checkpoint() -> None:
    controller = _save_controller()
    controller._max_steps = 5
    controller._logger = MagicMock()
    controller._checkpointer = MagicMock()
    controller._close_loaders = MagicMock()
    controller._save_checkpoint = MagicMock()
    controller._timeout = MagicMock()
    controller._timeout.check_save.side_effect = [False, True]

    def advance() -> dict[str, Any]:
        controller._save_state.total_steps += 1
        return {}

    controller._run_train_step = MagicMock(side_effect=advance)
    with patch("nemo_rl.algorithms.sft_v2.shutdown_telemetry") as shutdown:
        controller.run()

    # check_save latches after firing once, so the loop must exit instead of
    # training unsaved until the walltime kill.
    assert controller._run_train_step.call_count == 2
    controller._save_checkpoint.assert_called_once_with({})
    shutdown.assert_called_once_with()


def test_loader_dispatch_carries_the_controller_trace() -> None:
    controller = _controller()
    controller._master_config = SimpleNamespace(
        sft=SimpleNamespace(only_unmask_final=False),
        policy={"make_sequence_length_divisible_by": 1},
    )
    controller._placement_plan = SimpleNamespace(logical_world_size=2, copies=[0, 1])
    controller._trainer.worker_group.run_all_workers_single_data.return_value = [
        object(),
        object(),
    ]
    carrier = {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}

    with (
        patch(
            "nemo_rl.algorithms.sft_v2.trace_context_kwargs",
            return_value={TRACE_CARRIER_KWARG: carrier},
        ),
        patch(
            "nemo_rl.algorithms.sft_v2.ray.get",
            return_value=[_envelope(0), _envelope(1)],
        ),
    ):
        _ACTOR_CLS._load_envelopes(controller)

    kwargs = (
        controller._trainer.worker_group.run_all_workers_single_data.call_args.kwargs
    )
    assert kwargs[TRACE_CARRIER_KWARG] == carrier


@pytest.mark.parametrize(("step", "is_final"), [(10, False), (25, True)])
def test_save_checkpoint_uses_policy_signature_and_terminal_step(
    tmp_path: Path, step: int, is_final: bool
) -> None:
    controller = _save_controller()
    controller._save_state.total_steps = step
    # Enforce the public signature so removed or missing keywords fail this test.
    controller._trainer = create_autospec(Policy, instance=True)
    controller._placement_plan = SimpleNamespace(logical_world_size=2, copies=[0, 1])
    controller._loader_state_dicts = MagicMock(return_value=[{}, {}])
    controller._checkpointer = MagicMock()
    controller._checkpointer.init_tmp_checkpoint.return_value = str(tmp_path)
    controller._checkpointer.save_optimizer = True

    controller._save_checkpoint({})

    controller._trainer.save_checkpoint.assert_called_once_with(
        weights_path=str(tmp_path / "policy" / "weights"),
        optimizer_path=str(tmp_path / "policy" / "optimizer"),
        tokenizer_path=str(tmp_path / "policy" / "tokenizer"),
        is_final_checkpoint=is_final,
    )


def test_checkpoint_metric_tags_the_configured_train_metric() -> None:
    controller = _save_controller(metric_name="train:loss")

    assert controller._checkpoint_metric({"loss": 1.5, "grad_norm": 0.5}) == {
        "train:loss": 1.5
    }
    # CheckpointManager looks the value up under the full prefixed name.
    assert _save_controller()._checkpoint_metric({"loss": 1.5}) == {}


def test_checkpoint_metric_rejects_a_metric_no_step_produces() -> None:
    controller = _save_controller(metric_name="train:accuracy")

    # Every step reports the same keys, so a name that misses once misses
    # always -- fail on step 1 rather than at the first checkpoint.
    with pytest.raises(ValueError, match="did not produce"):
        controller._checkpoint_metric({"loss": 1.5})


@pytest.mark.parametrize(
    ("config_overrides", "message"),
    [
        ({"data_overrides": {"backend": "hf"}}, "requires data.backend=energon"),
        (
            {"policy_overrides": {"megatron_cfg": {"enabled": False}}},
            "only the Megatron policy",
        ),
        (
            {"policy_overrides": {"sequence_packing": {"enabled": True}}},
            "fixed batching",
        ),
        (
            {"policy_overrides": {"dynamic_batching": {"enabled": True}}},
            "fixed batching",
        ),
        (
            {"sft_overrides": {"val_period": 10}},
            "data.validation is null",
        ),
        (
            {"data_overrides": {"max_input_seq_length": None}},
            "max_input_seq_length",
        ),
    ],
)
def test_setup_rejects_configs_sft_v2_cannot_run(
    config_overrides: dict[str, Any], message: str
) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    config = _valid_setup_config(**config_overrides)

    with pytest.raises(ValueError, match=message):
        setup_sft_v2(config, MagicMock())


def test_setup_rejects_a_validation_checkpoint_metric() -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    with pytest.raises(ValueError, match="training metrics only"):
        setup_sft_v2(
            _valid_setup_config(metric_name="val:val_loss"),
            MagicMock(),
        )


@pytest.mark.parametrize(
    (
        "sequence_overrides",
        "dynamic_enabled",
        "megatron_overrides",
        "pad_multiple",
        "max_sequence_length",
        "message",
    ),
    [
        (
            {"fuse_loss": False},
            False,
            {},
            1,
            128,
            "Energon packing requires sequence_packing enabled with fuse_loss.",
        ),
        (
            {"algorithm": "unknown"},
            False,
            {},
            1,
            128,
            "Energon SFT requires a supported packing algorithm.",
        ),
        (
            {},
            True,
            {},
            1,
            128,
            "Energon packing does not support dynamic batching.",
        ),
        (
            {},
            False,
            {},
            4,
            130,
            "Energon packing requires max_input_seq_length to be divisible by make_sequence_length_divisible_by.",
        ),
        (
            {},
            False,
            {"tensor_model_parallel_size": 2, "sequence_parallel": True},
            1,
            128,
            "Energon packing requires make_sequence_length_divisible_by to be a multiple of 2.",
        ),
    ],
)
def test_setup_rejects_invalid_energon_packing_config(
    sequence_overrides: dict[str, Any],
    dynamic_enabled: bool,
    megatron_overrides: dict[str, Any],
    pad_multiple: int,
    max_sequence_length: int,
    message: str,
) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    sequence_packing = {
        "enabled": True,
        "fuse_loss": True,
        "algorithm": "greedy_knapsack",
    }
    sequence_packing.update(sequence_overrides)
    config = _valid_setup_config(
        data_overrides={
            "max_input_seq_length": max_sequence_length,
            "energon": SimpleNamespace(packing_buffer_size=64),
        },
        policy_overrides={
            "megatron_cfg": megatron_overrides,
            "sequence_packing": sequence_packing,
            "dynamic_batching": {"enabled": dynamic_enabled},
        },
    )
    config.policy["make_sequence_length_divisible_by"] = pad_multiple
    config.data[
        "energon"
    ].task_encoder.packing.options.sequence_length_pad_multiple = pad_multiple

    with pytest.raises(ValueError, match=re.escape(message)):
        setup_sft_v2(config, MagicMock())


def test_setup_loaders_enables_packing_from_energon_buffer() -> None:
    controller = object.__new__(_ACTOR_CLS)
    controller._master_config = SimpleNamespace(
        data={
            "max_input_seq_length": 128,
            "energon": SimpleNamespace(packing_buffer_size=64),
        },
        policy={
            "train_global_batch_size": 4,
            "sequence_packing": {
                "algorithm": "balanced_greedy_knapsack",
                "max_sequences_per_bin": 16,
            },
            "make_sequence_length_divisible_by": 8,
        },
        sft=SimpleNamespace(only_unmask_final=False),
    )
    controller._placement_plan = SimpleNamespace(
        logical_world_size=2,
        copies=[0, 1],
        placement_hash="placement",
    )
    controller._loader_states = None
    controller._trainer = MagicMock(mimo_config=None)
    futures = [object(), object()]
    controller._trainer.worker_group.run_all_workers_single_data.return_value = futures

    with patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[True, True]):
        controller._setup_loaders()

    kwargs = (
        controller._trainer.worker_group.run_all_workers_single_data.call_args.kwargs
    )
    assert kwargs["data_config"] is controller._master_config.data
    assert "packing_algorithm" not in kwargs
    assert "max_sequences_per_bin" not in kwargs


@pytest.mark.parametrize(
    ("megatron_overrides", "policy_multiple", "message"),
    [
        ({"context_parallel_size": 2}, 2, "multiple of 4"),
        (
            {
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "hybridep",
            },
            1,
            "HybridEP",
        ),
        (
            {"fp8_cfg": {"enabled": True, "fp8_recipe": "blockwise"}},
            1,
            "FP8 packed-token alignment",
        ),
    ],
)
def test_setup_rejects_unsupported_energon_packing_layouts(
    megatron_overrides: dict[str, Any], policy_multiple: int, message: str
) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    config = _valid_setup_config(
        data_overrides={
            "max_input_seq_length": 130,
            "energon": SimpleNamespace(packing_buffer_size=64),
        },
        policy_overrides={
            "megatron_cfg": megatron_overrides,
            "sequence_packing": {
                "enabled": True,
                "fuse_loss": True,
                "algorithm": "greedy_knapsack",
            },
        },
    )
    config.policy["make_sequence_length_divisible_by"] = policy_multiple

    with pytest.raises(ValueError, match=message):
        setup_sft_v2(config, MagicMock())


class _ClusterCreated(Exception):
    pass


def _het_setup_config(*, segment_size: int | None = None) -> SimpleNamespace:
    config = _valid_setup_config()
    config.cluster = ClusterConfig(
        num_nodes=2, gpus_per_node=4, segment_size=segment_size
    )
    config.policy["tokenizer"] = {}
    return config


def _patch_het_cluster(monkeypatch: Any, created: dict[str, Any]) -> None:
    def create_cluster(**kwargs: Any) -> None:
        created.update(kwargs)
        raise _ClusterCreated

    probe = MagicMock()
    probe.get_latest_checkpoint_path.return_value = None
    probe.get_resume_paths.return_value = (None, None)
    monkeypatch.setattr(
        "nemo_rl.algorithms.sft_v2.CheckpointManager", MagicMock(return_value=probe)
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.sft_v2.get_node_block_counts",
        lambda: {0: 1, 1: 1},
    )
    monkeypatch.setattr("nemo_rl.algorithms.sft_v2.RayVirtualCluster", create_cluster)


def test_setup_pins_nodes_to_node_blocks(monkeypatch: Any) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    created: dict[str, Any] = {}
    _patch_het_cluster(monkeypatch, created)

    with pytest.raises(_ClusterCreated):
        setup_sft_v2(_het_setup_config(), MagicMock())

    assert created["node_resource_constraints"] == [
        {"node_block_0": 0.001},
        {"node_block_1": 0.001},
    ]


def test_setup_rejects_segment_size_with_node_blocks(monkeypatch: Any) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    created: dict[str, Any] = {}
    _patch_het_cluster(monkeypatch, created)

    with pytest.raises(ValueError, match="segment_size must be null"):
        setup_sft_v2(_het_setup_config(segment_size=1), MagicMock())
    assert created == {}


def test_restore_rejects_changed_placement() -> None:
    from nemo_rl.algorithms.sft_v2 import _restore_save_state

    with pytest.raises(ValueError, match="placement changed"):
        _restore_save_state(
            {"placement_hash": "old", "total_steps": 3}, placement_hash="new"
        )


def test_mimo_topology_mapper_accepts_only_default() -> None:
    assert _validate_mimo_topology_mapper("default") is None

    with pytest.raises(ValueError, match="topology_mapper='default'"):
        _validate_mimo_topology_mapper("custom")


def test_owner_results_follow_placement_order() -> None:
    controller = object.__new__(_ACTOR_CLS)
    controller._trainer = SimpleNamespace(mimo_config=object())
    controller._train_cluster = MagicMock()
    controller._train_cluster.world_size.return_value = 5
    controller._placement_plan = SimpleNamespace(
        copies=(SimpleNamespace(owner_rank=4), SimpleNamespace(owner_rank=0))
    )

    ordered = controller._ordered_owner_results(
        ["encoder", None, None, None, "language"], operation="test"
    )

    assert ordered == ["language", "encoder"]


def test_ordinary_loader_rpc_still_targets_replica_leaders() -> None:
    controller = _controller()
    controller._trainer.worker_group = MagicMock()
    controller._loader_call = _ACTOR_CLS._loader_call.__get__(controller)
    with patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[None, None]):
        controller._loader_call("commit_sft_batch")
    from nemo_rl.distributed.named_sharding import REPLICATED_AXES

    controller._trainer.worker_group.run_all_workers_single_data.assert_called_once_with(
        "commit_sft_batch", run_rank_0_only_axes=list(REPLICATED_AXES)
    )


def test_mimo_gtp2_rejected_before_worker_creation() -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    config = _valid_setup_config(
        policy_overrides={
            "megatron_cfg": {
                "mimo": {"parallelism": {"module_parallelisms": {}}},
                "model_overrides": {"gtp_weight_remat_size": 2},
            }
        }
    )
    config.data_plane = {"impl": "transfer_queue"}
    config.data["energon"].topology_mapper = "default"
    config.data["energon"].task_encoder.packing = object()
    config.policy["train_micro_batch_size"] = 1
    with patch("nemo_rl.algorithms.sft_v2.RayVirtualCluster") as cluster:
        with pytest.raises(
            NotImplementedError, match="requires Bridge MIMO GTP support"
        ):
            setup_sft_v2(config, MagicMock())
        cluster.assert_not_called()


@pytest.mark.parametrize(("num_nodes", "language_ranks"), [(5, 16), (17, 64)])
def test_mimo_rank_spans_allow_one_vision_gpu_on_last_node(num_nodes, language_ranks):
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    parallelism = {
        "module_parallelisms": {
            "language": {
                "tensor_model_parallel_size": 1,
                "context_parallel_size": language_ranks,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": 1,
                "rank_offset": 0,
            },
            "images": {
                "tensor_model_parallel_size": 1,
                "context_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": 1,
                "rank_offset": language_ranks,
            },
        }
    }
    config = _valid_setup_config(
        sft_overrides={"max_num_steps": 1, "max_num_epochs": 1},
        policy_overrides={
            "megatron_cfg": {
                "context_parallel_size": language_ranks,
                "mimo": {"parallelism": parallelism},
            },
            "sequence_packing": {"enabled": True, "fuse_loss": True},
        },
    )
    config.data_plane = {"impl": "transfer_queue"}
    config.data["energon"].topology_mapper = "default"
    config.data["energon"].task_encoder.packing = SimpleNamespace(
        name="greedy_knapsack",
        options=SimpleNamespace(
            max_sequence_length=128, sequence_length_pad_multiple=128
        ),
    )
    config.data["train"] = {
        "path": "/dataset",
        "split": "train",
        "virtual_epoch_length": 1,
    }
    config.policy.update(
        {
            "train_micro_batch_size": 1,
            "max_total_sequence_length": 128,
            "make_sequence_length_divisible_by": 128,
            "tokenizer": {},
        }
    )
    config.cluster = ClusterConfig(
        num_nodes=num_nodes, gpus_per_node=4, segment_size=None
    )
    checkpointer = MagicMock()
    checkpointer.get_resume_paths.return_value = (None, None)
    with (
        patch("nemo_rl.algorithms.sft_v2.CheckpointManager", return_value=checkpointer),
        patch("nemo_rl.algorithms.sft_v2.RayVirtualCluster") as cluster,
        patch(
            "nemo_rl.algorithms.sft_v2.TQPolicy",
            side_effect=RuntimeError("stop before weights"),
        ),
    ):
        with pytest.raises(RuntimeError, match="stop before weights"):
            setup_sft_v2(config, MagicMock())
    assert cluster.call_args.kwargs["bundle_ct_per_node_list"] == [4] * (
        num_nodes - 1
    ) + [1]


def test_max_steps_is_bounded_by_virtual_epochs() -> None:
    config = SimpleNamespace(
        sft=SimpleNamespace(max_num_steps=100, max_num_epochs=3),
        data={
            "train": {
                "path": "/dataset",
                "split": "train",
                "virtual_epoch_length": 7,
            }
        },
    )

    assert _max_train_steps(config) == 21


@pytest.mark.parametrize(
    "pretrained", [None, {"format": "megatron_bridge", "path": "/checkpoint"}]
)
def test_setup_propagates_pretrained_checkpoint_before_validation(pretrained):
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    config = _valid_setup_config(data_overrides={"backend": "invalid"})
    config.checkpointing["pretrained_checkpoint"] = pretrained
    with pytest.raises(ValueError, match="data.backend=energon"):
        setup_sft_v2(config, MagicMock())
    assert config.policy.get("pretrained_checkpoint") == pretrained


@pytest.mark.parametrize("dispatch_padding", [None, False, True])
def test_setup_requires_uneven_dispatch_padding_for_packed_hybridep(
    dispatch_padding: bool | None,
) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    overrides = (
        {}
        if dispatch_padding is None
        else {"moe_hybridep_pad_uneven_dispatch_inputs": dispatch_padding}
    )
    config = _valid_setup_config(
        data_overrides={
            "max_input_seq_length": 130,
            "energon": SimpleNamespace(packing_buffer_size=64),
        },
        policy_overrides={
            "megatron_cfg": {
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "hybridep",
                "model_overrides": overrides,
            },
            "sequence_packing": {
                "enabled": True,
                "fuse_loss": True,
                "algorithm": "greedy_knapsack",
            },
        },
    )
    config.policy["make_sequence_length_divisible_by"] = 4
    config.data["energon"].task_encoder.packing.options.sequence_length_pad_multiple = 4
    # A padded HybridEP request reaches the existing alignment check after this gate.
    message = (
        "max_input_seq_length to be divisible"
        if dispatch_padding
        else "moe_hybridep_pad_uneven_dispatch_inputs=true"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        setup_sft_v2(config, MagicMock())


@pytest.mark.parametrize(
    "difference",
    [None, "source_ids", "pack_groups", "sequence_lengths", "valid_tokens"],
)
def test_independent_loaders_reject_batch_divergence(difference: str | None) -> None:
    controller = _controller()
    controller._trainer.mimo_config = SimpleNamespace(single_owner_loader=False)
    controller._placement_plan = SimpleNamespace(
        logical_world_size=1,
        copies=(
            LoaderCopy(
                logical_rank=0,
                logical_world_size=1,
                owner_rank=0,
                delivery_ranks=(0,),
                role="encoder",
                encoder_shard=0,
            ),
            LoaderCopy(
                logical_rank=0,
                logical_world_size=1,
                owner_rank=4,
                delivery_ranks=(4, 5),
                role="language",
            ),
        ),
    )
    controller._master_config.policy = {"make_sequence_length_divisible_by": 1}
    controller._master_config.sft.only_unmask_final = False
    controller._ordered_owner_results = MagicMock(
        side_effect=lambda results, **kw: results
    )
    encoder = replace(_envelope(0), mimo_role="encoder")
    language = replace(encoder, mimo_role="language")
    if difference == "pack_groups":
        language = replace(
            language, meta=replace(language.meta, tags=[{"source_ids": ["other"]}])
        )
    elif difference is not None:
        language = replace(
            language,
            **{
                difference: ("other",)
                if difference == "source_ids"
                else (99,)
                if difference == "sequence_lengths"
                else 99
            },
        )
    with patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[encoder, language]):
        if difference is not None:
            with pytest.raises(RuntimeError, match="Independent MiMo loaders diverged"):
                _ACTOR_CLS._load_envelopes(controller)
        else:
            assert _ACTOR_CLS._load_envelopes(controller) == [encoder, language]
    controller._trainer.begin_train_step.assert_not_called()


def test_independent_loader_accounting_counts_each_lane_once() -> None:
    controller = _controller()
    controller._trainer.mimo_config = SimpleNamespace(single_owner_loader=False)
    controller._placement_plan = SimpleNamespace(
        copies=[
            SimpleNamespace(delivery_ranks=(0,), role_delivery_ranks={}),
            SimpleNamespace(delivery_ranks=(4, 5, 6, 7), role_delivery_ranks={}),
        ]
    )
    encoder = replace(_envelope(0), mimo_role="encoder")
    language = replace(encoder, mimo_role="language")
    controller._load_envelopes.return_value = [encoder, language]
    metrics = controller._run_train_step()
    assert controller._save_state.consumed_samples == len(language.source_ids)
    assert metrics["valid_tokens"] == language.valid_tokens
    assert metrics["physical_packs"] == len(language.meta.sample_ids)
    call = controller._trainer.train_placed_microbatches.call_args
    assert call.args[0] == [encoder.meta, language.meta]
    assert call.kwargs["delivery_ranks"] == [(0,), (4, 5, 6, 7)]


def test_independent_loader_resume_restores_both_owners_per_lane() -> None:
    controller = _controller()
    controller._trainer.mimo_config = SimpleNamespace(single_owner_loader=False)
    controller._train_cluster = MagicMock()
    controller._train_cluster.world_size.return_value = 4
    controller._master_config.data = {"max_input_seq_length": 8, "validation": None}
    controller._master_config.policy = {"train_global_batch_size": 2}
    controller._master_config.sft.only_unmask_final = False
    controller._placement_plan = SimpleNamespace(
        logical_world_size=2,
        placement_hash="independent",
        copies=[SimpleNamespace(owner_rank=rank) for rank in [0, 2, 1, 3]],
    )
    states = controller._loader_states = [{"state": rank} for rank in [0, 2, 1, 3]]
    with patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[True] * 4):
        controller._setup_loaders()
    call = controller._trainer.worker_group.run_all_workers_multiple_data.call_args
    assert call.kwargs["restored_state"] == [states[0], states[2], states[1], states[3]]
    assert call.kwargs["common_kwargs"]["batch_size"] == 1
    assert call.kwargs["common_kwargs"]["single_owner"] is False
