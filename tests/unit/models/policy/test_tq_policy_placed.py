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

import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import ray

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    GLOBAL_FORWARD_PAD_SEQLEN,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
)
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.named_sharding import GTP_WEIGHT_REMAT_AXIS
from nemo_rl.models.megatron.mimo import MIMOSFTConfig
from nemo_rl.models.policy.tq_policy import TQPolicy


def _meta(rank: int, fields: list[str]) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="loader",
        sample_ids=[f"rank-{rank}-sample-0", f"rank-{rank}-sample-1"],
        fields=fields,
        sequence_lengths=[8 + rank, 16 + 8 * rank],
        extra_info={"logical_dp_rank": rank},
    )


def _policy() -> tuple[TQPolicy, MagicMock]:
    policy = object.__new__(TQPolicy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.mimo_config = None
    policy._router_replay_enabled = False
    policy._opd_full_field = None  # opd_full off, as __init__ leaves it
    policy.flops_tracker = None
    policy._queued_finish_refs = None
    policy.sharding_annotations = MagicMock()
    policy.sharding_annotations.names = [
        "pipeline_parallel",
        "data_parallel",
        "context_parallel",
        "tensor_parallel",
    ]
    policy.sharding_annotations.get_axis_size.return_value = 2
    worker_group = MagicMock()
    worker_group.cluster.world_size.return_value = 8
    policy.worker_group = worker_group
    return policy, worker_group


def test_train_placed_microbatches_routes_mimo_roles_and_empty_helpers() -> None:
    policy, worker_group = _policy()
    policy.mimo_config = MIMOSFTConfig(parallelism={})
    metas = [_meta(0, ["input_ids", "pixel_values"]), _meta(0, ["input_ids"])]
    stamped = policy._stamp_placed_pad_seqlen(metas)
    with patch("nemo_rl.models.policy.tq_policy.ray.get") as get:
        policy.train_placed_microbatches(metas, delivery_ranks=[(0,), (4, 5, 6, 7)])
    dispatch = worker_group.run_all_workers_multiple_data.call_args
    encoder, language = [replace(meta, task_name="train") for meta in stamped]
    assert dispatch.args[0] == "train_microbatch_presharded"
    assert dispatch.kwargs["meta"] == [
        encoder,
        None,
        None,
        None,
        language,
        language,
        language,
        language,
    ]
    assert dispatch.kwargs["common_kwargs"] == {
        "num_microbatches": len(metas[0].sample_ids),
        "sequence_length": stamped[0].extra_info[GLOBAL_FORWARD_PAD_SEQLEN],
    }
    get.assert_called_once_with(worker_group.run_all_workers_multiple_data.return_value)


def test_train_placed_microbatches_keeps_fields_and_replica_delivery() -> None:
    policy, worker_group = _policy()
    dp_metas = [
        _meta(0, ["input_ids", "pixel_values"]),
        _meta(1, ["input_ids", "image_grid_thw"]),
    ]

    phases = policy.train_placed_microbatches(dp_metas)

    assert set(phases) == {"stamp_pad", "dispatch"}
    assert all(value >= 0.0 for value in phases.values())

    dispatch = worker_group.run_all_workers_sharded_data.call_args
    dispatched = dispatch.kwargs["meta"]
    assert [meta.fields for meta in dispatched] == [
        ["input_ids", "pixel_values"],
        ["input_ids", "image_grid_thw"],
    ]
    assert [meta.task_name for meta in dispatched] == ["train", "train"]
    assert [meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] for meta in dispatched] == [
        24,
        24,
    ]

    assert dispatch.args[0] == "train_microbatch_presharded"
    assert dispatch.kwargs["in_sharded_axes"] == ["data_parallel"]
    assert dispatch.kwargs["replicate_on_axes"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    assert dispatch.kwargs["output_is_replicated"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    worker_group.get_all_worker_results.assert_called_once()


def test_train_placed_microbatches_requires_one_batch_per_logical_rank() -> None:
    policy, worker_group = _policy()

    with pytest.raises(ValueError, match="one batch per logical data rank"):
        policy.train_placed_microbatches([_meta(0, ["input_ids"])])

    worker_group.run_all_workers_sharded_data.assert_not_called()


def test_train_placed_microbatches_shards_over_dp_and_gtp() -> None:
    policy, worker_group = _policy()
    policy.sharding_annotations.names.insert(2, GTP_WEIGHT_REMAT_AXIS)
    policy.sharding_annotations.get_axis_size.side_effect = {
        "data_parallel": 2,
        GTP_WEIGHT_REMAT_AXIS: 2,
    }.__getitem__
    dp_metas = [_meta(rank, ["input_ids"]) for rank in range(4)]

    policy.train_placed_microbatches(dp_metas)

    dispatch = worker_group.run_all_workers_sharded_data.call_args
    assert dispatch.kwargs["in_sharded_axes"] == [
        "data_parallel",
        GTP_WEIGHT_REMAT_AXIS,
    ]
    assert [
        [meta.extra_info["logical_dp_rank"] for meta in dp_group]
        for dp_group in dispatch.kwargs["meta"]
    ] == [[0, 1], [2, 3]]


def test_train_placed_microbatches_rejects_dynamic_batching() -> None:
    policy, worker_group = _policy()
    policy.use_dynamic_batches = True
    policy.dynamic_batching_args = {}
    policy.cfg["dynamic_batching"] = {"train_mb_tokens": 4096}

    with pytest.raises(ValueError, match="dynamic batching"):
        policy.train_placed_microbatches(
            [_meta(0, ["input_ids"]), _meta(1, ["input_ids"])]
        )

    worker_group.run_all_workers_sharded_data.assert_not_called()


def test_train_placed_microbatches_requires_producer_packing_shapes() -> None:
    policy, worker_group = _policy()
    policy.use_sequence_packing = True
    policy.sequence_packing_args = {"algorithm": "modified_first_fit_decreasing"}
    policy.cfg["sequence_packing"] = {"train_mb_tokens": 4096}

    with pytest.raises(ValueError, match="producer microbatch shapes"):
        policy.train_placed_microbatches(
            [_meta(0, ["input_ids"]), _meta(1, ["input_ids"])]
        )

    worker_group.run_all_workers_sharded_data.assert_not_called()


def test_train_placed_microbatches_accepts_producer_packing_shapes() -> None:
    policy, worker_group = _policy()
    policy.use_sequence_packing = True
    policy.sequence_packing_args = {"algorithm": "modified_first_fit_decreasing"}
    policy.cfg["sequence_packing"] = {"train_mb_tokens": 4096}
    dp_metas = [_meta(0, ["input_ids"]), _meta(1, ["input_ids"])]
    for meta in dp_metas:
        meta.extra_info[MICRO_BATCH_INDICES] = [[[0, 1], [1, 2]]]
        meta.extra_info[MICRO_BATCH_LENGTHS] = [[8, 16]]

    policy.train_placed_microbatches(dp_metas)

    dispatched = worker_group.run_all_workers_sharded_data.call_args.kwargs["meta"]
    assert [meta.extra_info[MICRO_BATCH_INDICES] for meta in dispatched] == [
        [[[0, 1], [1, 2]]],
        [[[0, 1], [1, 2]]],
    ]
    assert [meta.extra_info[MICRO_BATCH_LENGTHS] for meta in dispatched] == [
        [[8, 16]],
        [[8, 16]],
    ]
    assert [meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] for meta in dispatched] == [
        24,
        24,
    ]
    assert [meta.task_name for meta in dispatched] == ["train", "train"]


def test_train_microbatches_gtp_rejects_dp_only_dispatch() -> None:
    policy, worker_group = _policy()
    policy.sharding_annotations.names.insert(2, GTP_WEIGHT_REMAT_AXIS)
    with pytest.raises(ValueError, match="producer-placed metadata"):
        policy._dispatch_train_microbatches([_meta(0, ["input_ids"])], timer=None)
    worker_group.run_all_workers_sharded_data.assert_not_called()


@ray.remote(num_cpus=0)
class _StepWorker(TQWorkerMixin):
    """Asyncio actor like the policy workers; records when its step calls end."""

    def __init__(self, train_seconds: float = 0.0, fail_train: bool = False) -> None:
        self._train_seconds = train_seconds
        self._fail_train = fail_train
        self._events: dict[str, float] = {}

    async def events(self) -> dict[str, float]:
        return self._events

    def _fetch(self, meta: KVBatchMeta) -> dict[str, Any]:
        return {}

    def _attach_or_repack_pack_metadata(
        self, data: dict[str, Any], meta: KVBatchMeta
    ) -> dict[str, Any]:
        return data

    def _is_replica_leader(self) -> bool:
        return True

    def train_microbatch(
        self, data: Any, num_microbatches: int, sequence_length: int
    ) -> None:
        time.sleep(self._train_seconds)
        if self._fail_train:
            raise RuntimeError("train failed")
        self._events["train"] = time.monotonic()

    def finish_train_step(self) -> dict[str, Any]:
        self._events["finish"] = time.monotonic()
        return {"global_loss": 1.0, "grad_norm": 0.5, "all_mb_metrics": {}}

    def abort_train_step(self) -> None:
        self._events["abort"] = time.monotonic()


class _ActorGroup:
    """The RayWorkerGroup fan-outs TQPolicy uses, over plain actors."""

    def __init__(self, workers: list[Any]) -> None:
        self.workers = workers
        self.cluster = SimpleNamespace(world_size=lambda: len(workers))

    def run_all_workers_multiple_data(
        self,
        method_name: str,
        *,
        common_kwargs: dict[str, Any] | None = None,
        **kwargs: list[Any],
    ) -> list[ray.ObjectRef]:
        return [
            getattr(worker, method_name).remote(
                **{key: values[rank] for key, values in kwargs.items()},
                **(common_kwargs or {}),
            )
            for rank, worker in enumerate(self.workers)
        ]

    def run_all_workers_single_data(
        self, method_name: str, **kwargs: Any
    ) -> list[ray.ObjectRef]:
        return [
            getattr(worker, method_name).remote(**kwargs) for worker in self.workers
        ]

    def shutdown(self, cleanup_method: str) -> bool:
        return True


def _mimo_policy(workers: list[Any], *, async_finish: bool) -> TQPolicy:
    """Policy over an encoder worker (rank 0) and a language worker (rank 1)."""
    policy, _ = _policy()
    policy.mimo_config = MIMOSFTConfig(
        parallelism={"module_parallelisms": {"language": {"rank_offset": 1}}},
        async_finish=async_finish,
    )
    policy.worker_group = _ActorGroup(workers)
    return policy


def _train_mimo_step(policy: TQPolicy) -> None:
    policy.train_placed_microbatches(
        [_meta(0, ["input_ids", "pixel_values"]), _meta(0, ["input_ids"])],
        delivery_ranks=[(0,), (1,)],
    )


@pytest.mark.parametrize("async_finish", [False, True])
def test_mimo_async_finish_waits_only_for_the_workers_own_train_call(
    async_finish: bool,
) -> None:
    encoder, language = _StepWorker.remote(), _StepWorker.remote(train_seconds=1.0)
    policy = _mimo_policy([encoder, language], async_finish=async_finish)
    # Start both actors first so that actor startup does not use up the 1 s margin.
    ray.get([encoder.events.remote(), language.events.remote()])

    _train_mimo_step(policy)
    result = policy.finish_train_step()

    encoder_events = ray.get(encoder.events.remote())
    language_events = ray.get(language.events.remote())
    assert list(encoder_events) == list(language_events) == ["train", "finish"]
    # Only with async finish does the encoder step while the language worker trains.
    assert (encoder_events["finish"] < language_events["train"]) is async_finish
    assert result["loss"] == 1.0


def test_mimo_async_finish_never_runs_after_a_failed_train_call() -> None:
    encoder, language = (
        _StepWorker.remote(train_seconds=0.5),
        _StepWorker.remote(fail_train=True),
    )
    policy = _mimo_policy([encoder, language], async_finish=True)
    ray.get([encoder.events.remote(), language.events.remote()])

    with pytest.raises(RuntimeError, match="train failed"):
        _train_mimo_step(policy)
    policy.abort_train_step()

    # The failed worker never finishes. The encoder finish is still queued when the
    # language train call fails; the abort runs after it.
    assert list(ray.get(language.events.remote())) == ["abort"]
    assert list(ray.get(encoder.events.remote())) == ["train", "finish", "abort"]
