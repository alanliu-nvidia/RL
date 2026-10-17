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

"""Runtime side of the MIMO encoder media split: loaders, controller, payload, setup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.mcore

from megatron.core.pipeline_parallel import bridge_communicator  # noqa: E402

from nemo_rl.algorithms.sft_v2 import (  # noqa: E402
    check_mimo_lanes_agree,
    mimo_accounting_envelopes,
)
from nemo_rl.data.energon.mimo_image_split import EncoderShard  # noqa: E402
from nemo_rl.data.energon.sft_types import StepEnvelope  # noqa: E402
from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker  # noqa: E402
from nemo_rl.data.energon.topology import (  # noqa: E402
    LoaderCopy,
    build_mimo_data_loader_placement,
)
from nemo_rl.data.multimodal_utils import (  # noqa: E402
    MIMO_MEDIA_OUTPUT_COUNTS_KEY,
    PackedTensor,
)
from nemo_rl.data_plane.interfaces import KVBatchMeta  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.models.megatron.data import ProcessedMicrobatch  # noqa: E402
from nemo_rl.models.megatron.setup import (  # noqa: E402
    _validate_mimo_encoder_media_split,
)
from nemo_rl.models.megatron.train import (  # noqa: E402
    _check_mimo_encoder_rows,
    _mimo_encoder_batch,
    _mimo_encoder_media_split_active,
)

PATCH = 14


# ── loader roles on the workers ──────────────────────────────────────────────


class _Group:
    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _module(dp: int, offset: int, cp: int = 1) -> dict[str, int]:
    return {
        "tensor_model_parallel_size": 1,
        "context_parallel_size": cp,
        "pipeline_model_parallel_size": 1,
        "data_parallel_size": dp,
        "rank_offset": offset,
    }


def _worker(*, module: str, dp_rank: int, dp_size: int, language_dp: int) -> Any:
    # The Ray actor wraps the worker class; build one without Ray or __init__.
    actor_class = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    worker = object.__new__(actor_class)
    encoder_dp = dp_size if module == "images" else 8
    worker.mimo_runtime = SimpleNamespace(
        active_module_name=module,
        local_pg_collection=SimpleNamespace(dp=_Group(dp_rank, dp_size)),
        megatron_mimo_infra=SimpleNamespace(
            module_to_grid_map={
                "language": SimpleNamespace(
                    dim_names=["tp", "cp", "dp", "pp"], shape=[1, 2, language_dp, 1]
                )
            }
        ),
    )
    worker._is_replica_leader = lambda: True
    worker._sft_encoder_dp_client = None
    worker._sft_single_owner = None
    parallelism = {
        "module_parallelisms": {
            "images": _module(encoder_dp, 0),
            "language": _module(language_dp, encoder_dp, cp=2),
        }
    }
    worker.cfg = {
        "megatron_cfg": {
            "mimo": {
                "parallelism": parallelism,
                "encoder_media_split_frame_weight": 7,
            }
        }
    }
    world_size = encoder_dp + 2 * language_dp
    # Global rank of this worker in the controller's placement (DP-major, TP=CP=PP=1).
    worker._test_global_rank = (
        dp_rank if module == "images" else encoder_dp + 2 * dp_rank
    )
    worker._test_world_size = world_size
    return worker


def _setup(worker: Any, *, global_rank: int | None = None) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    worker._configure_sft_dataloader = lambda **kwargs: captured.update(kwargs)
    rank = worker._test_global_rank if global_rank is None else global_rank
    with (
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.get_world_size", return_value=worker._test_world_size),
    ):
        assert worker.setup_sft_dataloader(
            data_config={},
            batch_size=1,
            max_sequence_length=8,
            placement_fingerprint="placement",
            only_unmask_final=False,
            single_owner=False,
        )
    return captured


@pytest.mark.parametrize(("dp_size", "language_dp"), [(8, 1), (8, 4), (4, 4)])
def test_every_encoder_replica_loads_its_lane_with_a_shard(
    dp_size: int, language_dp: int
) -> None:
    stride = dp_size // language_dp
    for dp_rank in range(dp_size):
        captured = _setup(
            _worker(
                module="images",
                dp_rank=dp_rank,
                dp_size=dp_size,
                language_dp=language_dp,
            )
        )
        assert captured["mimo_role"] == "encoder"
        assert captured["logical_rank"] == dp_rank // stride
        assert captured["logical_world_size"] == language_dp
        assert captured["encoder_shard"] == EncoderShard(
            index=dp_rank % stride, count=stride
        )


def test_encoder_rank_out_of_placement_order_fails_setup() -> None:
    # Encoder DP rank 1 running as global rank 2: the placement expects shard 0 of lane 1.
    worker = _worker(module="images", dp_rank=1, dp_size=8, language_dp=4)
    with pytest.raises(RuntimeError, match="not in the order of their global ranks"):
        _setup(worker, global_rank=2)


def test_language_loader_has_no_shard() -> None:
    captured = _setup(_worker(module="language", dp_rank=3, dp_size=4, language_dp=4))
    assert captured["mimo_role"] == "language"
    assert captured["logical_rank"] == 3
    assert captured["encoder_shard"] is None


def test_encoder_shards_name_partitions_apart_and_only_shard_zero_publishes() -> None:
    worker = _worker(module="images", dp_rank=0, dp_size=8, language_dp=1)
    worker._sft_mimo_role = "encoder"
    worker._sft_single_owner = True
    worker._sft_encoder_shard = EncoderShard(index=0, count=1)
    assert worker._sft_partition_role() == "_encoder"
    assert worker._sft_publishes_language_batch()
    worker._sft_encoder_shard = EncoderShard(index=3, count=8)
    assert worker._sft_partition_role() == "_encoder3"
    assert not worker._sft_publishes_language_batch()
    worker._sft_encoder_shard = EncoderShard(index=0, count=8)
    assert worker._sft_partition_role() == "_encoder0"
    assert worker._sft_publishes_language_batch()
    worker._sft_single_owner = False
    assert not worker._sft_publishes_language_batch()


def test_frame_weight_comes_from_the_mimo_config() -> None:
    worker = _worker(module="images", dp_rank=0, dp_size=2, language_dp=1)
    assert worker._encoder_media_split_frame_weight() == 7


# ── controller ───────────────────────────────────────────────────────────────


def _envelope(
    lane: int, role: str, *, sources: tuple[str, ...] = ("a",)
) -> StepEnvelope:
    return StepEnvelope(
        meta=KVBatchMeta(
            partition_id=f"{role}{lane}",
            task_name="train",
            sample_ids=["row0"],
            tags=[{"source_id": sources[0]}],
        ),
        logical_rank=lane,
        logical_world_size=2,
        source_ids=sources,
        field_names=("input_ids",),
        sequence_lengths=(16,),
        load_seconds=0.0,
        valid_tokens=10,
        mimo_role=role,
    )


def _copies(*, single_owner: bool) -> tuple[LoaderCopy, ...]:
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
                "tensor_model_parallel_size": 1,
                "context_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": 2,
                "rank_offset": 4,
            },
        }
    }
    return build_mimo_data_loader_placement(
        parallelism, world_size=6, single_owner=single_owner
    ).copies


@pytest.mark.parametrize("single_owner", [False, True])
def test_lane_check_accepts_agreeing_loaders(single_owner: bool) -> None:
    copies = _copies(single_owner=single_owner)
    envelopes = [_envelope(copy.logical_rank, copy.role) for copy in copies]
    check_mimo_lanes_agree(copies, envelopes)


@pytest.mark.parametrize("single_owner", [False, True])
def test_lane_check_rejects_a_diverged_encoder_shard(single_owner: bool) -> None:
    copies = _copies(single_owner=single_owner)
    envelopes = [_envelope(copy.logical_rank, copy.role) for copy in copies]
    # Encoder shard 1 of lane 0 read another source.
    envelopes[1] = _envelope(0, "encoder", sources=("b",))
    with pytest.raises(RuntimeError, match="diverged on logical lane 0"):
        check_mimo_lanes_agree(copies, envelopes)


def test_lane_check_rejects_an_envelope_of_another_lane() -> None:
    copies = _copies(single_owner=False)
    envelopes = [_envelope(copy.logical_rank, copy.role) for copy in copies]
    envelopes[0] = _envelope(1, "encoder")
    with pytest.raises(RuntimeError, match="returned lane 1"):
        check_mimo_lanes_agree(copies, envelopes)


@pytest.mark.parametrize("single_owner", [False, True])
def test_each_lane_is_counted_once(single_owner: bool) -> None:
    copies = _copies(single_owner=single_owner)
    envelopes = [_envelope(copy.logical_rank, copy.role) for copy in copies]
    counted = mimo_accounting_envelopes(copies, envelopes, single_owner=single_owner)
    assert [envelope.logical_rank for envelope in counted] == [0, 1]


# ── encoder payload ──────────────────────────────────────────────────────────


def _mimo_model() -> Any:
    encoder = SimpleNamespace(patch_dim=PATCH)
    return SimpleNamespace(
        modality_submodules={"images": SimpleNamespace(encoders={"pixtral": encoder})}
    )


def _microbatch(pixels: torch.Tensor | None, sizes: list[list[int]] | None) -> Any:
    input_ids = torch.zeros((1, 32), dtype=torch.long)
    data = BatchedDataDict({"input_ids": input_ids})
    if pixels is not None:
        data["pixel_values"] = PackedTensor(
            [pixels],
            dim_to_pack=0,
            preprocess_mode="patchify",
            preprocess_kwargs={"patch_dim": PATCH},
        )
        data["imgs_sizes"] = PackedTensor(
            [torch.tensor(sizes, dtype=torch.int32)], dim_to_pack=0
        )
    else:
        data["pixel_values"] = PackedTensor(
            [None],
            dim_to_pack=0,
            preprocess_mode="patchify",
            preprocess_kwargs={"patch_dim": PATCH},
        )
        data["imgs_sizes"] = PackedTensor([None], dim_to_pack=0)
    return ProcessedMicrobatch(
        data_dict=data,
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids,
        attention_mask=None,
        position_ids=None,
        packed_seq_params=None,
        cu_seqlens_padded=None,
    )


def test_encoder_payload_carries_only_the_range_and_no_input_ids() -> None:
    # Two images of this rank's range: 2x2 and 4x4 patches.
    pixels = torch.randn((1, 4 + 16, 3 * PATCH * PATCH))
    batch = _mimo_encoder_batch(
        _microbatch(pixels, [[28, 28], [56, 56]]), _mimo_model()
    )
    assert batch["input_ids"] is None
    payload = batch["modality_inputs"]["images"]["pixtral"]
    assert torch.equal(payload["x"], pixels)
    assert "pixel_values" not in payload
    assert payload["imgs_sizes"].tolist() == [[28, 28], [56, 56]]
    assert payload["packed_seq_params"].cu_seqlens_q.tolist() == [0, 4, 20]
    assert payload["packed_seq_params"].max_seqlen_q == 16


@pytest.mark.parametrize("microbatch", [None, "empty"])
def test_encoder_rank_with_an_empty_range_sends_nothing(microbatch) -> None:
    if microbatch == "empty":
        microbatch = _microbatch(None, None)
    batch = _mimo_encoder_batch(microbatch, _mimo_model())
    assert batch == {"input_ids": None, "modality_inputs": {}}


# ── setup ────────────────────────────────────────────────────────────────────


_DDP = SimpleNamespace(overlap_grad_reduce=False)


class _CoreWithoutFanInSplit:
    def recv_forward(self, recv_shape=None):
        raise AssertionError("not called")


class _CoreWith557(_CoreWithoutFanInSplit):
    def recv_forward(self, recv_shape=None, *, expect_backward=True):
        raise AssertionError("not called")


class _CoreWithFanInReset(_CoreWithoutFanInSplit):
    def reset_forward_fanin_state(self) -> None:
        raise AssertionError("not called")


def test_split_accepts_a_core_with_557(monkeypatch) -> None:
    monkeypatch.setattr(bridge_communicator, "BridgeCommunicator", _CoreWith557)
    _validate_mimo_encoder_media_split(_DDP)


@pytest.mark.parametrize("core", [_CoreWithoutFanInSplit, _CoreWithFanInReset])
def test_split_rejects_a_core_without_557(monkeypatch, core) -> None:
    monkeypatch.setattr(bridge_communicator, "BridgeCommunicator", core)
    with pytest.raises(NotImplementedError, match="nv-mistralai-megatron#557"):
        _validate_mimo_encoder_media_split(_DDP)


def test_split_rejects_overlapped_encoder_grad_reduce(monkeypatch) -> None:
    monkeypatch.setattr(bridge_communicator, "BridgeCommunicator", _CoreWith557)
    with pytest.raises(ValueError, match="overlap_grad_reduce=false"):
        _validate_mimo_encoder_media_split(SimpleNamespace(overlap_grad_reduce=True))


# ── encoder rows per rank ────────────────────────────────────────────────────


def _counted_microbatch(counts: list[int] | None) -> Any:
    microbatch = _microbatch(None, None)
    microbatch.data_dict[MIMO_MEDIA_OUTPUT_COUNTS_KEY] = PackedTensor(
        [None if counts is None else torch.tensor(counts, dtype=torch.int32)],
        dim_to_pack=0,
    )
    return microbatch


def test_encoder_rows_match_the_range_counts() -> None:
    hidden = torch.zeros((3, 8))
    _check_mimo_encoder_rows(
        {"images": hidden}, _counted_microbatch([2, 1]), require_counts=True
    )
    # An empty range: no media, no rows.
    _check_mimo_encoder_rows(
        {"images": torch.zeros((0, 8))}, _counted_microbatch(None), require_counts=True
    )
    _check_mimo_encoder_rows({}, _counted_microbatch(None), require_counts=True)
    # No pixels and no counts, or no batch: nothing to check.
    _check_mimo_encoder_rows(
        {"images": hidden}, _microbatch(None, None), require_counts=True
    )
    _check_mimo_encoder_rows({}, None, require_counts=True)


def test_encoder_rows_need_counts_when_the_loader_splits_media() -> None:
    pixels = torch.randn((1, 4, 3 * PATCH * PATCH))
    output = {"images": torch.zeros((1, 8))}
    with pytest.raises(RuntimeError, match="pixels but no MIMO media output counts"):
        _check_mimo_encoder_rows(
            output, _microbatch(pixels, [[28, 28]]), require_counts=True
        )
    # Without a split, task encoders that emit no counts keep working.
    _check_mimo_encoder_rows(
        output, _microbatch(pixels, [[28, 28]]), require_counts=False
    )


@pytest.mark.parametrize(
    ("encoder_dp", "language_dp", "active"), [(8, 1, True), (8, 4, True), (4, 4, False)]
)
def test_media_split_is_active_only_with_more_encoder_ranks(
    encoder_dp: int, language_dp: int, active: bool
) -> None:
    def grid(dp: int) -> Any:
        return SimpleNamespace(dim_names=["tp", "cp", "dp", "pp"], shape=[1, 1, dp, 1])

    mimo_model = SimpleNamespace(
        mimo_config=SimpleNamespace(
            module_to_grid_map={
                "images": grid(encoder_dp),
                "language": grid(language_dp),
            }
        )
    )
    assert _mimo_encoder_media_split_active(mimo_model) is active


def test_encoder_rows_that_disagree_with_the_counts_are_an_error() -> None:
    with pytest.raises(RuntimeError, match="produced 4 media embedding rows"):
        _check_mimo_encoder_rows(
            {"images": torch.zeros((4, 8))},
            _counted_microbatch([2, 1]),
            require_counts=True,
        )


def test_media_output_counts_stay_on_the_host() -> None:
    batch = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            MIMO_MEDIA_OUTPUT_COUNTS_KEY: PackedTensor(
                [torch.tensor([2, 1], dtype=torch.int32)], dim_to_pack=0
            ),
        }
    ).to("meta")
    assert batch["input_ids"].is_meta
    assert batch[MIMO_MEDIA_OUTPUT_COUNTS_KEY].tensors[0].device.type == "cpu"


def test_validation_loader_refuses_a_media_split() -> None:
    worker = _worker(module="images", dp_rank=0, dp_size=8, language_dp=1)
    worker._is_replica_leader = lambda: True
    worker._sft_validation_loader = None
    worker._sft_processor = object()
    worker._sft_logical_rank = 0
    worker._sft_logical_world_size = 1
    worker._sft_mimo_role = "encoder"
    worker._sft_encoder_shard = EncoderShard(index=0, count=8)
    with pytest.raises(NotImplementedError, match="validation"):
        worker.setup_sft_validation_dataloader(
            data_config={"validation": {}},
            batch_size=1,
            max_sequence_length=8,
            placement_fingerprint="placement",
            only_unmask_final=False,
        )
