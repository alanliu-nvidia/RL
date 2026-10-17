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

"""The MIMO encoder media split: cut, per-pack selection, and loader placement."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from megatron.energon.errors import ErrorContext, FatalSampleError

from nemo_rl.data.energon.mimo_image_split import (
    EncoderShard,
    media_units,
    select_encoder_shard_media,
    split_media_units,
)
from nemo_rl.data.energon.multimodal.task_encoders import nemotron4_multimodal
from nemo_rl.data.energon.multimodal.task_encoders.nemotron4_multimodal import (
    Nemotron4MultiModalTaskEncoder,
)
from nemo_rl.data.energon.topology import build_mimo_data_loader_placement
from nemo_rl.data.multimodal_utils import MIMO_MEDIA_OUTPUT_COUNTS_KEY, PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

PATCH = 14
FEATURES = 3 * PATCH * PATCH
IMAGE_TOKEN = 99


# ── cut ──────────────────────────────────────────────────────────────────────


def _brute_force_max(costs: list[int], shards: int) -> int:
    """Smallest maximum range cost over all contiguous cuts into <= shards ranges."""
    best = sum(costs)
    for cut_count in range(min(shards, len(costs))):
        for cuts in itertools.combinations(range(1, len(costs)), cut_count):
            bounds = (0, *cuts, len(costs))
            best = min(
                best,
                max(sum(costs[a:b]) for a, b in zip(bounds, bounds[1:])),
            )
    return best


def _check_ranges(ranges: list[tuple[int, int]], units: int, shards: int) -> None:
    assert len(ranges) == shards
    assert ranges[0][0] == 0 and ranges[-1][1] == units
    for (start, end), (next_start, _) in zip(ranges, ranges[1:]):
        assert start <= end == next_start


def test_split_is_optimal_contiguous_and_ordered() -> None:
    rng = random.Random(0)
    for _ in range(300):
        units = rng.randint(1, 10)
        shards = rng.randint(1, 5)
        costs = [rng.randint(1, 50) for _ in range(units)]
        ranges = split_media_units(costs, shards)
        _check_ranges(ranges, units, shards)
        got = max(sum(costs[start:end]) for start, end in ranges)
        assert got == _brute_force_max(costs, shards), (costs, shards, ranges)


def test_split_is_deterministic() -> None:
    costs = [5, 1, 9, 3, 3, 7, 2, 8, 6, 4]
    assert split_media_units(costs, 4) == split_media_units(list(costs), 4)


def test_split_leaves_trailing_shards_empty_when_units_are_few() -> None:
    assert split_media_units([3, 4], 4) == [(0, 1), (1, 2), (2, 2), (2, 2)]
    assert split_media_units([], 3) == [(0, 0), (0, 0), (0, 0)]


def test_split_gives_a_unit_larger_than_the_rest_its_own_range() -> None:
    ranges = split_media_units([1, 1, 100, 1, 1], 3)
    assert max(sum([1, 1, 100, 1, 1][a:b]) for a, b in ranges) == 100


@pytest.mark.parametrize(
    ("costs", "shards"), [([1, 2], 0), ([1, 0, 2], 2), ([1, -3], 2)]
)
def test_split_rejects_bad_input(costs: list[int], shards: int) -> None:
    with pytest.raises(ValueError):
        split_media_units(costs, shards)


# ── units ────────────────────────────────────────────────────────────────────


def test_units_keep_tubelets_whole() -> None:
    sizes = [(28, 28)] + [(28, 42)] * 4 + [(56, 56)]
    units = media_units(
        sizes, [1, 4, 1], patch_dim=PATCH, temporal_patch_size=2, frame_weight=10
    )
    assert [(u.occurrence, u.frame_start, u.frame_count) for u in units] == [
        (0, 0, 1),
        (1, 1, 2),
        (1, 3, 2),
        (2, 5, 1),
    ]
    # Patches of one frame + weight per frame.
    assert [u.cost for u in units] == [4 + 10, 6 + 20, 6 + 20, 16 + 10]


def test_units_with_temporal_patch_one_split_videos_per_frame() -> None:
    units = media_units(
        [(28, 28)] * 3, [3], patch_dim=PATCH, temporal_patch_size=1, frame_weight=0
    )
    assert [(u.frame_start, u.frame_count) for u in units] == [(0, 1), (1, 1), (2, 1)]


@pytest.mark.parametrize(
    ("sizes", "frames", "message"),
    [
        ([(28, 28)] * 3, [3], "not divisible"),
        ([(28, 28)] * 2, [1], "every frame"),
        ([(28, 30)], [1], "multiple of patch_dim"),
        ([(28, 28)], [0, 1], "positive"),
    ],
)
def test_units_reject_inconsistent_metadata(sizes, frames, message) -> None:
    with pytest.raises(ValueError, match=message):
        media_units(
            sizes, frames, patch_dim=PATCH, temporal_patch_size=2, frame_weight=0
        )


# ── per-pack selection ───────────────────────────────────────────────────────


def _frames(count: int, height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((count, 3, height, width), generator=generator)


def _patchified(segments: list[torch.Tensor]) -> torch.Tensor:
    return PackedTensor(
        segments,
        dim_to_pack=0,
        preprocess_mode="patchify",
        preprocess_kwargs={"patch_dim": PATCH},
    ).as_tensor()


# A pack is (frame segments, num_frames, output counts); None is a text-only pack.
# Pack 0: image (2x2 patches), 4-frame video (2x3 patches per frame), image (4x4).
PACK0 = (
    [_frames(1, 28, 28, 0), _frames(4, 28, 42, 1), _frames(1, 56, 56, 2)],
    [1, 4, 1],
    [1, 2, 2, 4],  # one per image or 2-frame tubelet
)
PACK0_SIZES = [[28, 28]] + [[28, 42]] * 4 + [[56, 56]]
PACK2 = ([_frames(1, 42, 28, 3)], [1], [1])
PACKS = [PACK0, None, PACK2]


def _sizes(segments: list[torch.Tensor]) -> list[list[int]]:
    return [
        [segment.shape[2], segment.shape[3]]
        for segment in segments
        for _ in range(segment.shape[0])
    ]


def _input_ids(packs) -> torch.Tensor:
    ids = torch.zeros((len(packs), 40), dtype=torch.long)
    for row, pack in enumerate(packs):
        if pack is not None:
            ids[row, 5 : 5 + sum(pack[2])] = IMAGE_TOKEN
    return ids


def _flat_batch(packs=PACKS) -> BatchedDataDict:
    """Rows as the packed SFT path builds them: one flattened tensor per pack."""

    def rows(make):
        return [None if pack is None else make(*pack) for pack in packs]

    return BatchedDataDict(
        {
            "input_ids": _input_ids(packs),
            "pixel_values": PackedTensor(
                rows(lambda segments, frames, counts: _patchified(segments)),
                dim_to_pack=0,
                preprocess_mode="patchify",
                preprocess_kwargs={"patch_dim": PATCH},
            ),
            "imgs_sizes": PackedTensor(
                rows(
                    lambda segments, frames, counts: torch.tensor(
                        _sizes(segments), dtype=torch.int32
                    )
                ),
                dim_to_pack=0,
            ),
            "num_frames": PackedTensor(
                rows(lambda segments, frames, counts: torch.tensor(frames)),
                dim_to_pack=0,
            ),
            MIMO_MEDIA_OUTPUT_COUNTS_KEY: PackedTensor(
                rows(
                    lambda segments, frames, counts: torch.tensor(
                        counts, dtype=torch.int32
                    )
                ),
                dim_to_pack=0,
            ),
            "cu_seqlens": PackedTensor(
                [torch.tensor([0, 40], dtype=torch.int32)] * len(packs), dim_to_pack=0
            ),
        }
    )


def _segmented(rows: list[list[torch.Tensor] | None], **kwargs) -> PackedTensor:
    """Rows as several deduplicated segments per pack, one per occurrence."""
    packed = []
    for segments in rows:
        if segments is None:
            template = PackedTensor(
                [torch.zeros((1, 3, PATCH, PATCH))], dim_to_pack=0, **kwargs
            ).enable_deduplication()
            packed.append(PackedTensor.empty_rows_like(template, 1))
            continue
        packed.append(
            PackedTensor.merge_segments(
                [
                    PackedTensor(
                        segment, dim_to_pack=0, **kwargs
                    ).enable_deduplication()
                    for segment in segments
                ]
            )
        )
    return PackedTensor.concat(packed)


def _segmented_batch(packs=PACKS) -> BatchedDataDict:
    """The same packs with one segment per image or video."""

    def per_occurrence(make):
        return [None if pack is None else make(*pack) for pack in packs]

    def counts_by_occurrence(frames: list[int], counts: list[int]) -> list[list[int]]:
        result, start = [], 0
        for value in frames:
            units = 1 if value == 1 else value // 2
            result.append(counts[start : start + units])
            start += units
        return result

    return BatchedDataDict(
        {
            "input_ids": _input_ids(packs),
            "pixel_values": _segmented(
                per_occurrence(lambda segments, frames, counts: segments),
                preprocess_mode="patchify",
                preprocess_kwargs={"patch_dim": PATCH},
            ),
            "imgs_sizes": _segmented(
                per_occurrence(
                    lambda segments, frames, counts: [
                        torch.tensor(_sizes([segment]), dtype=torch.int32)
                        for segment in segments
                    ]
                )
            ),
            "num_frames": _segmented(
                per_occurrence(
                    lambda segments, frames, counts: [
                        torch.tensor([value]) for value in frames
                    ]
                )
            ),
            MIMO_MEDIA_OUTPUT_COUNTS_KEY: _segmented(
                per_occurrence(
                    lambda segments, frames, counts: [
                        torch.tensor(part, dtype=torch.int32)
                        for part in counts_by_occurrence(frames, counts)
                    ]
                )
            ),
        }
    )


def _row(value: PackedTensor, row: int) -> torch.Tensor | None:
    return value.slice([row]).as_tensor()


def _select(batch: BatchedDataDict, index: int, count: int, temporal: int = 2):
    return select_encoder_shard_media(
        batch,
        shard=EncoderShard(index=index, count=count),
        patch_dim=PATCH,
        temporal_patch_size=temporal,
        frame_weight=0,
        image_token_id=IMAGE_TOKEN,
    )


@pytest.mark.parametrize("layout", [_flat_batch, _segmented_batch])
@pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
def test_shards_join_back_to_the_full_pack(layout, count: int) -> None:
    batch = layout()
    shards = [_select(layout(), index, count) for index in range(count)]
    for row in range(3):
        for key in ("pixel_values", "imgs_sizes", MIMO_MEDIA_OUTPUT_COUNTS_KEY):
            parts = [
                part for shard in shards if (part := _row(shard[key], row)) is not None
            ]
            expected = _row(batch[key], row)
            if expected is None:
                assert parts == []
                continue
            joined = torch.cat(parts, dim=1 if key == "pixel_values" else 0)
            assert torch.equal(joined, expected), (key, row, count)
        # Video parts carry their kept frame count; parts add up per occurrence.
        frames = [
            _row(shard["num_frames"], row)
            for shard in shards
            if _row(shard["num_frames"], row) is not None
        ]
        if row == 0:
            assert int(torch.cat(frames).sum()) == sum(PACK0[1])


def test_ranges_follow_cost() -> None:
    # Units of pack 0 cost 4, 6, 6, 16 patches: two shards cut before the last image.
    first, second = (_select(_flat_batch(), index, 2) for index in range(2))
    assert _row(first["imgs_sizes"], 0).tolist() == PACK0_SIZES[:5]
    assert _row(first["num_frames"], 0).tolist() == [1, 4]
    assert _row(first[MIMO_MEDIA_OUTPUT_COUNTS_KEY], 0).tolist() == [1, 2, 2]
    assert _row(second["imgs_sizes"], 0).tolist() == [PACK0_SIZES[5]]
    assert _row(second["num_frames"], 0).tolist() == [1]


@pytest.mark.parametrize("layout", [_flat_batch, _segmented_batch])
def test_a_video_is_cut_between_tubelets_only(layout) -> None:
    # Image (4 patches) + 4-frame video (two tubelets of 6): the best cut of two
    # shards is inside the video, after its first tubelet.
    pack = ([_frames(1, 28, 28, 4), _frames(4, 28, 42, 5)], [1, 4], [1, 2, 2])
    first, second = (_select(layout([pack]), index, 2) for index in range(2))
    assert _row(first["num_frames"], 0).tolist() == [1, 2]
    assert _row(first["imgs_sizes"], 0).tolist() == [[28, 28], [28, 42], [28, 42]]
    assert _row(first[MIMO_MEDIA_OUTPUT_COUNTS_KEY], 0).tolist() == [1, 2]
    assert _row(second["num_frames"], 0).tolist() == [2]
    assert _row(second["imgs_sizes"], 0).tolist() == [[28, 42], [28, 42]]
    assert torch.equal(
        torch.cat(
            [_row(first["pixel_values"], 0), _row(second["pixel_values"], 0)], dim=1
        ),
        _patchified(pack[0]),
    )


def test_text_only_packs_and_extra_shards_are_empty() -> None:
    selected = [_select(_flat_batch(), index, 8) for index in range(8)]
    for shard in selected:
        assert _row(shard["pixel_values"], 1) is None
    # Pack 2 has one image: shard 0 keeps it, the others are empty.
    assert _row(selected[0]["pixel_values"], 2) is not None
    assert all(_row(shard["pixel_values"], 2) is None for shard in selected[1:])


def test_text_only_batch_has_nothing_to_select() -> None:
    batch = BatchedDataDict({"input_ids": torch.zeros((2, 4), dtype=torch.long)})
    assert _select(batch, 0, 2) == {}


def test_kept_pixels_do_not_hold_the_full_pack() -> None:
    batch = _flat_batch()
    full = batch["pixel_values"].tensors[0]
    kept = _select(batch, 1, 2)["pixel_values"].tensors[0]
    assert kept.untyped_storage().data_ptr() != full.untyped_storage().data_ptr()
    assert kept.untyped_storage().nbytes() == kept.numel() * kept.element_size()


def test_placeholder_mismatch_is_an_error() -> None:
    batch = _flat_batch()
    batch["input_ids"][0, 0] = IMAGE_TOKEN
    with pytest.raises(ValueError, match="placeholders"):
        _select(batch, 0, 2)


def test_pixels_must_match_sizes() -> None:
    batch = _flat_batch()
    batch["imgs_sizes"].tensors[0] = torch.tensor([[28, 28]] * 6, dtype=torch.int32)
    with pytest.raises(ValueError, match="disagree with imgs_sizes"):
        _select(batch, 0, 2)


def test_unsupported_media_fields_are_an_error() -> None:
    batch = _flat_batch()
    batch["sound_clips"] = PackedTensor([torch.zeros(4), None, None], dim_to_pack=0)
    with pytest.raises(ValueError, match="sound_clips"):
        _select(batch, 0, 2)


# ── task encoder ─────────────────────────────────────────────────────────────


def _task_encoder(shard: EncoderShard | None) -> Nemotron4MultiModalTaskEncoder:
    encoder = object.__new__(Nemotron4MultiModalTaskEncoder)
    encoder._multimodal_adapter = SimpleNamespace(
        patch_dim=PATCH, temporal_patch_size=2
    )
    encoder._mimo_encoder_shard = shard
    encoder._encoder_media_split_frame_weight = 0
    encoder._image_token_id = IMAGE_TOKEN
    encoder._pixel_transport_dtype = torch.bfloat16
    return encoder


def test_encode_batch_selects_before_the_pixel_cast() -> None:
    expected = _select(_flat_batch(), 1, 2)
    batch = _task_encoder(EncoderShard(index=1, count=2)).encode_batch(_flat_batch())
    pixels = batch["pixel_values"].tensors[0]
    assert pixels.dtype == torch.bfloat16
    assert torch.equal(pixels, expected["pixel_values"].tensors[0].bfloat16())
    assert torch.equal(batch["input_ids"], _input_ids(PACKS))


def test_a_bad_pack_stops_the_loader_instead_of_being_skipped() -> None:
    # Energon calls its handler and skips the batch for an ordinary error.
    handler = MagicMock()
    with ErrorContext("encode_batch", handler=handler, tolerance=0).handle_errors(None):
        raise ValueError("an ordinary error")
    handler.assert_called_once()
    # A split error must not be skipped: it would drop the batch on encoder
    # shards only. It stops the loader with the split's message.
    handler.reset_mock()
    batch = _flat_batch()
    batch["input_ids"][0, 0] = IMAGE_TOKEN
    encoder = _task_encoder(EncoderShard(index=0, count=2))
    context = ErrorContext("encode_batch", handler=handler, tolerance=0)
    with pytest.raises(FatalSampleError, match="placeholders"):
        with context.handle_errors(batch):
            encoder.encode_batch(batch)
    handler.assert_not_called()


@pytest.mark.parametrize("error", [KeyError("missing"), IndexError("range")])
def test_any_split_error_is_fatal(monkeypatch, error: Exception) -> None:
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(nemotron4_multimodal, "select_encoder_shard_media", fail)
    encoder = _task_encoder(EncoderShard(index=0, count=2))
    with pytest.raises(FatalSampleError, match="encoder media split failed") as caught:
        encoder.encode_batch(_flat_batch())
    assert caught.value.__cause__ is error


def test_a_fatal_split_error_is_not_wrapped(monkeypatch) -> None:
    fatal = FatalSampleError("already fatal")

    def fail(*args, **kwargs):
        raise fatal

    monkeypatch.setattr(nemotron4_multimodal, "select_encoder_shard_media", fail)
    with pytest.raises(FatalSampleError) as caught:
        _task_encoder(EncoderShard(index=0, count=2)).encode_batch(_flat_batch())
    assert caught.value is fatal


def test_encode_batch_without_a_split_keeps_every_pixel() -> None:
    batch = _task_encoder(None).encode_batch(_flat_batch())
    assert batch["pixel_values"].tensors[0].shape == _patchified(PACK0[0]).shape


# ── placement ────────────────────────────────────────────────────────────────


def _parallelism(encoder_dp: int, language_dp: int) -> dict:
    def module(dp: int, offset: int, cp: int = 1) -> dict:
        return {
            "tensor_model_parallel_size": 1,
            "context_parallel_size": cp,
            "pipeline_model_parallel_size": 1,
            "data_parallel_size": dp,
            "rank_offset": offset,
        }

    return {
        "module_parallelisms": {
            "images": module(encoder_dp, 0),
            "language": module(language_dp, encoder_dp, cp=2),
        }
    }


@pytest.mark.parametrize("single_owner", [False, True])
@pytest.mark.parametrize(("encoder_dp", "language_dp"), [(8, 1), (8, 4)])
def test_placement_gives_every_encoder_replica_a_shard(
    encoder_dp: int, language_dp: int, single_owner: bool
) -> None:
    world_size = encoder_dp + 2 * language_dp
    plan = build_mimo_data_loader_placement(
        _parallelism(encoder_dp, language_dp),
        world_size=world_size,
        single_owner=single_owner,
    )
    stride = encoder_dp // language_dp
    assert plan.logical_world_size == language_dp
    copies = list(plan.copies)
    per_lane = stride + (0 if single_owner else 1)
    assert len(copies) == language_dp * per_lane
    for lane in range(language_dp):
        lane_copies = copies[lane * per_lane : (lane + 1) * per_lane]
        encoders = lane_copies[:stride]
        assert [copy.encoder_shard for copy in encoders] == list(range(stride))
        assert [copy.owner_rank for copy in encoders] == [
            lane * stride + shard for shard in range(stride)
        ]
        assert all(copy.role == "encoder" for copy in encoders)
        assert all(copy.logical_rank == lane for copy in lane_copies)
        language_ranks = (encoder_dp + 2 * lane, encoder_dp + 2 * lane + 1)
        if single_owner:
            assert encoders[0].role_delivery_ranks == {"language": language_ranks}
            assert all(copy.role_delivery_ranks == {} for copy in encoders[1:])
        else:
            assert lane_copies[-1].role == "language"
            assert lane_copies[-1].delivery_ranks == language_ranks
    delivered = [rank for copy in copies for rank in copy.delivery_ranks]
    delivered += [
        rank
        for copy in copies
        for ranks in copy.role_delivery_ranks.values()
        for rank in ranks
    ]
    assert sorted(delivered) == list(range(world_size))


def test_placement_hash_is_unchanged_without_a_split() -> None:
    parallelism = _parallelism(2, 2)
    plan = build_mimo_data_loader_placement(
        parallelism, world_size=6, single_owner=False
    )
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
    payload = {
        "kind": "mimo-native-v2-role-loaders-v2",
        "world_size": 6,
        "logical_world_size": 2,
        "modules": modules,
        "owners": {"encoder": 0, "language": 2},
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert plan.placement_hash == expected
    split = build_mimo_data_loader_placement(
        _parallelism(4, 2), world_size=8, single_owner=False
    )
    assert split.placement_hash != expected


def test_placement_rejects_encoder_dp_not_a_multiple_of_language_dp() -> None:
    with pytest.raises(ValueError, match="multiple of language DP"):
        build_mimo_data_loader_placement(
            _parallelism(6, 4), world_size=14, single_owner=False
        )
