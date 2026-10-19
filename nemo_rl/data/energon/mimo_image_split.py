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

"""Split each pack's media over the encoder ranks of one MIMO data lane.

When the encoder has more data-parallel ranks than the language model, every
encoder rank of a lane runs the lane's loader and builds the same packs. Each
rank keeps one contiguous range of the pack's media units, so the Bridge fan-in
joins the encoder outputs in media order. The ranges are cut by encoder cost and
need no communication: every rank computes the same cut from the same pack.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from nemo_rl.data.multimodal_utils import (
    MIMO_MEDIA_OUTPUT_COUNTS_KEY,
    PackedTensor,
    mimo_media_output_counts,
)

# Media fields that the split selects per range. Every other PackedTensor field
# of an encoder batch is unsupported, except the pack boundaries below.
SPLIT_MEDIA_KEYS = (
    "pixel_values",
    "imgs_sizes",
    "num_frames",
    MIMO_MEDIA_OUTPUT_COUNTS_KEY,
)
_PACK_BOUNDARY_KEYS = frozenset({"cu_seqlens", "cu_seqlens_padded"})


@dataclass(frozen=True)
class EncoderShard:
    """This encoder rank's position among the encoder ranks of its lane."""

    index: int
    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError(f"Encoder shard count must be positive, got {self.count}.")
        if not 0 <= self.index < self.count:
            raise ValueError(
                f"Encoder shard index {self.index} is outside [0, {self.count})."
            )


def encoder_shard_for_rank(
    *, dp_rank: int, dp_size: int, language_dp_size: int
) -> tuple[EncoderShard, int]:
    """Return an encoder rank's media shard and the language lane it feeds.

    Encoder DP is a whole multiple of language DP, so the encoder ranks divide
    into lanes of ``stride`` ranks, one lane per language rank. Every rank of a
    lane sees the same packs and keeps one media range of each.
    """
    if language_dp_size <= 0:
        raise ValueError("Language DP size must be positive.")
    if dp_size <= 0 or dp_size % language_dp_size:
        raise ValueError(
            f"Encoder DP size {dp_size} must be a positive multiple of "
            f"language DP size {language_dp_size}."
        )
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"Encoder DP rank {dp_rank} is outside [0, {dp_size}).")
    stride = dp_size // language_dp_size
    return EncoderShard(index=dp_rank % stride, count=stride), dp_rank // stride


@dataclass(frozen=True)
class MediaUnit:
    """One image or video tubelet: the smallest piece the split keeps whole."""

    occurrence: int
    frame_start: int
    frame_count: int
    cost: int


def media_units(
    frame_sizes: Sequence[tuple[int, int]],
    num_frames: Sequence[int],
    *,
    patch_dim: int,
    temporal_patch_size: int,
    frame_weight: int,
) -> list[MediaUnit]:
    """Group a pack's frames into units, in pack order.

    An occurrence with one frame is one unit. A video occurrence with more
    frames is one unit per ``temporal_patch_size`` frames. The cost of a unit is
    the patch count of one of its frames plus ``frame_weight`` per frame.

    A cut can fall between two tubelets of one video, so one video can be
    encoded on two ranks. This is correct
    for encoders that encode each tubelet on its own, such as the Nemotron4
    vision encoder. An encoder with state across the tubelets of a video would
    need cuts at occurrence boundaries.
    """
    if patch_dim <= 0:
        raise ValueError(f"patch_dim must be positive, got {patch_dim}.")
    if temporal_patch_size <= 0:
        raise ValueError(
            f"temporal_patch_size must be positive, got {temporal_patch_size}."
        )
    if frame_weight < 0:
        raise ValueError(f"frame_weight must be non-negative, got {frame_weight}.")
    if sum(num_frames) != len(frame_sizes):
        raise ValueError(
            "num_frames must describe every frame of the pack: "
            f"sum={sum(num_frames)}, frames={len(frame_sizes)}."
        )
    units: list[MediaUnit] = []
    frame = 0
    for occurrence, frames in enumerate(num_frames):
        if frames <= 0:
            raise ValueError(f"num_frames entries must be positive, got {frames}.")
        if frames > 1 and frames % temporal_patch_size:
            raise ValueError(
                f"A video of {frames} frames is not divisible by "
                f"temporal_patch_size={temporal_patch_size}."
            )
        unit_frames = 1 if frames == 1 else temporal_patch_size
        for start in range(frame, frame + frames, unit_frames):
            height, width = frame_sizes[start]
            if height <= 0 or width <= 0 or height % patch_dim or width % patch_dim:
                raise ValueError(
                    f"Frame size {(height, width)} is not a positive multiple of "
                    f"patch_dim={patch_dim}."
                )
            patches = (height // patch_dim) * (width // patch_dim)
            units.append(
                MediaUnit(
                    occurrence=occurrence,
                    frame_start=start,
                    frame_count=unit_frames,
                    cost=patches + frame_weight * unit_frames,
                )
            )
        frame += frames
    return units


def split_media_units(costs: Sequence[int], num_shards: int) -> list[tuple[int, int]]:
    """Cut in-order unit costs into contiguous ranges with the smallest maximum cost.

    Returns one ``[start, end)`` range per shard. A binary search finds the
    smallest capacity that a greedy fill from the left covers in ``num_shards``
    ranges; the cut is that greedy fill. The result depends only on ``costs``,
    so every shard computes the same ranges. Trailing ranges can be empty.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}.")
    if any(cost <= 0 for cost in costs):
        raise ValueError("Media unit costs must be positive.")
    if not costs:
        return [(0, 0)] * num_shards

    def ranges_at(capacity: int) -> list[tuple[int, int]]:
        ranges = []
        start = 0
        load = 0
        for index, cost in enumerate(costs):
            if load + cost > capacity:
                ranges.append((start, index))
                start = index
                load = 0
            load += cost
        ranges.append((start, len(costs)))
        return ranges

    low, high = max(costs), sum(costs)
    while low < high:
        middle = (low + high) // 2
        if len(ranges_at(middle)) <= num_shards:
            high = middle
        else:
            low = middle + 1
    ranges = ranges_at(low)
    return ranges + [(len(costs), len(costs))] * (num_shards - len(ranges))


def _row_value(value: PackedTensor, row: int) -> Optional[torch.Tensor]:
    """Return one logical row as a tensor without copying a single segment."""
    row_value = value.slice([row])
    if len(row_value.tensors) == 1:
        tensor = row_value.tensors[0]
        if tensor is None:
            return None
        # Pixel rows are usually one pre-patchified [1, patches, features] segment.
        if value.preprocess_mode != "patchify" or tensor.ndim == 3:
            return tensor
    return row_value.as_tensor()


def _rebuilt(value: PackedTensor, rows: list[Optional[torch.Tensor]]) -> PackedTensor:
    return PackedTensor(
        rows,
        value.dim_to_pack,
        preprocess_mode=value.preprocess_mode,
        preprocess_kwargs=value.preprocess_kwargs,
    )


@dataclass(frozen=True)
class EncoderMediaSplit:
    """Everything an encoder rank needs to keep only its own media range."""

    shard: EncoderShard
    patch_dim: int
    temporal_patch_size: int
    frame_weight: int
    image_token_id: Optional[int] = None


def apply_encoder_media_split(
    batch: MutableMapping[str, Any], split: EncoderMediaSplit
) -> None:
    """Reduce a rollout batch in place to this encoder rank's media range.

    This is the GRPO counterpart of what the Energon task encoder does in
    ``encode_batch``: text stays whole on every rank, media does not. GRPO has
    no task encoder, so the output counts are derived here first.
    """
    attach_mimo_media_output_counts(batch, patch_dim=split.patch_dim)
    batch.update(
        select_encoder_shard_media(
            batch,
            shard=split.shard,
            patch_dim=split.patch_dim,
            temporal_patch_size=split.temporal_patch_size,
            frame_weight=split.frame_weight,
            image_token_id=split.image_token_id,
        )
    )


def attach_mimo_media_output_counts(
    batch: MutableMapping[str, Any], *, patch_dim: int
) -> None:
    """Give a rollout batch the per-media output counts the encoder split needs.

    Energon batches arrive with these counts already set by the task encoder, so
    this is a no-op for them beyond an agreement check. GRPO batches never run
    that encoder, and ``select_encoder_shard_media`` treats missing counts as
    "unknown", which would leave encoder ranks unable to validate their media
    rows. Rebuilding them from ``imgs_sizes`` keeps one row per pack, matching
    the layout the split expects.
    """
    pixels = batch.get("pixel_values")
    if pixels is None:
        return
    sizes = batch.get("imgs_sizes")
    if not isinstance(sizes, PackedTensor):
        raise ValueError("MIMO media output counts require 'imgs_sizes'.")

    rows: list[Optional[torch.Tensor]] = []
    for row in range(len(sizes)):
        row_sizes = _row_value(sizes, row)
        if row_sizes is None:
            rows.append(None)
            continue
        rows.append(mimo_media_output_counts(row_sizes, patch_dim=patch_dim))

    existing = batch.get(MIMO_MEDIA_OUTPUT_COUNTS_KEY)
    if isinstance(existing, PackedTensor):
        for row, derived in enumerate(rows):
            present = _row_value(existing, row)
            if present is None and derived is None:
                continue
            if (
                present is None
                or derived is None
                or not torch.equal(present.to(torch.int32), derived)
            ):
                raise ValueError(
                    "Derived MIMO media output counts disagree with the counts "
                    f"already on the batch at pack {row}."
                )
        return

    batch[MIMO_MEDIA_OUTPUT_COUNTS_KEY] = _rebuilt(sizes, rows)


def select_encoder_shard_media(
    batch: Mapping[str, Any],
    *,
    shard: EncoderShard,
    patch_dim: int,
    temporal_patch_size: int,
    frame_weight: int,
    image_token_id: Optional[int] = None,
) -> dict[str, PackedTensor]:
    """Return the media fields of ``batch`` reduced to this shard's range per pack.

    ``batch`` is a prepared packed batch with one row per pack. Text fields are
    not changed by the caller: every shard keeps the full token stream. Pixel
    rows of the kept range are copied, so a batch carries only its own pixels.
    A pack with no media, or with fewer units than shards, gives empty rows.

    The placeholder check compares the whole pack: the sum of its output counts
    with its image tokens. Each range is consistent by construction, because
    the output count of every unit comes from the same visual plan that
    expanded that unit's placeholders.
    """
    for key, value in batch.items():
        if (
            isinstance(value, PackedTensor)
            and key not in SPLIT_MEDIA_KEYS
            and key not in _PACK_BOUNDARY_KEYS
        ):
            raise ValueError(f"The encoder image split does not support {key!r}.")
    pixels = batch.get("pixel_values")
    if pixels is None:
        return {}
    for key in ("imgs_sizes", "num_frames"):
        if not isinstance(batch.get(key), PackedTensor):
            raise ValueError(f"The encoder image split requires {key!r}.")
    sizes = batch["imgs_sizes"]
    frames = batch["num_frames"]
    counts = batch.get(MIMO_MEDIA_OUTPUT_COUNTS_KEY)
    input_ids = batch.get("input_ids")
    num_rows = len(pixels)
    if any(len(value) != num_rows for value in (sizes, frames)) or (
        counts is not None and len(counts) != num_rows
    ):
        raise ValueError("Media fields of one batch must have one row per pack.")

    selected: dict[str, list[Optional[torch.Tensor]]] = {
        key: [] for key in SPLIT_MEDIA_KEYS if batch.get(key) is not None
    }
    for row in range(num_rows):
        row_pixels = _row_value(pixels, row)
        if row_pixels is None:
            for rows in selected.values():
                rows.append(None)
            continue
        row_sizes = _row_value(sizes, row)
        row_frames = _row_value(frames, row)
        if row_sizes is None or row_frames is None:
            raise ValueError("A pack with pixels requires imgs_sizes and num_frames.")
        frame_sizes = [
            (int(height), int(width)) for height, width in row_sizes.tolist()
        ]
        frame_counts = [int(value) for value in row_frames.reshape(-1).tolist()]
        units = media_units(
            frame_sizes,
            frame_counts,
            patch_dim=patch_dim,
            temporal_patch_size=temporal_patch_size,
            frame_weight=frame_weight,
        )
        patch_offsets = [0]
        for height, width in frame_sizes:
            patch_offsets.append(
                patch_offsets[-1] + (height // patch_dim) * (width // patch_dim)
            )
        # Energon rows carry a leading segment axis, [1, patches, features];
        # rollout rows are the same data without it. The patch axis is the
        # second to last on both.
        if row_pixels.ndim not in (2, 3) or row_pixels.shape[-2] != patch_offsets[-1]:
            raise ValueError(
                "Pack pixels disagree with imgs_sizes: "
                f"pixels={tuple(row_pixels.shape)}, patches={patch_offsets[-1]}."
            )
        row_counts = None
        if counts is not None:
            row_counts = _row_value(counts, row)
            if row_counts is None or row_counts.numel() != len(units):
                raise ValueError(
                    "Media output counts need one entry per image or tubelet: "
                    f"units={len(units)}, counts="
                    f"{None if row_counts is None else row_counts.numel()}."
                )
            if image_token_id is not None and isinstance(input_ids, torch.Tensor):
                placeholders = int((input_ids[row] == image_token_id).sum())
                if placeholders != int(row_counts.sum()):
                    raise ValueError(
                        "Media output counts disagree with the image placeholders: "
                        f"counts={int(row_counts.sum())}, "
                        f"placeholders={placeholders}."
                    )

        start, end = split_media_units([unit.cost for unit in units], shard.count)[
            shard.index
        ]
        if start == end:
            for rows in selected.values():
                rows.append(None)
            continue
        kept = units[start:end]
        first_frame = kept[0].frame_start
        last_frame = kept[-1].frame_start + kept[-1].frame_count
        # One entry per kept part of an occurrence: its kept frames for a video,
        # 1 for an image.
        kept_frames: dict[int, int] = {}
        for unit in kept:
            if frame_counts[unit.occurrence] == 1:
                kept_frames[unit.occurrence] = 1
            else:
                kept_frames[unit.occurrence] = (
                    kept_frames.get(unit.occurrence, 0) + unit.frame_count
                )
        patch_start = patch_offsets[first_frame]
        selected["pixel_values"].append(
            row_pixels.narrow(
                row_pixels.ndim - 2,
                patch_start,
                patch_offsets[last_frame] - patch_start,
            ).clone()
        )
        selected["imgs_sizes"].append(row_sizes[first_frame:last_frame].clone())
        selected["num_frames"].append(
            torch.tensor(list(kept_frames.values()), dtype=row_frames.dtype)
        )
        if row_counts is not None:
            selected[MIMO_MEDIA_OUTPUT_COUNTS_KEY].append(
                row_counts.reshape(-1)[start:end].clone()
            )
    return {key: _rebuilt(batch[key], rows) for key, rows in selected.items()}


__all__ = [
    "EncoderShard",
    "MediaUnit",
    "SPLIT_MEDIA_KEYS",
    "media_units",
    "select_encoder_shard_media",
    "split_media_units",
]
