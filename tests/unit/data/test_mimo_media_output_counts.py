# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch

from nemo_rl.data.energon.mimo_image_split import (
    EncoderMediaSplit,
    EncoderShard,
    apply_encoder_media_split,
    attach_mimo_media_output_counts,
    encoder_shard_for_rank,
)
from nemo_rl.data.multimodal_utils import (
    MIMO_MEDIA_OUTPUT_COUNTS_KEY,
    PackedTensor,
    mimo_media_output_counts,
    radio_token_count,
)

# NT4 Nano's vision_config: patch_size 14, and a 1036x1036 processed image, for
# which the Nemotron4 processor emits 1369 image placeholders.
NT4_PATCH_DIM = 14


def test_matches_observed_placeholder_count():
    assert radio_token_count(height=1036, width=1036, patch_dim=NT4_PATCH_DIM) == 1369


def test_counts_are_per_image_and_order_follows_rows():
    # 1008 is 72 patches wide, which the 2x2 shuffle can halve; 518 would be 37
    # and is rejected, so a non-square case has to keep both grid dims even.
    sizes = torch.tensor([[1036, 1036], [1036, 1008]], dtype=torch.int32)
    counts = mimo_media_output_counts(sizes, patch_dim=NT4_PATCH_DIM)
    assert counts.tolist() == [1369, 74 * 72 // 4]
    assert counts.dtype == torch.int32


def test_flat_image_sizes_are_reshaped():
    flat = torch.tensor([1036, 1036], dtype=torch.int32)
    assert mimo_media_output_counts(flat, patch_dim=NT4_PATCH_DIM).tolist() == [1369]


def test_rejects_sizes_that_are_not_whole_patches():
    # 1036 is divisible by 14 but not by 16, which is why the count cannot be
    # derived without the model's own patch size.
    with pytest.raises(ValueError, match="not divisible by patch_dim"):
        radio_token_count(height=1036, width=1036, patch_dim=16)


def test_rejects_patch_grids_the_pixel_shuffle_cannot_halve():
    with pytest.raises(ValueError, match="2x2 RADIO pixel shuffle"):
        radio_token_count(height=14, width=14, patch_dim=NT4_PATCH_DIM)


def test_rejects_nonpositive_patch_dim():
    with pytest.raises(ValueError, match="greater than zero"):
        radio_token_count(height=1036, width=1036, patch_dim=0)


def _rollout_media_batch(sizes_per_pack):
    """Mimic the media fields rollout_reassembler builds: one row per pack."""
    return {
        "pixel_values": PackedTensor(
            [torch.zeros((sum(len(s) for s in sizes_per_pack), 3))], dim_to_pack=0
        ),
        "imgs_sizes": PackedTensor(
            [torch.tensor(sizes, dtype=torch.int32) for sizes in sizes_per_pack],
            dim_to_pack=0,
        ),
    }


def test_attaches_counts_to_a_rollout_batch():
    batch = _rollout_media_batch([[[1036, 1036], [1036, 1008]]])
    attach_mimo_media_output_counts(batch, patch_dim=NT4_PATCH_DIM)
    counts = batch[MIMO_MEDIA_OUTPUT_COUNTS_KEY]
    assert counts.tensors[0].tolist() == [1369, 74 * 72 // 4]


def test_attach_is_a_no_op_without_pixels():
    batch = {"imgs_sizes": PackedTensor([torch.tensor([[1036, 1036]])], dim_to_pack=0)}
    attach_mimo_media_output_counts(batch, patch_dim=NT4_PATCH_DIM)
    assert MIMO_MEDIA_OUTPUT_COUNTS_KEY not in batch


def test_attach_requires_imgs_sizes_when_pixels_are_present():
    batch = {"pixel_values": PackedTensor([torch.zeros((1, 3))], dim_to_pack=0)}
    with pytest.raises(ValueError, match="require 'imgs_sizes'"):
        attach_mimo_media_output_counts(batch, patch_dim=NT4_PATCH_DIM)


def test_attach_rejects_counts_that_disagree_with_geometry():
    batch = _rollout_media_batch([[[1036, 1036]]])
    batch[MIMO_MEDIA_OUTPUT_COUNTS_KEY] = PackedTensor(
        [torch.tensor([1234], dtype=torch.int32)], dim_to_pack=0
    )
    with pytest.raises(ValueError, match="disagree with the counts"):
        attach_mimo_media_output_counts(batch, patch_dim=NT4_PATCH_DIM)


def test_attach_accepts_counts_the_task_encoder_already_set():
    batch = _rollout_media_batch([[[1036, 1036]]])
    batch[MIMO_MEDIA_OUTPUT_COUNTS_KEY] = PackedTensor(
        [torch.tensor([1369], dtype=torch.int32)], dim_to_pack=0
    )
    attach_mimo_media_output_counts(batch, patch_dim=NT4_PATCH_DIM)
    assert batch[MIMO_MEDIA_OUTPUT_COUNTS_KEY].tensors[0].tolist() == [1369]


# One 1036x1036 image is a 74x74 patch grid, so 5476 patches and 1369 outputs.
PATCHES_PER_IMAGE = 74 * 74
OUTPUTS_PER_IMAGE = PATCHES_PER_IMAGE // 4


def _two_image_rollout_batch(*, patchified: bool):
    """Build the media of one pack holding two equal images.

    Energon rows keep a leading segment axis; rollout rows drop it. Both carry
    the same patches in the same order.
    """
    pixels = torch.arange(2 * PATCHES_PER_IMAGE * 4, dtype=torch.float32).reshape(
        2 * PATCHES_PER_IMAGE, 4
    )
    if patchified:
        pixels = pixels.unsqueeze(0)
    return {
        "pixel_values": PackedTensor([pixels], dim_to_pack=0),
        "imgs_sizes": PackedTensor(
            [torch.tensor([[1036, 1036], [1036, 1036]], dtype=torch.int32)],
            dim_to_pack=0,
        ),
        "num_frames": PackedTensor(
            [torch.tensor([1, 1], dtype=torch.int32)], dim_to_pack=0
        ),
    }


@pytest.mark.parametrize("patchified", [False, True])
def test_split_halves_two_images_in_either_pixel_layout(patchified):
    kept = []
    for index in range(2):
        batch = _two_image_rollout_batch(patchified=patchified)
        apply_encoder_media_split(
            batch,
            EncoderMediaSplit(
                shard=EncoderShard(index=index, count=2),
                patch_dim=NT4_PATCH_DIM,
                temporal_patch_size=2,
                frame_weight=165,
            ),
        )
        pixels = batch["pixel_values"].tensors[0]
        assert pixels.ndim == (3 if patchified else 2)
        assert pixels.shape[-2] == PATCHES_PER_IMAGE
        assert batch[MIMO_MEDIA_OUTPUT_COUNTS_KEY].tensors[0].tolist() == [
            OUTPUTS_PER_IMAGE
        ]
        kept.append(pixels.reshape(-1, 4))

    whole = _two_image_rollout_batch(patchified=patchified)["pixel_values"].tensors[0]
    assert torch.equal(torch.cat(kept, dim=0), whole.reshape(-1, 4))


def test_split_rejects_pixels_that_disagree_with_imgs_sizes():
    batch = _two_image_rollout_batch(patchified=False)
    batch["pixel_values"] = PackedTensor(
        [torch.zeros((PATCHES_PER_IMAGE, 4))], dim_to_pack=0
    )
    with pytest.raises(ValueError, match="disagree with imgs_sizes"):
        apply_encoder_media_split(
            batch,
            EncoderMediaSplit(
                shard=EncoderShard(index=0, count=2),
                patch_dim=NT4_PATCH_DIM,
                temporal_patch_size=2,
                frame_weight=165,
            ),
        )


def test_encoder_shard_for_rank_lays_ranks_out_in_lanes():
    assigned = [
        encoder_shard_for_rank(dp_rank=rank, dp_size=8, language_dp_size=4)
        for rank in range(8)
    ]
    assert [(shard.index, lane) for shard, lane in assigned] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
    ]
    assert all(shard.count == 2 for shard, _ in assigned)


def test_encoder_shard_for_rank_requires_a_whole_multiple():
    with pytest.raises(ValueError, match="multiple of"):
        encoder_shard_for_rank(dp_rank=0, dp_size=6, language_dp_size=4)
