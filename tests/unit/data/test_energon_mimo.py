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

from typing import Any

import torch

from nemo_rl.data.energon.mimo_batch import mimo_data_plane_batch
from nemo_rl.data.multimodal_utils import (
    MIMO_MEDIA_OUTPUT_COUNTS_KEY,
    PackedTensor,
)


def _prepared_visual_batch() -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([[10, 99, 99, 20, 30, 0, 0, 0]]),
        "token_mask": torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0]]),
        "sample_mask": torch.tensor([1.0]),
        "cu_seqlens": [torch.tensor([0, 5], dtype=torch.int32)],
        "cu_seqlens_padded": [torch.tensor([0, 8], dtype=torch.int32)],
        "source_ids": [["sample-0"]],
        "source_lengths": [[5]],
        MIMO_MEDIA_OUTPUT_COUNTS_KEY: PackedTensor(
            [torch.tensor([2], dtype=torch.int32)], dim_to_pack=0
        ),
        "pixel_values": PackedTensor(
            [torch.empty((1, 3, 4, 8), dtype=torch.float32, device="cpu")],
            dim_to_pack=0,
        ),
        "imgs_sizes": PackedTensor(
            [torch.tensor([[4, 8]], dtype=torch.int32)], dim_to_pack=0
        ),
        "num_frames": PackedTensor(
            [torch.tensor([1], dtype=torch.long)], dim_to_pack=0
        ),
    }


def test_language_data_plane_batch_drops_all_multimodal_payloads() -> None:
    prepared = _prepared_visual_batch()

    language = mimo_data_plane_batch(prepared, role="language")
    encoder = mimo_data_plane_batch(prepared, role="encoder")

    assert not any(isinstance(value, PackedTensor) for value in language.values())
    assert encoder["pixel_values"] is prepared["pixel_values"]


def test_language_data_plane_batch_preserves_packed_boundaries() -> None:
    prepared = _prepared_visual_batch()
    boundaries = PackedTensor([torch.tensor([0, 4], dtype=torch.int32)], dim_to_pack=0)
    prepared["cu_seqlens"] = boundaries
    prepared["cu_seqlens_padded"] = boundaries
    language = mimo_data_plane_batch(prepared, role="language")
    assert language["cu_seqlens"] is boundaries
    assert language["cu_seqlens_padded"] is boundaries
    assert "pixel_values" not in language
