# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.data.multimodal_utils import (
    _materialize_ragged_pixel_values,
    extract_multimodal_model_inputs,
)


class PixtralProcessor:
    tokenizer = SimpleNamespace(model_input_names=["input_ids", "attention_mask"])
    image_processor = SimpleNamespace(
        model_input_names=["pixel_values", "image_sizes"], patch_size=16
    )
    model_input_names = ["input_ids", "pixel_values", "image_sizes"]


@pytest.mark.parametrize("alias", ["image_sizes", "imgs_sizes", "both", "neither"])
def test_stacked_images_emit_only_canonical_sizes(alias: str) -> None:
    sizes = torch.tensor([[16, 32], [16, 32]])
    processed = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.zeros(2, 3, 16, 32),
    }
    if alias in {"image_sizes", "both"}:
        processed["image_sizes"] = sizes
    if alias in {"imgs_sizes", "both"}:
        processed["imgs_sizes"] = sizes.to(torch.int32)
    inputs = extract_multimodal_model_inputs(PixtralProcessor(), processed)
    assert "image_sizes" not in inputs
    torch.testing.assert_close(inputs["imgs_sizes"].as_tensor(), sizes.to(torch.int32))
    torch.testing.assert_close(
        inputs["num_frames"].as_tensor(), torch.ones(2, dtype=torch.int32)
    )
    assert ("image_sizes" in processed) == (alias in {"image_sizes", "both"})


def test_ragged_images_preserve_processor_sizes_before_patchification() -> None:
    processed = {
        "input_ids": [[1, 2, 3]],
        "pixel_values": [torch.zeros(3, 16, 32), torch.zeros(3, 32, 16)],
        "image_sizes": [[16, 32], [32, 16]],
    }
    materialized = _materialize_ragged_pixel_values(processed, PixtralProcessor())
    inputs = extract_multimodal_model_inputs(PixtralProcessor(), materialized)
    assert "image_sizes" not in materialized and "image_sizes" not in inputs
    torch.testing.assert_close(
        inputs["imgs_sizes"].as_tensor(),
        torch.tensor([[16, 32], [32, 16]], dtype=torch.int32),
    )
    assert inputs["pixel_values"].as_tensor().shape == (1, 4, 768)


def test_conflicting_size_aliases_are_rejected() -> None:
    processed = {
        "input_ids": torch.tensor([[1]]),
        "pixel_values": torch.zeros(1, 3, 16, 32),
        "image_sizes": torch.tensor([[16, 32]]),
        "imgs_sizes": torch.tensor([[32, 16]]),
    }
    with pytest.raises(ValueError, match="image_sizes and imgs_sizes disagree"):
        extract_multimodal_model_inputs(PixtralProcessor(), processed)


def test_other_processors_keep_their_image_sizes_field() -> None:
    processor = SimpleNamespace(
        tokenizer=PixtralProcessor.tokenizer,
        image_processor=PixtralProcessor.image_processor,
        model_input_names=PixtralProcessor.model_input_names,
    )
    processed = {
        "input_ids": torch.tensor([[1]]),
        "pixel_values": torch.zeros(1, 3, 16, 32),
        "image_sizes": torch.tensor([[16, 32]]),
    }
    inputs = extract_multimodal_model_inputs(processor, processed)
    assert "image_sizes" in inputs and "imgs_sizes" not in inputs
