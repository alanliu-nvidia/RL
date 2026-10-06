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

# ruff: noqa: E402

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.energon")

from nemo_rl.data.energon.multimodal.task_encoders.nemotron4_visual import (
    _NemotronVisualProcessorAdapter,
)
from nemo_rl.data.energon.multimodal.types import MediaRef

pytestmark = pytest.mark.mcore


@pytest.fixture
def adapter() -> _NemotronVisualProcessorAdapter:
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(),
        apply_chat_template=lambda *args, **kwargs: "",
        image_processor=SimpleNamespace(
            min_num_patches=1,
            max_num_patches=12_100,
            _downsample_factor=2,
            patch_size=14,
            video_target_num_patches=1_024,
        ),
    )
    return _NemotronVisualProcessorAdapter(
        processor=processor,
        max_sequence_length=524_288,
        patch_dim=14,
        temporal_patch_size=1,
    )


@pytest.mark.parametrize("num_tokens_available", [-1, 0, 1, 16])
@pytest.mark.parametrize("num_images", [1, 3])
@pytest.mark.parametrize(
    "width,height", [(448, 224), (1, 1), (14_000, 14), (14, 14_000)]
)
def test_image_grids_remain_valid_with_small_budgets(
    adapter: _NemotronVisualProcessorAdapter,
    num_tokens_available: int,
    num_images: int,
    width: int,
    height: int,
) -> None:
    occurrences = [
        (
            index,
            MediaRef(
                modality="image",
                value=f"image-{index}",
                metadata=(("width", width), ("height", height)),
            ),
        )
        for index in range(num_images)
    ]
    plans = adapter._plan_visual_occurrences(
        occurrences,
        video_selections={},
        num_tokens_available=num_tokens_available,
        data_augment=False,
    )

    assert len(plans) == num_images
    for plan in plans:
        grid_width, grid_height = plan.patch_sizes[0]
        assert grid_width >= 2 and grid_height >= 2
        assert grid_width % 2 == 0 and grid_height % 2 == 0
        assert plan.num_embeddings == (grid_width // 2) * (grid_height // 2)
        assert plan.num_embeddings >= 1


def test_normal_image_budget_keeps_expected_grid(
    adapter: _NemotronVisualProcessorAdapter,
) -> None:
    plans = adapter._plan_visual_occurrences(
        [(0, MediaRef("image", "image", (("width", 448), ("height", 224))))],
        video_selections={},
        num_tokens_available=16,
        data_augment=False,
    )

    assert plans[0].patch_sizes == ((12, 4),)
    assert plans[0].num_embeddings == 12
