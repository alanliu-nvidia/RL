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

from copy import deepcopy
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from PIL import Image

pytest.importorskip("megatron.energon")

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonTaskEncoderConfig
from nemo_rl.data.energon.mimo_image_split import EncoderShard
from nemo_rl.data.energon.multimodal.packing import pack_selected_samples
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    HFMultimodalSFTProcessorAdapter,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron4_multimodal import (
    Nemotron4MultiModalTaskEncoder,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron4_visual import (
    NemotronVisualPlan,
)
from nemo_rl.data.energon.multimodal.types import EncodedSFTSample, MediaRef
from nemo_rl.data.energon.sft_dataloader import _task_encoder, _v2_topology
from nemo_rl.data.multimodal_utils import MIMO_MEDIA_OUTPUT_COUNTS_KEY, PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytestmark = pytest.mark.mcore


def _encoder(
    role: str | None, dtype: torch.dtype | None = None
) -> Nemotron4MultiModalTaskEncoder:
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0),
        apply_chat_template=lambda *args, **kwargs: "",
    )
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=128,
        add_bos=False,
        add_eos=False,
        add_generation_prompt=False,
    )
    return Nemotron4MultiModalTaskEncoder(
        adapter=adapter,
        cooker_functions=[],
        include_source_ids=True,
        tokenizer=processor.tokenizer,
        patch_dim=2,
        temporal_patch_size=2,
        video_min_num_frames=2,
        video_max_num_frames=2,
        mimo_role=role,
        pixel_transport_dtype=dtype,
    )


def test_language_video_uses_plan_without_reading_payload() -> None:
    class LazyVideo:
        def get(self, sample: object) -> None:
            pytest.fail("Language loaders must not read video payloads.")

    encoder = _encoder("language")
    plan = NemotronVisualPlan(
        message_index=0,
        media_index=0,
        modality="video",
        patch_sizes=((2, 2), (2, 2)),
        embedding_widths=(1,),
        frame_timestamps=(0.25, 0.75),
    )
    values, width = encoder._multimodal_adapter._process_media(
        MediaRef("video", LazyVideo()), plan, object()
    )
    assert width == 1
    assert values["pixel_values"].tensors[0].is_meta
    assert values["imgs_sizes"].as_tensor().tolist() == [[4, 4], [4, 4]]
    assert values["num_frames"].as_tensor().tolist() == [2]
    assert values[MIMO_MEDIA_OUTPUT_COUNTS_KEY].as_tensor().tolist() == [1]


@pytest.mark.parametrize("role", [None, "encoder", "language"])
def test_image_output_counts_follow_mimo_role(role: str | None) -> None:
    encoder = _encoder(role)
    plan = NemotronVisualPlan(
        message_index=0,
        media_index=0,
        modality="image",
        patch_sizes=((4, 2),),
        embedding_widths=(2,),
    )
    values, width = encoder._multimodal_adapter._process_media(
        MediaRef("image", Image.new("RGB", (8, 4))), plan, object()
    )
    assert width == 2
    assert values["imgs_sizes"].as_tensor().tolist() == [[4, 8]]
    if role is None:
        assert MIMO_MEDIA_OUTPUT_COUNTS_KEY not in values
    else:
        assert values[MIMO_MEDIA_OUTPUT_COUNTS_KEY].as_tensor().tolist() == [2]


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.bfloat16])
def test_pixels_cast_before_transport_preserves_geometry(
    dtype: torch.dtype | None,
) -> None:
    encoder = _encoder("encoder", dtype)
    normalized = torch.linspace(-2, 2, 48).reshape(1, 3, 4, 4)
    pixels = PackedTensor(
        [normalized, torch.empty_like(normalized, device="meta"), None],
        dim_to_pack=0,
        preprocess_mode="patchify",
        preprocess_kwargs={"patch_dim": 2},
    )
    sizes = PackedTensor(
        [torch.tensor([[4, 4]], dtype=torch.int32), None, None], dim_to_pack=0
    )
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
    batch = BatchedDataDict(
        {"input_ids": ids, "pixel_values": pixels, "imgs_sizes": sizes}
    )
    assert encoder.encode_batch(batch) is batch
    converted = batch["pixel_values"]
    expected_dtype = dtype or torch.float32
    torch.testing.assert_close(converted.tensors[0], normalized.to(expected_dtype))
    assert converted.tensors[1].is_meta
    assert converted.tensors[2] is None
    assert converted.preprocess_mode == "patchify"
    assert converted.preprocess_kwargs == {"patch_dim": 2}
    assert converted.dim_to_pack == 0
    assert converted.tensors[0].numel() * converted.tensors[0].element_size() == (
        96 if dtype == torch.bfloat16 else 192
    )
    assert normalized.dtype == torch.float32
    assert batch["imgs_sizes"].tensors[0].dtype == torch.int32
    assert batch["input_ids"] is ids


@pytest.mark.parametrize("packed", [False, True])
def test_language_batch_combines_real_images_and_meta_video(packed: bool) -> None:
    encoder = _encoder("language")
    samples = []
    for index, device in enumerate(["cpu", "meta"]):
        samples.append(
            EncodedSFTSample(
                __key__=str(index),
                __restore_key__=(str(index),),
                message_log=[
                    {
                        "role": "assistant",
                        "token_ids": torch.tensor([1, 2, 3]),
                        "token_loss_mask": torch.tensor([0.0, 1.0, 1.0]),
                        "pixel_values": PackedTensor(
                            torch.empty((1, 3, 4, 4), device=device), dim_to_pack=0
                        ),
                    }
                ],
                length=3,
                packing_cost=3,
                loss_multiplier=1.0,
                group_key=("visual",),
                sample_key=str(index),
            )
        )
    rows = (
        [
            pack_selected_samples(
                samples, pack_capacity=16, sequence_length_pad_multiple=4
            )
        ]
        if packed
        else samples
    )
    batch = encoder.batch(rows)
    pixels = [
        message["pixel_values"] for sample in samples for message in sample.message_log
    ]
    assert all(value.tensors[0].is_meta for value in pixels)
    if packed:
        assert batch["input_lengths"].tolist() == [8]
        assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 1, 2, 3, 0]]
        assert batch["cu_seqlens_padded"].as_tensor().tolist() == [0, 4, 8]
    else:
        assert batch["source_ids"] == ["0", "1"]
        assert len(batch["message_log"]) == 2


@pytest.mark.parametrize("role", ["encoder", "language"])
def test_nm4_loader_passes_role_and_precision(
    role: Literal["encoder", "language"],
) -> None:
    adapter = _encoder(None)._multimodal_adapter
    config = EnergonLoaderConfig(
        model_family="nemotron",
        task_encoder=EnergonTaskEncoderConfig(name="nemotron4_multimodal"),
    )
    encoder = _task_encoder(
        loader_config=config,
        adapter=HFMultimodalSFTProcessorAdapter(
            processor=adapter.processor,
            max_sequence_length=128,
            add_bos=False,
            add_eos=False,
            add_generation_prompt=False,
        ),
        include_source_ids=True,
        max_sequence_length=128,
        tokenizer=adapter.processor.tokenizer,
        only_unmask_final=False,
        mimo_role=role,
        pixel_transport_dtype=torch.bfloat16,
    )
    assert encoder._mimo_role == role
    assert encoder._multimodal_adapter.mimo_role == role
    assert encoder._pixel_transport_dtype == torch.bfloat16


def test_generic_loader_supports_mimo_role_without_nm4_options() -> None:
    config = EnergonLoaderConfig(model_family="qwen")
    encoder = _task_encoder(
        loader_config=config,
        adapter=object(),
        include_source_ids=True,
        max_sequence_length=128,
        tokenizer=SimpleNamespace(pad_token_id=0),
        only_unmask_final=False,
        mimo_role="encoder",
    )
    assert encoder.adapter is not None
    with pytest.raises(ValueError, match="pixel_transport_dtype requires"):
        _task_encoder(
            loader_config=config,
            adapter=object(),
            include_source_ids=True,
            max_sequence_length=128,
            tokenizer=SimpleNamespace(pad_token_id=0),
            only_unmask_final=False,
            mimo_role="encoder",
            pixel_transport_dtype=torch.bfloat16,
        )


def test_role_loader_states_have_distinct_identities() -> None:
    config = EnergonLoaderConfig(model_family="nemotron")
    ordinary, language, encoder = [
        _v2_topology(
            loader_config=config,
            placement_fingerprint="placement",
            logical_rank=0,
            logical_world_size=1,
            state_identity=role,
        )
        for role in (None, "language", "encoder")
    ]
    assert "state_identity" not in ordinary
    assert language != encoder != ordinary
    assert (
        _encoder(None)._multimodal_adapter.fingerprint
        != _encoder("language")._multimodal_adapter.fingerprint
    )


@pytest.mark.parametrize("role", [None, "encoder", "language"])
def test_packed_rows_use_source_lengths_in_every_role(role: str | None) -> None:
    encoder = _encoder(role)
    samples = [
        EncodedSFTSample(
            __key__=str(index),
            __restore_key__=(str(index),),
            message_log=[
                {
                    "role": "assistant",
                    "token_ids": torch.arange(1, length + 1),
                    "token_loss_mask": torch.ones(length),
                }
            ],
            length=length,
            packing_cost=length,
            loss_multiplier=1.0,
            group_key=("text",),
            sample_key=str(index),
        )
        for index, length in enumerate([5, 3, 3])
    ]
    packs = [
        pack_selected_samples(
            samples[:2], pack_capacity=128, sequence_length_pad_multiple=4
        ),
        pack_selected_samples(
            samples[2:], pack_capacity=128, sequence_length_pad_multiple=4
        ),
    ]
    batch = encoder.batch(packs)
    assert batch["input_ids"].shape == (2, 12)
    assert batch["input_lengths"].tolist() == [12, 4]
    assert batch.slice(0, 1)["cu_seqlens"].as_tensor().tolist() == [0, 5, 8]
    assert batch.slice(0, 1)["cu_seqlens_padded"].as_tensor().tolist() == [0, 8, 12]
    assert batch.slice(1, 2)["cu_seqlens_padded"].as_tensor().tolist() == [0, 4]
    assert batch["token_mask"][1, 3:].count_nonzero() == 0


_IMAGE_TOKEN = 9


def _split_encoder(shard: EncoderShard | None) -> Nemotron4MultiModalTaskEncoder:
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        unk_token_id=None,
        convert_tokens_to_ids=lambda token: _IMAGE_TOKEN if token == "<image>" else 1,
    )
    processor = SimpleNamespace(
        tokenizer=tokenizer, apply_chat_template=lambda *args, **kwargs: ""
    )
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=128,
        add_bos=False,
        add_eos=False,
        add_generation_prompt=False,
    )
    return Nemotron4MultiModalTaskEncoder(
        adapter=adapter,
        cooker_functions=[],
        include_source_ids=True,
        tokenizer=tokenizer,
        patch_dim=2,
        temporal_patch_size=2,
        video_min_num_frames=2,
        video_max_num_frames=2,
        mimo_role="encoder",
        mimo_encoder_shard=shard,
        encoder_media_split_frame_weight=0,
    )


def _image_sample(
    encoder: Nemotron4MultiModalTaskEncoder,
    key: str,
    tokens: list[int],
    images: list[tuple[int, int]],
) -> EncodedSFTSample:
    """One sample with images decoded by the task encoder's own media path."""
    media: dict[str, list[PackedTensor]] = {}
    for index, (patch_width, patch_height) in enumerate(images):
        plan = NemotronVisualPlan(
            message_index=0,
            media_index=index,
            modality="image",
            patch_sizes=((patch_width, patch_height),),
            embedding_widths=(patch_width * patch_height // 4,),
        )
        pixels = Image.new(
            "RGB", (patch_width * 2, patch_height * 2), (index * 40, 90, 200)
        )
        values, _ = encoder._multimodal_adapter._process_media(
            MediaRef("image", pixels), plan, object()
        )
        for name, value in values.items():
            media.setdefault(name, []).append(value)
    message = {
        "role": "assistant",
        "token_ids": torch.tensor(tokens),
        "token_loss_mask": torch.ones(len(tokens)),
        **{name: PackedTensor.merge_segments(values) for name, values in media.items()},
    }
    return EncodedSFTSample(
        __key__=key,
        __restore_key__=(key,),
        message_log=[message],
        length=len(tokens),
        packing_cost=len(tokens),
        loss_multiplier=1.0,
        group_key=("visual",),
        sample_key=key,
    )


def test_encoder_shards_split_a_real_pack_and_keep_the_text() -> None:
    """Shard task encoders of one lane batch the same pack; joined media = the whole pack."""
    unsplit = _split_encoder(None)
    shards = [_split_encoder(EncoderShard(index=index, count=2)) for index in range(2)]
    image = _IMAGE_TOKEN
    # Images of 8, 4 and 16 patches; 2, 1 and 4 output rows after the pixel shuffle.
    samples = [
        _image_sample(unsplit, "a", [5, image, image, 6, image], [(4, 2), (2, 2)]),
        _image_sample(unsplit, "b", [7, image, image, image, image, 8], [(4, 4)]),
    ]
    pack = pack_selected_samples(
        samples, pack_capacity=16, sequence_length_pad_multiple=4
    )
    full = unsplit.encode_batch(unsplit.batch([deepcopy(pack)]))
    parts = [shard.encode_batch(shard.batch([deepcopy(pack)])) for shard in shards]

    for part in parts:
        for key in ("input_ids", "input_lengths", "token_mask", "sample_mask"):
            assert torch.equal(part[key], full[key]), key
        assert part["cu_seqlens"].as_tensor().tolist() == (
            full["cu_seqlens"].as_tensor().tolist()
        )
        assert part["source_ids"] == full["source_ids"]
    # The cut [8, 4 | 16] has the smallest maximum cost.
    assert parts[0]["imgs_sizes"].as_tensor().tolist() == [[4, 8], [4, 4]]
    assert parts[1]["imgs_sizes"].as_tensor().tolist() == [[8, 8]]
    for key, dim in (
        ("pixel_values", 1),
        ("imgs_sizes", 0),
        ("num_frames", 0),
        (MIMO_MEDIA_OUTPUT_COUNTS_KEY, 0),
    ):
        joined = torch.cat([part[key].as_tensor() for part in parts], dim=dim)
        assert torch.equal(joined, full[key].as_tensor()), key
