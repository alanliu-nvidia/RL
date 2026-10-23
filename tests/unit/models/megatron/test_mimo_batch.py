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

"""MiMo role batches match the pinned Core model and native Pixtral APIs."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.data.multimodal_utils import MIMO_MEDIA_OUTPUT_COUNTS_KEY, PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.train import _mimo_encoder_batch, _mimo_language_batch

pytestmark = pytest.mark.mcore


def test_mimo_language_batch_preserves_packed_positions_and_boundaries():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    boundaries = torch.tensor([0, 3, 6], dtype=torch.int32)
    microbatch = SimpleNamespace(
        input_ids=tokens,
        position_ids=positions,
        packed_seq_params=SimpleNamespace(
            cu_seqlens_q=boundaries, cu_seqlens_q_padded=boundaries, max_seqlen_q=3
        ),
        data_dict={
            "token_mask": torch.tensor([[0, 0, 1, 0, 0, 1]]),
            "sample_mask": torch.ones(1),
        },
    )
    batch = _mimo_language_batch(microbatch, SimpleNamespace(special_token_ids={}))
    assert batch["position_ids"] is positions
    torch.testing.assert_close(batch["cu_seqlens"], boundaries.unsqueeze(0))
    torch.testing.assert_close(batch["cu_seqlens_padded"], boundaries.unsqueeze(0))
    assert batch["max_seqlen"] == 3
    assert isinstance(batch["max_seqlen"], torch.Tensor)
    assert batch["max_seqlen"].dtype == torch.int32
    assert "packing_kwargs" not in batch
    assert batch["labels"].tolist() == [[-100, 3, -100, -100, 6, -100]]
    assert batch["loss_mask"].tolist() == [[0, 1, 0, 0, 1, 0]]


def test_mimo_language_batch_packs_masks_left_on_their_rows():
    # Runtime packing packs the tokens at microbatch time and leaves the masks
    # on the rows they came from, so they reach here in the unpacked layout.
    rows = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    packed = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 7]])
    # Every PackedSeqParams boundary is a padded one; the three-token first
    # sequence is padded to four before the second one starts.
    padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    microbatch = SimpleNamespace(
        input_ids=packed,
        position_ids=None,
        packed_seq_params=SimpleNamespace(
            cu_seqlens_q=padded, cu_seqlens_q_padded=padded, max_seqlen_q=4
        ),
        data_dict={
            "input_ids": rows,
            "input_lengths": torch.tensor([3, 4]),
            "token_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            "sample_mask": torch.ones(2),
        },
    )
    batch = _mimo_language_batch(microbatch, SimpleNamespace(special_token_ids={}))
    assert batch["loss_mask"].shape == packed.shape
    # Each sequence predicts only its own tokens: position 2 is the last of the
    # first sequence and must not be asked to predict the second sequence's 4.
    assert batch["loss_mask"].tolist() == [[1, 1, 0, 0, 1, 1, 1, 0]]
    assert batch["labels"].tolist() == [[2, 3, -100, -100, 5, 6, 7, -100]]


def test_mimo_language_batch_rejects_masks_that_fit_no_layout():
    boundaries = torch.tensor([0, 4], dtype=torch.int32)
    microbatch = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        position_ids=None,
        packed_seq_params=SimpleNamespace(
            cu_seqlens_q=boundaries, cu_seqlens_q_padded=boundaries, max_seqlen_q=4
        ),
        data_dict={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "input_lengths": torch.tensor([3]),
            "token_mask": torch.tensor([[1, 1]]),
            "sample_mask": torch.ones(1),
        },
    )
    with pytest.raises(ValueError, match="do not match the input batch rows"):
        _mimo_language_batch(microbatch, SimpleNamespace(special_token_ids={}))


def test_mimo_language_batch_withholds_labels_for_logits():
    tokens = torch.tensor([[1, 2, 3]])
    microbatch = SimpleNamespace(
        input_ids=tokens,
        position_ids=torch.tensor([[0, 1, 2]]),
        packed_seq_params=None,
        data_dict={
            "token_mask": torch.tensor([[1, 1, 1]]),
            "sample_mask": torch.ones(1),
        },
    )
    batch = _mimo_language_batch(
        microbatch, SimpleNamespace(special_token_ids={}), with_labels=False
    )
    # Labels make the language model return per-token losses; every NeMo-RL
    # post-processor consumes the logits it returns without them.
    assert batch["labels"] is None
    assert batch["loss_mask"].tolist() == [[1, 1, 0]]


def test_mimo_pixtral_batch_sends_x_and_no_input_ids() -> None:
    pixels = torch.ones(4, 588, dtype=torch.bfloat16)
    microbatch = SimpleNamespace(
        input_ids=torch.tensor([[1, 99, 99, 99, 99, 2]]),
        data_dict=BatchedDataDict(
            {
                "pixel_values": PackedTensor([pixels], dim_to_pack=0),
                "imgs_sizes": PackedTensor([torch.tensor([[28, 28]])], dim_to_pack=0),
            }
        ),
    )
    model = SimpleNamespace(
        modality_submodules={
            "images": SimpleNamespace(
                encoders={"pixtral": SimpleNamespace(patch_dim=14)}
            )
        }
    )
    batch = _mimo_encoder_batch(microbatch, model)
    # Encoder ranks send no input_ids: the pack's image placeholders can also
    # cover media of other encoder ranks of the lane.
    assert batch["input_ids"] is None
    payload = batch["modality_inputs"]["images"]["pixtral"]
    torch.testing.assert_close(payload["x"], pixels.unsqueeze(0))
    assert set(payload) == {"x", "imgs_sizes", "packed_seq_params"}
    assert payload["imgs_sizes"].device.type == "cpu"
    assert payload["packed_seq_params"].cu_seqlens_q.tolist() == [0, 4]
    assert payload["packed_seq_params"].max_seqlen_q == 4
    microbatch.data_dict["imgs_sizes"] = PackedTensor(
        [torch.tensor([[14, 28]])], dim_to_pack=0
    )
    with pytest.raises(ValueError, match="disagree with imgs_sizes"):
        _mimo_encoder_batch(microbatch, model)


def test_mimo_encoder_batch_passes_frames_and_output_counts() -> None:
    model = SimpleNamespace(
        modality_submodules={
            "images": SimpleNamespace(
                encoders={"pixtral": SimpleNamespace(patch_dim=14)}
            )
        }
    )
    # One image followed by a two-frame video; temporal patch size is one.
    pixels = torch.arange(12 * 588, dtype=torch.float32).reshape(12, 588)
    data = BatchedDataDict(
        {
            "pixel_values": PackedTensor([pixels], dim_to_pack=0),
            "imgs_sizes": PackedTensor(
                [torch.tensor([[28, 28], [28, 28], [28, 28]])], dim_to_pack=0
            ),
            "num_frames": PackedTensor([torch.tensor([1, 2])], dim_to_pack=0),
            MIMO_MEDIA_OUTPUT_COUNTS_KEY: PackedTensor(
                [torch.tensor([4, 4, 4])], dim_to_pack=0
            ),
        }
    )
    microbatch = SimpleNamespace(
        input_ids=torch.ones(1, 16, dtype=torch.long), data_dict=data
    )
    payload = _mimo_encoder_batch(microbatch, model)["modality_inputs"]["images"][
        "pixtral"
    ]
    assert set(payload) == {
        "x",
        "imgs_sizes",
        "packed_seq_params",
        "num_frames",
        "expected_output_counts",
    }
    torch.testing.assert_close(payload["x"], pixels.unsqueeze(0))
    assert payload["num_frames"].tolist() == [1, 2]
    assert payload["num_frames"].dtype == torch.int32
    assert payload["expected_output_counts"].tolist() == [4, 4, 4]
    assert payload["expected_output_counts"].dtype == torch.int32
    assert payload["expected_output_counts"].device.type == "cpu"
    assert payload["packed_seq_params"].cu_seqlens_q.tolist() == [0, 4, 8, 12]

    # Without output counts the payload has no expected_output_counts.
    del data[MIMO_MEDIA_OUTPUT_COUNTS_KEY]
    payload = _mimo_encoder_batch(microbatch, model)["modality_inputs"]["images"][
        "pixtral"
    ]
    assert "expected_output_counts" not in payload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_mimo_pixtral_keeps_geometry_on_cpu_and_attention_boundaries_on_cuda():
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 99, 99, 99, 99, 2]]),
            "pixel_values": PackedTensor([torch.ones(4, 588)], dim_to_pack=0),
            "imgs_sizes": PackedTensor([torch.tensor([[28, 28]])], dim_to_pack=0),
        }
    ).to("cuda")
    assert data["imgs_sizes"].as_tensor().device.type == "cpu"
    model = SimpleNamespace(
        modality_submodules={
            "images": SimpleNamespace(
                encoders={"pixtral": SimpleNamespace(patch_dim=14)}
            )
        }
    )
    payload = _mimo_encoder_batch(
        SimpleNamespace(input_ids=data["input_ids"], data_dict=data), model
    )["modality_inputs"]["images"]["pixtral"]
    assert payload["x"].device.type == "cuda"
    assert payload["imgs_sizes"].device.type == "cpu"
    assert payload["packed_seq_params"].cu_seqlens_q.device.type == "cuda"
    assert payload["packed_seq_params"].cu_seqlens_q.tolist() == [0, 4]
