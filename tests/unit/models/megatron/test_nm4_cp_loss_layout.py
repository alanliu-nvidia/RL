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

"""Check NM4 zigzag outputs, MTP inputs, losses and gradients against a reference."""

import functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import yaml

from nemo_rl.algorithms.loss import prepare_packed_loss_input
from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _check_layout_loss(
    rank: int, world_size: int, *, prepacked: bool, padding: bool = False
) -> dict[str, float]:
    # MCore is optional outside the Megatron test lane.
    from megatron.core.context_parallel.layout import ContextParallelLayoutManager
    from megatron.core.context_parallel.utils import get_batches_on_this_cp_rank

    tp_group = None
    for peer in range(world_size):
        group = dist.new_group([peer])
        if peer == rank:
            tp_group = group
    bounds = torch.tensor([0, 128, 320, 512], dtype=torch.int32, device="cuda")
    logical_bounds = (
        torch.tensor([0, 117, 291, 462], dtype=torch.int32, device="cuda")
        if padding
        else bounds
    )
    tokens = (torch.arange(512, device="cuda") % 63 + 1).unsqueeze(0)
    if padding:
        for index, start in enumerate(bounds[:-1].tolist()):
            length = int(logical_bounds[index + 1] - logical_bounds[index])
            tokens[:, start + length : int(bounds[index + 1])] = 0
    batch = {
        "tokens": tokens,
        "cu_seqlens": logical_bounds.unsqueeze(0),
        "cu_seqlens_padded": bounds.unsqueeze(0),
        "max_seqlen": torch.tensor([192], dtype=torch.int32, device="cuda"),
    }
    cfg = yaml.safe_load(
        Path(
            "examples/configs/sft_v2_tests/nm4_generalist_16n_mxfp8_cp64ep64_gbs1.yaml"
        ).read_text()
    )
    overrides = cfg["policy"]["megatron_cfg"]["model_overrides"]
    boundary = overrides["linear_cp_layout"]
    attention = overrides["attention_cp_layout"]
    assert boundary == "contiguous" and attention == "zigzag"
    cp_batch = get_batches_on_this_cp_rank(
        batch,
        boundary_layout=boundary,
        is_hybrid_cp=False,
        cp_group=dist.group.WORLD,
        additional_layouts={attention},
        tp_group=tp_group,
        tp_cp_group=dist.group.WORLD,
    )
    expected_tokens = tokens.chunk(world_size, dim=1)[rank]
    torch.testing.assert_close(cp_batch.get_batch()["tokens"], expected_tokens)
    manager = ContextParallelLayoutManager(
        layer_layouts=(attention,),
        boundary_layout=boundary,
        sequence_parallel=False,
        cp_group=dist.group.WORLD,
        tp_group=tp_group,
        tp_cp_group=dist.group.WORLD,
    )
    hidden = expected_tokens.T.unsqueeze(-1).float()
    attention_hidden = manager.prepare_layer_input(0, hidden, cp_batch.thd_plan)
    output_hidden = manager.finalize_layer_output(
        0, attention_hidden, cp_batch.thd_plan
    )
    torch.testing.assert_close(output_hidden, hidden)

    shifted = tokens.clone()
    mask = torch.ones_like(tokens, dtype=torch.float32)
    for start, end in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        shifted[:, start:end] = tokens[:, start:end].roll(-1, 1)
        mask[:, start] = 0
    # A nonuniform mask exposes incorrect reconstruction even when every
    # correctly selected token has similar likelihood.
    mask[:, 13::7] = 0
    mask[tokens == 0] = 0
    full_logits = torch.randn(
        1,
        512,
        128,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    full_logits.requires_grad_()
    reference_logprobs = (
        full_logits.log_softmax(-1)
        .gather(-1, shifted.unsqueeze(-1))
        .squeeze(-1)[:, :-1]
    )
    reference_loss = -(reference_logprobs * mask[:, 1:]).sum() / mask[:, 1:].sum()
    reference_loss.backward()
    logits = (
        full_logits.detach().chunk(world_size, dim=1)[rank].clone().requires_grad_()
    )
    if prepacked:
        data = BatchedDataDict(
            {
                "input_ids": tokens,
                "token_mask": mask,
                "sample_mask": torch.ones(1, device="cuda"),
                "cu_seqlens": logical_bounds.unsqueeze(0),
            }
        )
    else:
        rows = torch.zeros(3, 192, dtype=torch.long, device="cuda")
        masks = torch.zeros_like(rows, dtype=torch.float32)
        for row, (start, end) in enumerate(
            zip(bounds[:-1].tolist(), bounds[1:].tolist())
        ):
            length = int(logical_bounds[row + 1] - logical_bounds[row])
            rows[row, :length] = tokens[0, start : start + length]
            masks[row, :length] = mask[0, start : start + length]
        data = BatchedDataDict(
            {
                "input_ids": rows,
                "token_mask": masks,
                "sample_mask": torch.ones(3, device="cuda"),
            }
        )
    from megatron.bridge.models.experimental_nm4_llava_provider import NM4LlavaModel
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionBlock,
    )

    from nemo_rl.distributed.model_utils import (
        from_parallel_logits_to_logprobs_packed_sequences,
    )

    model = object.__new__(NM4LlavaModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model.post_process = True
    model.context_parallel_lm = world_size
    model.sequence_parallel_lm = False
    model.config = SimpleNamespace(
        linear_cp_layout=boundary, attention_cp_layout=attention, mtp_num_layers=2
    )
    model.pg_collection = SimpleNamespace(
        cp=dist.group.WORLD, tp=tp_group, tp_cp=dist.group.WORLD
    )
    model.language_model = MagicMock(return_value=logits)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical_bounds,
        cu_seqlens_kv=logical_bounds,
        cu_seqlens_q_padded=bounds,
        cu_seqlens_kv_padded=bounds,
        max_seqlen_q=192,
        max_seqlen_kv=192,
    )
    output, output_mask = model(
        input_ids=tokens,
        position_ids=tokens + 1000,
        loss_mask=mask,
        padding_mask=mask == 0,
        packed_seq_params=params,
    )
    passed = model.language_model.call_args.kwargs
    actual_batch = passed["cp_batch"]
    assert actual_batch.boundary_layout == boundary
    torch.testing.assert_close(passed["input_ids"], expected_tokens)
    assert passed["packed_seq_params"] is actual_batch.get_packed_seq_params(boundary)
    torch.testing.assert_close(
        output_mask, actual_batch.get_batch("zigzag")["loss_mask"]
    )
    expected_output = torch.cat(
        [
            torch.cat(
                (
                    full_logits.detach()[:, start:end].chunk(2 * world_size, 1)[rank],
                    full_logits.detach()[:, start:end].chunk(2 * world_size, 1)[
                        2 * world_size - rank - 1
                    ],
                ),
                1,
            )
            for start, end in zip(bounds[:-1].tolist(), bounds[1:].tolist())
        ],
        1,
    )
    torch.testing.assert_close(output, expected_output)
    mtp_self = SimpleNamespace(
        config=model.config,
        cp_group=dist.group.WORLD,
        sequence_parallel=False,
        tp_group=tp_group,
        tp_cp_group=dist.group.WORLD,
    )
    mtp = MultiTokenPredictionBlock.prepare_cp_layout(
        mtp_self,
        input_ids=passed["input_ids"],
        position_ids=passed["position_ids"],
        hidden_states=expected_tokens.T.unsqueeze(-1).float(),
        labels=None,
        loss_mask=passed["loss_mask"],
        mtp_input_mask=None,
        packed_seq_params=passed["packed_seq_params"],
        cp_batch=actual_batch,
    )
    torch.testing.assert_close(
        mtp.hidden_states,
        actual_batch.get_batch("zigzag")["tokens"].T.unsqueeze(-1).float(),
    )
    torch.testing.assert_close(mtp.loss_mask, output_mask)
    assert mtp.packed_seq_params is actual_batch.get_packed_seq_params("zigzag")
    # Exercise the native labels-provided token-loss output as well as logits.
    model.language_model.return_value = logits[..., 0]
    token_losses, token_mask = model(
        input_ids=tokens, labels=tokens, loss_mask=mask, packed_seq_params=params
    )
    torch.testing.assert_close(token_losses, expected_output[..., 0])
    torch.testing.assert_close(token_mask, output_mask)
    model.language_model.return_value = logits
    # Direct logprob callers must work with the default zigzag path.
    logprobs = from_parallel_logits_to_logprobs_packed_sequences(
        output,
        tokens,
        bounds,
        512,
        0,
        128,
        tp_group,
        cp_group=dist.group.WORLD,
        chunk_size=32,
        return_packed_layout=True,
    )
    torch.testing.assert_close(logprobs, reference_logprobs, atol=1e-5, rtol=1e-5)
    loss_fn = NLLLossFn()
    loss_input, prepared = prepare_packed_loss_input(
        output,
        data,
        loss_fn,
        bounds,
        bounds,
        vocab_parallel_rank=0,
        vocab_parallel_group=tp_group,
        context_parallel_group=dist.group.WORLD,
        chunk_size=32,
    )
    loss, _ = loss_fn(
        **loss_input,
        data=prepared,
        global_valid_seqs=None,
        global_valid_toks=mask[:, 1:].sum(),
    )
    if not prepacked:
        from nemo_rl.algorithms.loss import prepare_loss_input
        from nemo_rl.algorithms.loss.wrapper import SequencePackingLossWrapper

        wrapper = SequencePackingLossWrapper(
            loss_fn,
            prepare_loss_input,
            logical_bounds,
            bounds,
            vocab_parallel_rank=0,
            vocab_parallel_group=tp_group,
            context_parallel_group=dist.group.WORLD,
        )
        wrapper_loss, _ = wrapper(
            output, data, global_valid_seqs=None, global_valid_toks=mask[:, 1:].sum()
        )
        torch.testing.assert_close(wrapper_loss, reference_loss, atol=1e-5, rtol=1e-5)
    (loss / world_size).backward()
    torch.testing.assert_close(loss, reference_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        logits.grad,
        full_logits.grad.chunk(world_size, dim=1)[rank],
        atol=1e-6,
        rtol=1e-5,
    )
    return {
        "loss": float(loss.detach()),
        "reference_loss": float(reference_loss.detach()),
        "max_gradient_error": float(
            (logits.grad - full_logits.grad.chunk(world_size, dim=1)[rank]).abs().max()
        ),
        "max_logprob_error": float(
            (logprobs.detach() - reference_logprobs.detach()).abs().max()
        ),
    }


@pytest.mark.mcore
@pytest.mark.parametrize("prepacked", [True, False])
@pytest.mark.parametrize("padding", [True, False])
def test_nm4_zigzag_output_and_loss(
    distributed_test_runner, prepacked: bool, padding: bool
) -> None:
    distributed_test_runner(
        functools.partial(_check_layout_loss, prepacked=prepacked, padding=padding),
        world_size=4,
    )


def _check_sequence_parallel_output(rank: int, world_size: int) -> None:
    from megatron.bridge.models.experimental_nm4_llava_provider import NM4LlavaModel
    from megatron.core.packed_seq_params import PackedSeqParams

    assert world_size == 4
    tp_group = cp_group = None
    for peers in ([0, 1], [2, 3], [0, 2], [1, 3]):
        group = dist.new_group(peers)
        if rank in peers:
            if peers[1] - peers[0] == 1:
                tp_group = group
            else:
                cp_group = group
    cp_rank = dist.get_rank(cp_group)
    bounds = torch.tensor([0, 128, 320, 512], dtype=torch.int32, device="cuda")
    tokens = torch.arange(512, device="cuda").view(1, -1)
    mask = (tokens % 3 != 0).float()
    logits = (
        tokens.float().unsqueeze(-1).chunk(2, dim=1)[cp_rank].clone().requires_grad_()
    )
    model = object.__new__(NM4LlavaModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model.post_process = True
    model.context_parallel_lm = 2
    model.sequence_parallel_lm = True
    model.config = SimpleNamespace(
        linear_cp_layout="contiguous", attention_cp_layout="zigzag", mtp_num_layers=2
    )
    model.pg_collection = SimpleNamespace(
        cp=cp_group, tp=tp_group, tp_cp=dist.group.WORLD
    )
    model.language_model = MagicMock(return_value=logits)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=bounds,
        cu_seqlens_kv=bounds,
        cu_seqlens_q_padded=bounds,
        cu_seqlens_kv_padded=bounds,
        max_seqlen_q=192,
        max_seqlen_kv=192,
    )
    output, output_mask = model(
        input_ids=tokens, position_ids=tokens, loss_mask=mask, packed_seq_params=params
    )
    expected_tokens = torch.cat(
        [
            torch.cat(
                (
                    tokens[:, start:end].chunk(4, dim=1)[cp_rank],
                    tokens[:, start:end].chunk(4, dim=1)[3 - cp_rank],
                ),
                1,
            )
            for start, end in zip(bounds[:-1].tolist(), bounds[1:].tolist())
        ],
        1,
    )
    torch.testing.assert_close(output, expected_tokens.float().unsqueeze(-1))
    torch.testing.assert_close(output_mask, (expected_tokens % 3 != 0).float())
    input_batch = model.language_model.call_args.kwargs["cp_batch"]
    assert input_batch.boundary_layout == "contiguous"
    assert input_batch.thd_plan.contiguous_local_token_count == 128
    assert output.shape[1] == 256
    output.sum().backward()
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))


@pytest.mark.mcore
def test_nm4_sequence_parallel_output(distributed_test_runner) -> None:
    distributed_test_runner(_check_sequence_parallel_output, world_size=4)
