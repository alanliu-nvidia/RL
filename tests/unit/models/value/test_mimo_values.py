# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU checks for packed MIMO values, including real two-rank CP gradients."""

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.models.megatron.value import ValuePostProcessor, unpack_mimo_values


def test_values_shift_within_each_packed_sample_and_keep_gradients() -> None:
    output = torch.arange(1.0, 9.0).reshape(1, 8, 1).requires_grad_()
    values = unpack_mimo_values(
        output, (0, 4, 8), 3, cp_group=None, cp_layout="contiguous"
    )
    torch.testing.assert_close(values, torch.tensor([[0.0, 1.0, 2.0], [0.0, 5.0, 6.0]]))
    values.sum().backward()
    torch.testing.assert_close(
        output.grad.flatten(), torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    )


def test_value_collector_passes_masks_old_values_and_returns_to_loss() -> None:
    data = {
        "returns": torch.zeros(2, 3),
        "values": torch.ones(2, 3),
        "sample_mask": torch.tensor([1.0, 0.0]),
        "token_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
    }
    observed = []

    def loss_fn(*, logits: torch.Tensor, data: Any, **kwargs: Any) -> tuple:
        observed.append(data)
        mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
        loss = ((logits - data["returns"]) ** 2 * mask).sum() / kwargs[
            "global_valid_toks"
        ]
        return loss, {"loss": loss.detach()}

    collect = ValuePostProcessor(loss_fn=loss_fn, num_microbatches=2)(
        data_dict=data,
        cu_seqlens_padded=(0, 4, 8),
        original_seq_length=3,
        cp_group=None,
        cp_layout="contiguous",
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(2.0),
    )
    output = torch.arange(1.0, 9.0).reshape(1, 8, 1).requires_grad_()
    loss, metrics = collect(output)
    assert observed[0] is data
    torch.testing.assert_close(loss, torch.tensor(5.0))
    torch.testing.assert_close(metrics["loss"], torch.tensor(2.5))
    loss.backward()
    torch.testing.assert_close(
        output.grad.flatten(), torch.tensor([2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


def _cp_worker(rank: int, rendezvous: str, layout: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        full = torch.arange(1.0, 9.0)
        indices = torch.tensor([0, 1, 2, 3] if rank == 0 else [4, 5, 6, 7])
        if layout == "zigzag":
            indices = torch.tensor([0, 3, 4, 7] if rank == 0 else [1, 2, 5, 6])
        output = full[indices].reshape(1, 4, 1).requires_grad_()
        values = unpack_mimo_values(
            output, (0, 4, 8), 4, cp_group=dist.group.WORLD, cp_layout=layout
        )
        expected = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 5.0, 6.0, 7.0]])
        torch.testing.assert_close(values, expected)

        # Each CP replica sees the same full loss. Dividing by CP compensates
        # the differentiable gather's backward sum over those replicas.
        def loss_fn(*, logits: torch.Tensor, **kwargs: Any) -> tuple:
            torch.testing.assert_close(logits, expected)
            return logits.square().sum(), {}

        collector = ValuePostProcessor(loss_fn=loss_fn, num_microbatches=3)(
            data_dict={},
            cu_seqlens_padded=(0, 4, 8),
            original_seq_length=4,
            cp_group=dist.group.WORLD,
            cp_layout=layout,
            global_valid_seqs=None,
            global_valid_toks=None,
        )
        loss, _ = collector(output)
        # Simulate Megatron schedule's CP / num_microbatches multiplier.
        (loss * 2 / 3).backward()
        full_grad = torch.tensor([2.0, 4.0, 6.0, 0.0, 10.0, 12.0, 14.0, 0.0])
        torch.testing.assert_close(output.grad.flatten(), full_grad[indices])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
def test_two_rank_cp_values_and_gradients(tmp_path: Path, layout: str) -> None:
    mp.spawn(
        _cp_worker, args=(f"file://{tmp_path / layout}", layout), nprocs=2, join=True
    )


def test_rejects_language_vocabulary_logits() -> None:
    with pytest.raises(ValueError, match="scalar value head"):
        unpack_mimo_values(
            torch.ones(1, 4, 16), (0, 4), 4, cp_group=None, cp_layout="contiguous"
        )
