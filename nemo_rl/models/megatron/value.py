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

"""Scalar value collection for the shared MIMO forward/backward schedule."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor
from nemo_rl.utils.sequence_lengths import CpuIntTuple, to_cpu_int_tuple

if TYPE_CHECKING:
    from nemo_rl.algorithms.loss.interfaces import LossFunction
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def unpack_mimo_values(
    output: torch.Tensor,
    boundaries: torch.Tensor | CpuIntTuple | None,
    sequence_length: int,
    *,
    cp_group: Any,
    cp_layout: str,
) -> torch.Tensor:
    """Gather scalar predictions and shift within each original sequence.

    Contiguous CP partitions the complete pack, whereas zigzag CP partitions
    each sequence. Gather before slicing for the former so no sequence is
    silently read from a neighbouring rank's position.
    """
    if output.ndim != 3 or output.shape[-1] != 1:
        raise ValueError(f"Expected scalar value head [B, S, 1], got {output.shape}.")
    values = output.squeeze(-1)
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    if boundaries is None:
        if cp_size > 1:
            raise ValueError("MIMO values with CP require sequence packing.")
        return torch.cat((torch.zeros_like(values[:, :1]), values[:, :-1]), dim=1)[
            :, :sequence_length
        ]
    starts = to_cpu_int_tuple(boundaries)
    if cp_layout not in ("contiguous", "zigzag"):
        raise ValueError(f"Unknown CP layout: {cp_layout}.")
    if cp_size > 1 and cp_layout == "contiguous":
        values = allgather_cp_sharded_tensor(
            values, cp_group, seq_dim=1, cp_layout=cp_layout
        )
    rows = []
    for start, end in zip(starts[:-1], starts[1:]):
        if cp_size > 1 and cp_layout == "zigzag":
            row = allgather_cp_sharded_tensor(
                values[:, start // cp_size : end // cp_size],
                cp_group,
                seq_dim=1,
                cp_layout=cp_layout,
            )
        else:
            row = values[:, start:end]
        row = torch.cat((torch.zeros_like(row[:, :1]), row[:, :-1]), dim=1)
        row = row[:, :sequence_length]
        rows.append(torch.nn.functional.pad(row, (0, sequence_length - row.shape[1])))
    return torch.cat(rows, dim=0)


class ValuePostProcessor:
    """Collect image-conditioned values, or apply the PPO value loss to them."""

    def __init__(
        self, loss_fn: LossFunction | None = None, num_microbatches: int = 1
    ) -> None:
        self.loss_fn = loss_fn
        self.num_microbatches = num_microbatches

    def __call__(
        self,
        *,
        data_dict: BatchedDataDict[Any],
        cu_seqlens_padded: Any,
        original_seq_length: int,
        cp_layout: str,
        cp_group: Any,
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor | None,
    ) -> Callable:
        """Build a collector without applying LM temperature or vocabulary logic."""
        cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)

        def collect(output: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            values = unpack_mimo_values(
                output,
                cu_seqlens_padded,
                original_seq_length,
                cp_group=cp_group,
                cp_layout=cp_layout,
            )
            if self.loss_fn is None:
                return values.new_zeros(()), {"values": values}
            loss, metrics = self.loss_fn(
                logits=values,
                data=data_dict,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
            )
            # Match LossPostProcessor: cancel schedule microbatch averaging and
            # its CP multiplier, then compensate replicated full-sequence loss.
            return loss * self.num_microbatches / (cp_size * cp_size), metrics

        return collect
