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

"""Filter prepared NeMo-RL Energon batches for each MIMO loader role."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from nemo_rl.data.multimodal_utils import PackedTensor

MIMORole = Literal["language", "encoder"]


def mimo_data_plane_batch(
    prepared: Mapping[str, Any], *, role: MIMORole
) -> dict[str, Any]:
    """Return the prepared fields published by one role's existing V2 loader.

    The encoder view retains pixels locally. The language view of the same
    canonical batch contains token and packing fields, without media payloads.
    """
    if role == "encoder":
        return dict(prepared)
    if role == "language":
        return {
            key: value
            for key, value in prepared.items()
            if not isinstance(value, PackedTensor)
            or key in {"cu_seqlens", "cu_seqlens_padded"}
        }
    raise ValueError(f"Unsupported MIMO role {role!r}.")


__all__ = ["MIMORole", "mimo_data_plane_batch"]
