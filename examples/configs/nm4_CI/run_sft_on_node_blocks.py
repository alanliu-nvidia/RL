# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Test-only SFT entry point selecting explicit MR10 node blocks."""

import json
import os
import runpy
from functools import partial
from pathlib import Path


def main() -> None:
    """Keep the control on the LLM nodes while using the normal training entry point."""
    blocks = json.loads(os.environ["NM4_CI_NODE_BLOCKS"])
    if (
        not isinstance(blocks, list)
        or not blocks
        or any(type(block) is not int or block < 0 for block in blocks)
    ):
        raise ValueError(
            "NM4_CI_NODE_BLOCKS must be a nonempty list of nonnegative integers"
        )
    # Training imports are only needed for an actual GPU run, not CPU planning.
    from nemo_rl.algorithms import sft_v2

    original = sft_v2.node_block_constraints
    sft_v2.node_block_constraints = partial(original, blocks=blocks)
    try:
        runpy.run_path(
            str(Path(__file__).resolve().parents[3] / "examples/run_sft_v2.py"),
            run_name="__main__",
        )
    finally:
        sft_v2.node_block_constraints = original


if __name__ == "__main__":
    main()
