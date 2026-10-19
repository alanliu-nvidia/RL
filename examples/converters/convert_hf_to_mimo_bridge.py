# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from nemo_rl.models.megatron.mimo import (
    resolve_mimo_sft_config,
    validate_mimo_parallelism,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

"""Convert a Hugging Face checkpoint into a MegatronMIMO bridge checkpoint.

A MIMO run cannot start from Hugging Face weights. ``setup_megatron_mimo``
only ever restores a dist-checkpoint, and the HF->Megatron conversion cache
that ordinary Megatron policies rely on is written before the MIMO provider
replaces the standard one, so it holds the flat ``vision_model.*`` layout
rather than ``modality_submodules.images.*``. Every MIMO recipe in this repo
therefore points ``checkpointing.pretrained_checkpoint`` at a checkpoint
produced here.

The MIMO parallelism is read from the recipe that will consume the checkpoint,
so the conversion lays weights out exactly as training will ask for them. Run
with as many ranks as that recipe's module_parallelisms cover.

NOTE: this script requires mcore. Launch it with the mcore extra:
uv run --extra mcore torchrun --nproc_per_node=4 --nnodes=4 \
  examples/converters/convert_hf_to_mimo_bridge.py \
  --config examples/configs/recipes/vlm/<recipe>.yaml \
  --output-path /path/to/mimo_bridge_ckpt
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a HF checkpoint to a MegatronMIMO bridge checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Recipe whose policy.megatron_cfg.mimo block defines the layout",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to write the MegatronMIMO checkpoint into",
    )
    parser.add_argument(
        "--hf-model-path",
        type=str,
        default=None,
        help="HF model to convert; defaults to the recipe's policy.model_name",
    )
    return parser.parse_args()


def main():
    """Build the MIMO model from HF weights and save it as a bridge checkpoint."""
    args = parse_args()

    # Device first: NCCL binds every local rank to device 0 otherwise.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

    # Plain containers: pydantic validates dicts, not DictConfig nodes, and
    # resolving here expands the ${oc.env:...} model paths recipes use. The
    # training entry points register the arithmetic resolvers recipes also use.
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(args.config), resolve=True)
    policy_cfg = config["policy"]
    mimo_config = resolve_mimo_sft_config(policy_cfg)
    if mimo_config is None:
        raise ValueError(
            f"{args.config} has no policy.megatron_cfg.mimo block, so there is "
            "no MIMO layout to convert into."
        )
    parallelism = validate_mimo_parallelism(
        mimo_config,
        world_size=dist.get_world_size(),
        megatron_config=policy_cfg["megatron_cfg"],
    )

    hf_model_path = args.hf_model_path or policy_cfg["model_name"]

    # Imported here, not at module scope: megatron.training reaches for
    # mistral_adapter, which only becomes importable once nemo_rl has appended
    # the Megatron-LM checkout to sys.path, and import sorting would otherwise
    # put megatron ahead of nemo_rl.
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.megatron_mimo.conversion import MegatronMIMOBridge

    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)

    mimo_bridge = MegatronMIMOBridge.from_bridge(
        bridge, parallelism_config=parallelism
    )
    mimo_bridge.import_ckpt(args.output_path, hf_tokenizer_path=hf_model_path)

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Wrote MegatronMIMO checkpoint to {args.output_path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
