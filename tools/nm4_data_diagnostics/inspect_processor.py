#!/usr/bin/env python3
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

"""Inspect the real processor's vocabulary on CPU; never load model weights."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/nm4-diagnostics-hf-modules")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_processor(config_path, processor_path=None):
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    from nemo_rl.utils.config import load_config

    config = load_config(config_path)
    settings = OmegaConf.to_container(config.policy.tokenizer, resolve=True)
    processor = AutoProcessor.from_pretrained(
        processor_path or settings["name"],
        trust_remote_code=True,
        **{"use_fast": True, **(settings.get("tokenizer_kwargs") or {})},
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = settings.get("chat_template", "default")
    if template != "default":
        # Match non-default training templates without importing model setup.
        if template is None:
            raise ValueError("Passthrough template is unsupported by this diagnostic.")
        tokenizer.chat_template = (
            Path(template).read_text() if template.endswith(".jinja") else template
        )
    if settings.get("chat_template_kwargs"):
        from functools import partial

        tokenizer.apply_chat_template = partial(
            tokenizer.apply_chat_template, **settings["chat_template_kwargs"]
        )
    return config, processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--processor", help="Override /data processor path when running on the host"
    )
    args = parser.parse_args()
    _, processor = load_processor(args.config, args.processor)
    import torch

    from nemo_rl.data.energon.multimodal.task_encoders.nemotron_tokenization import (
        _NEMOTRON6_TOKENIZER_LAYOUT,
    )

    tokenizer = processor.tokenizer
    layout = []
    for ids, expected in _NEMOTRON6_TOKENIZER_LAYOUT.items():
        actual = tokenizer.decode(list(ids), clean_up_tokenization_spaces=False)
        layout.append(
            {
                "expected": expected,
                "hardcoded_ids": list(ids),
                "actual_decoding": actual,
                "matches": actual == expected,
                "actual_encoding": tokenizer.encode(expected, add_special_tokens=False),
            }
        )
    report = {
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "processor_path": tokenizer.name_or_path,
        "vocab_size": len(tokenizer),
        "cuda_available": torch.cuda.is_available(),
        "nemotron6_layout": layout,
        "media_tokens": {
            s: tokenizer.convert_tokens_to_ids(s)
            for s in [
                "<image>",
                "<so_embedding>",
                "<sound>",
                "<sound_start>",
                "<sound_end>",
            ]
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
