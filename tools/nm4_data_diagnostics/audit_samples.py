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

"""Run cook/preencode/postencode on CPU and record every sampled rejection."""

import argparse
import importlib.metadata
import json
import os
import random
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/nm4-diagnostics-hf-modules")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inspect_processor import load_processor


def synthetic_samples():
    from PIL import Image

    from nemo_rl.data.energon.multimodal.types import (
        CanonicalSFTSample,
        MediaRef,
        freeze_media_metadata,
    )

    for name, media, skip in [
        ("text_chat", [], False),
        ("text_raw", [], True),
        (
            "image_chat",
            [
                MediaRef(
                    "image",
                    Image.new("RGB", (64, 64)),
                    freeze_media_metadata({"width": 64, "height": 64}),
                )
            ],
            False,
        ),
        (
            "image_raw",
            [
                MediaRef(
                    "image",
                    Image.new("RGB", (64, 64)),
                    freeze_media_metadata({"width": 64, "height": 64}),
                )
            ],
            True,
        ),
    ]:
        content = [{"type": "image", "media_index": 0}] if media else []
        content.append({"type": "text", "text": "Describe this sample."})
        yield CanonicalSFTSample(
            __key__=name,
            __restore_key__=(name,),
            __subflavors__={"skip_chat_template": skip},
            messages=[
                {"role": "user", "content": content},
                {"role": "assistant", "content": "A sample."},
            ],
            media=media,
            tools=None,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="JSONL observations; companion .summary.json is also written",
    )
    parser.add_argument(
        "--processor", help="Override /data processor path when running on the host"
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--dataset",
        help="Override blend/source path, e.g. a small local MetadatasetV2 YAML",
    )
    parser.add_argument(
        "--data-root",
        help="Host directory mounted as /data in training; only paths are translated",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Maximum cooked samples; upstream errors also have a separate bound",
    )
    parser.add_argument("--max-source-errors", type=int, default=100)
    parser.add_argument(
        "--prompt-formats", default="nemotron6-moe,processor_chat_template"
    )
    parser.add_argument("--diagnostics", choices=["0", "1", "both"], default="both")
    parser.add_argument(
        "--postencode", action="store_true", help="Also decode/process media on CPU"
    )
    parser.add_argument(
        "--packed-batch",
        action="store_true",
        help="Also select a singleton pack and build its full-capacity CPU batch",
    )
    parser.add_argument(
        "--implementation", choices=["original", "nm4_copies"], default="original"
    )
    parser.add_argument(
        "--save-rejected-payloads",
        action="store_true",
        help="Include complete raw JSON for source/cooker failures in the report",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.max_source_errors < 1:
        parser.error("Sample/error bounds must be positive.")
    if args.packed_batch and not args.postencode:
        parser.error("--packed-batch requires --postencode.")
    import torch
    from megatron.energon import (
        DefaultTaskEncoder,
        FileStoreCachePool,
        WorkerConfig,
        get_savable_loader,
        get_train_dataset,
        stateless,
    )
    from omegaconf import OmegaConf

    from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
        build_processor_adapter,
    )
    from nemo_rl.data.energon.sft_dataloader import (
        _loader_config,
        _set_nvdataset_cache_dir,
        _task_encoder,
    )

    config, processor = load_processor(args.config, args.processor)
    loader_config = _loader_config(
        OmegaConf.to_container(config.data.energon, resolve=True)
    )
    capacity = int(config.policy.max_total_sequence_length)
    adapter = build_processor_adapter(
        processor_adapter=loader_config.processor_adapter,
        processor=processor,
        max_sequence_length=capacity,
        add_bos=config.data.get("add_bos", True),
        add_eos=config.data.get("add_eos", True),
        add_generation_prompt=config.data.get("add_generation_prompt", False),
    )
    encoder = _task_encoder(
        loader_config=loader_config,
        adapter=adapter,
        include_source_ids=True,
        max_sequence_length=capacity,
        tokenizer=processor.tokenizer,
        only_unmask_final=config.sft.only_unmask_final,
    )
    if args.implementation == "nm4_copies":
        from megatron.energon import Cooker

        from nemo_rl.data.energon.multimodal.cookers import (
            nemotron4 as nemotron4_cookers,
        )
        from nemo_rl.data.energon.multimodal.task_encoders.nemotron4_multimodal import (
            Nemotron4MultiModalTaskEncoder,
        )

        copied_cookers = [
            Cooker(
                getattr(nemotron4_cookers, cooker.cook.__name__),
                has_subflavors=cooker.has_subflavors,
            )
            for cooker in encoder.cookers
        ]
        encoder = Nemotron4MultiModalTaskEncoder(
            adapter=adapter,
            cooker_functions=copied_cookers,
            include_source_ids=True,
            packer=encoder.packer,
            tokenizer=processor.tokenizer,
            sequence_length_pad_multiple=encoder.sequence_length_pad_multiple,
            only_unmask_final=config.sft.only_unmask_final,
            **loader_config.task_encoder.options.model_dump(),
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    source_errors = 0
    diagnostic_states = {}
    output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "config": str(Path(args.config).resolve()),
                "implementation": args.implementation,
                "dataset": args.dataset or str(config.data.train.path),
                "data_root": args.data_root,
                "processor": processor.tokenizer.name_or_path,
                "cpu_only": not torch.cuda.is_available(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ["torch", "transformers", "megatron-energon"]
                },
                "sampling": "train blend, seed 42, one logical rank, no packing, no loader workers",
                "diagnostic_semantics": "independent persistent first-visual flag for each prompt/diagnostic combination",
            },
            indent=2,
        )
        + "\n"
    )

    class SourceErrorLimit(BaseException):
        pass

    with output.open("w") as stream:

        def record(row):
            counts[
                (
                    row.get("prompt_format", "source"),
                    row.get("diagnostics", ""),
                    row["stage"],
                    row["status"],
                    row.get("error_type", ""),
                    row.get("error", ""),
                )
            ] += 1
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()

        def source_error(exception, sample, sources=None):
            nonlocal source_errors
            source_errors += 1
            raw = sample if isinstance(sample, dict) else {}
            payload = raw.get("json", {})
            if isinstance(payload, (str, bytes)):
                try:
                    payload = json.loads(payload)
                except (ValueError, UnicodeError):
                    payload = {}
            record(
                {
                    "stage": "source_or_cook",
                    "status": "rejected",
                    "sample_key": raw.get("__key__", getattr(sample, "__key__", None)),
                    "subflavors": raw.get("__subflavors__", {}),
                    "payload_keys": list(payload) if isinstance(payload, dict) else [],
                    "payload_structure": {
                        key: type(value).__name__ for key, value in payload.items()
                    }
                    if isinstance(payload, dict)
                    else {},
                    **({"payload": payload} if args.save_rejected_payloads else {}),
                    "error_type": type(exception).__name__,
                    "error": str(exception),
                    "traceback": traceback.format_exc(),
                    "sources": [str(s) for s in sources or []],
                }
            )
            if source_errors >= args.max_source_errors:
                raise SourceErrorLimit()

        if args.synthetic:
            samples = synthetic_samples()
        else:
            if loader_config.nvdataset_cache_dir:
                _set_nvdataset_cache_dir(loader_config.nvdataset_cache_dir)

            class CookOnlyEncoder(DefaultTaskEncoder):
                def __init__(self):
                    super().__init__()
                    self.cookers = encoder.cookers
                    self.decoder = encoder.decoder

                @stateless
                def cook_crude_sample(self, sample, cooker, aux):
                    return encoder.cook_crude_sample(sample, cooker, aux)

                def encode_sample(self, sample):
                    return sample

                def batch(self, samples):
                    return samples

                def encode_batch(self, batch):
                    return batch

            source = args.dataset or str(config.data.train.path)
            if source.startswith("/opt/nemo-rl/"):
                source = str(
                    Path(__file__).resolve().parents[2]
                    / source.removeprefix("/opt/nemo-rl/")
                )
            translated_source = None
            if args.data_root and source.endswith((".yaml", ".yml")):
                import yaml

                def translate(value):
                    if isinstance(value, dict):
                        return {key: translate(item) for key, item in value.items()}
                    if isinstance(value, list):
                        return [translate(item) for item in value]
                    if isinstance(value, str) and value.startswith("/data/"):
                        return str(Path(args.data_root) / value.removeprefix("/data/"))
                    return value

                original = yaml.safe_load(Path(source).read_text())
                # This recipe uses absolute paths and DSS URIs; a temporary
                # host-path copy leaves the original dataset YAML untouched.
                translated_source = tempfile.TemporaryDirectory(prefix="nm4-source-")
                source_copy = Path(translated_source.name) / Path(source).name
                source_copy.write_text(
                    yaml.safe_dump(translate(original), sort_keys=False)
                )
                source = str(source_copy)
            dataset = get_train_dataset(
                source,
                split_part=str(config.data.train.split),
                worker_config=WorkerConfig(
                    rank=0,
                    world_size=1,
                    num_workers=0,
                    seed_offset=42,
                    global_error_handler=source_error,
                    restore_error_handler=source_error,
                ),
                batch_size=1,
                batch_drop_last=False,
                packing_buffer_size=None,
                shuffle_buffer_size=loader_config.shuffle_buffer_size,
                shuffle_over_epochs_multiplier=1,
                max_samples_per_sequence=loader_config.max_samples_per_sequence,
                virtual_epoch_length=args.samples,
                task_encoder=CookOnlyEncoder(),
            )
            loader = get_savable_loader(
                dataset, cache_pool=FileStoreCachePool(method="raw", num_workers=1)
            )
            samples = (sample for batch in loader for sample in batch)

        sampled = 0
        try:
            for sample in samples:
                sampled += 1
                base = {
                    "sample_key": sample.__key__,
                    "subflavors": sample.__subflavors__,
                    "modalities": [ref.modality for ref in sample.media],
                    "media_payload_types": [
                        type(ref.value).__name__ for ref in sample.media
                    ],
                    "media_stores": [
                        str(getattr(ref.value, "ds", "in-memory"))[:1000]
                        for ref in sample.media
                    ],
                    "media_metadata": [dict(ref.metadata) for ref in sample.media],
                    "roles": [m["role"] for m in sample.messages],
                    "sources": [str(s) for s in sample.__sources__ or []],
                }
                for prompt in args.prompt_formats.split(","):
                    for diagnostic in (
                        ["0", "1"] if args.diagnostics == "both" else [args.diagnostics]
                    ):
                        random.seed(42)
                        torch.manual_seed(42)
                        os.environ["NRL_SFT_MULTIMODAL_DIAGNOSTICS"] = diagnostic
                        encoder.adapter.prompt_format = prompt
                        encoder.adapter._logged_multimodal_diagnostic = (
                            diagnostic_states.get((prompt, diagnostic), False)
                        )
                        row = {
                            **base,
                            "prompt_format": prompt,
                            "diagnostics": diagnostic,
                            "diagnostic_already_logged": encoder.adapter._logged_multimodal_diagnostic,
                        }
                        stage = "preencode"
                        # The real @stateless methods require an Energon worker
                        # context even for a single direct CPU call.
                        diagnostic_worker = WorkerConfig(
                            rank=0, world_size=1, num_workers=0, seed_offset=42
                        )
                        diagnostic_worker.worker_activate(sampled)
                        try:
                            encoded = encoder.preencode_sample(sample)
                            row["preencoded_length"] = encoded.length
                            if args.postencode:
                                stage = "postencode"
                                encoded = encoder.postencode_sample(encoded)
                            if args.packed_batch:
                                stage = "pack_selection"
                                packs = encoder.select_samples_to_pack([encoded])
                                assert len(packs) == 1 and len(packs[0]) == 1
                                stage = "packing"
                                packed = encoder.pack_selected_samples(packs[0])
                                stage = "batch"
                                batch = encoder.batch([packed])
                                stage = "encode_batch"
                                batch = encoder.encode_batch(batch)
                                row["batch_keys"] = sorted(batch.keys())
                                row["batch_input_shape"] = list(
                                    batch["input_ids"].shape
                                )
                            row.update(
                                stage=stage,
                                status="accepted",
                                length=encoded.length,
                                input_keys=sorted(
                                    {
                                        key
                                        for message in encoded.message_log
                                        for key in message
                                    }
                                ),
                                trainable_tokens=sum(
                                    int(m["token_loss_mask"].sum())
                                    for m in encoded.message_log
                                ),
                            )
                        except Exception as error:
                            frames = traceback.extract_tb(error.__traceback__)
                            row.update(
                                stage=stage,
                                status="rejected",
                                error_type=type(error).__name__,
                                error=str(error),
                                failure_function=frames[-1].name,
                                failure_file=frames[-1].filename,
                                failure_line=frames[-1].lineno,
                                traceback=traceback.format_exc(),
                            )
                        finally:
                            diagnostic_states[(prompt, diagnostic)] = (
                                encoder.adapter._logged_multimodal_diagnostic
                            )
                            diagnostic_worker.worker_deactivate()
                        record(row)
                if sampled >= args.samples:
                    break
        except SourceErrorLimit:
            print(
                "Stopped at the source/cooker error bound; see JSONL.", file=sys.stderr
            )

    summary = {
        "cpu_only": not torch.cuda.is_available(),
        "sampled": sampled,
        "source_errors": source_errors,
        "postencode_checked": args.postencode,
        "packed_batch_checked": args.packed_batch,
        "counts": [
            {
                "prompt_format": key[0],
                "diagnostics": key[1],
                "stage": key[2],
                "status": key[3],
                "error_type": key[4],
                "error": key[5],
                "count": value,
            }
            for key, value in counts.items()
        ],
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
