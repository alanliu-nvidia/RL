# NM4 CPU data diagnostics

These scripts load the real processor and the existing Energon cookers/task
encoder without loading model weights, Ray actors, Megatron-Core, or CUDA.
The NM4 copies live in new `nemo_rl/data/energon/multimodal/task_encoders/nemotron4_*.py`
modules and `cookers/nemotron4.py`. Existing Nemotron implementations are
unchanged. The requested recipe selects separate NM4 keys; the minimal registry
and option-dispatch wiring is included in the training loader.

## Local environment

Tested on the current aarch64 CPU node with Python 3.13:

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache uv --no-config venv --python 3.13 /tmp/nm4-cpu-diagnostics-venv
UV_CACHE_DIR=/tmp/codex-uv-cache uv --no-config pip install \
  --python /tmp/nm4-cpu-diagnostics-venv/bin/python \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' 'setuptools>=80.10.2,<82'
UV_CACHE_DIR=/tmp/codex-uv-cache uv --no-config pip install \
  --python /tmp/nm4-cpu-diagnostics-venv/bin/python \
  'transformers==5.12.1' 'megatron-energon[av-decode]==7.4.0' \
  omegaconf hydra-core datasets scipy soundfile librosa einops 'pydantic>=2' \
  'fsspec[http]==2026.6.0' 's3fs==2026.6.0'
```

Every script hides GPUs. Reports also record whether CUDA is available.
Run from the repository root. The processor path below is the host directory
mounted as `/data/nemotron4/processor` during training.

```bash
/tmp/nm4-cpu-diagnostics-venv/bin/python tools/nm4_data_diagnostics/inspect_processor.py \
  --config examples/configs/sft_v2_tests/nm4_generalist_16n_mxfp8_cp64ep64_gbs1.yaml \
  --processor /home/rohitkumarj/data/nemotron4/processor \
  --output logbook/nm4_cpu_diagnostics/processor.json

/tmp/nm4-cpu-diagnostics-venv/bin/python tools/nm4_data_diagnostics/audit_samples.py \
  --config examples/configs/sft_v2_tests/nm4_generalist_16n_mxfp8_cp64ep64_gbs1.yaml \
  --processor /home/rohitkumarj/data/nemotron4/processor \
  --dataset examples/configs/sft_v2_tests/sft_super35_generalist_full_cmh.yaml \
  --data-root /home/rohitkumarj/data --samples 100 --postencode \
  --output logbook/nm4_cpu_diagnostics/original.jsonl

/tmp/nm4-cpu-diagnostics-venv/bin/python tools/nm4_data_diagnostics/audit_samples.py \
  --config examples/configs/sft_v2_tests/nm4_generalist_16n_mxfp8_cp64ep64_gbs1.yaml \
  --processor /home/rohitkumarj/data/nemotron4/processor \
  --dataset examples/configs/sft_v2_tests/sft_super35_generalist_full_cmh.yaml \
  --data-root /home/rohitkumarj/data --samples 100 --postencode --packed-batch \
  --implementation nm4_copies --prompt-formats nemotron6-moe --diagnostics 1 \
  --save-rejected-payloads --output logbook/nm4_cpu_diagnostics/nm4.jsonl

/tmp/nm4-cpu-diagnostics-venv/bin/python tools/nm4_data_diagnostics/verify_nm4_masks.py \
  --config examples/configs/sft_v2_tests/nm4_generalist_16n_mxfp8_cp64ep64_gbs1.yaml \
  --processor /home/rohitkumarj/data/nemotron4/processor \
  --output logbook/nm4_cpu_diagnostics/masks.json
```

`--data-root` translates `/data/` paths in a temporary copy of the complete
blend. All 974 leaves, weights, and subflavors remain present. No CLEVR-only
dataset is substituted. Sampling uses the train blend, one logical rank,
zero loader workers, and seed offset 42. It is reproducible but does not
reproduce the exact two-worker ordering of the distributed training job.

The original audit compares `nemotron6-moe` and `processor_chat_template`,
with diagnostics off/on. It keeps the first-visual diagnostic state separately
for each combination. JSONL contains sample/source identity, modality, stage,
exceptions, and tracebacks. Companion `.summary.json` and `.metadata.json`
files contain counts and dependency versions. `--save-rejected-payloads`
preserves complete raw JSON for cooker failures to inspect separately.

`--packed-batch` checks singleton pack selection, physical packing, and batch
creation at the recipe's 524,288-token capacity. It does not recreate the
5,000-sample packing buffer or test model execution/GPU memory.
`--synthetic` supplies controlled text and image cases without dataset access.

## Findings and isolated copies

The real processor is `PixtralProcessor`, with a 262,144-token vocabulary.
Native boundary IDs 10, 11, 16, and 1010 match. Role IDs differ:

| Role | Existing N6 assumptions | NM4 actual encoding |
| --- | --- | --- |
| assistant | 1503, 19464 | 6291 |
| user | 3263 | 2837 |

Native chat samples fail in `_validate_nemotron6_tokenizer`, during preencoding,
before media decoding. Raw-text leaves have `skip_chat_template: true` and
bypass this check. That branch rejects media placeholders by design.

The full blend has 867 chat-template leaves and 107 raw-text leaves. In the
100-sample audit, 93 cooked samples had images, five had videos, and two were
raw text. The original native path rejected all 98 chat samples. Changing
only the prompt to `processor_chat_template` admitted the samples but exposed
an additional first-visual diagnostic error: `NemotronVisualPlan` provides
`num_embeddings`, while the diagnostic reads `total_embeddings`. The flag is
set before the error, so this drops the first visual sample per worker rather
than every subsequent visual sample.

The NM4 copies change only the processor-specific role IDs, the corresponding
assistant header/mask offsets, the visual diagnostic field, local imports,
and class names. The cooker copy preserves the original cooker behavior.
Audio plans still use their existing `total_embeddings` property. The mask
verification covers all turns, explicit turn masks, thinking-trace filtering,
and native tool-response boundaries. It preserves the existing rule that the
last-turn flag alone does not filter when no thinking trace or native tool
boundary is present.

Verified results from the complete blend (bounded sample, not a dataset census):

| Implementation/check | Observed result |
| --- | --- |
| Original native format | 98 chat samples rejected; two raw-text samples passed |
| Original processor template, diagnostics on | 99/100 cooked samples decoded; first visual diagnostic failed |
| NM4 copies, native format, diagnostics on | 100/100 cooked samples decoded |
| NM4 copies through packed CPU batch creation | 100/100 batches created with shape `[1, 524288]` |
| NM4 answer masks | Six checks passed |

Full JSONL evidence is under `logbook/nm4_cpu_diagnostics/`: the original
comparison is `full_generalist.jsonl`, the final copied implementation is
`nm4_final.jsonl`, and mask checks are `nm4_masks.json`. Both blend audits
encountered the two additional cooker rejects below.

Two source/cooker rejects appeared before 100 cooked samples were emitted:

- `benchfit_qa@v1/03-10a_stage5_output.jsonl`, source index 435377:
  `conversation` is an empty list.
- `output_molmo2_video_pointqa_custom_safetyfix0323@v1/`
  `output_molmo2_video_pointqa_custom_safetyfix0323.jsonl`, index 10944:
  literal `<image>` text inside the assistant reasoning is interpreted as a
  media tag, but the record has only a `video` attachment.

These are reported rather than silently converted or removed. A bounded
sample audit is not a census of every record in the 974-leaf blend.

## Training integration

The requested recipe selects the NM4 encoder and all four NM4 cookers through
the production registry. The loader forwards the existing Nemotron options to
both encoder implementations. A 10-sample full-blend CPU audit using the actual
YAML/registry path passed through packed batch creation and recorded both
malformed source records. `--implementation nm4_copies` remains available for
explicit diagnostic selection.

Set `NRL_NM4_SAMPLE_DROP_DIR` to collect per-process JSONL records of cooker,
preencode, and postencode failures. The recipe sets it to
`/opt/nemo-rl/logbook/nm4_sample_drops`. Counts are processing attempts; a source
record can repeat across epochs. `summarize_drops.py DIRECTORY` reports counts
by stage, cooker, and error type without relying on Ray log deduplication.

`copy_manifest.json` records hashes of the originals and copies.
`nm4_changes.patch` shows the precise differences for review; it is not an
instruction to overwrite the legacy files.
