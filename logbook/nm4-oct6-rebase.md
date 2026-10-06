# NM4 production branch migration — 2026-10-06

## Scope and base

The isolated AGA rem project is `sft_v2_n4_oct6_rebase`. Its worktree is
`/home/rohitkumarj/code/RL/nemo-rl-sft_v2_n4_oct6_rebase`, on
`rohit/sft_v2_n4_oct6_rebase`. The running production project and branch were
left unchanged.

| Reference | Commit |
| --- | --- |
| Production source, `rohit/sft_v2_up_n4` | `3222f30293276c414175a256da435f6a04f1fc9d` |
| PR #4515 before the requested packing update | `8785c3fab834914adf742278efdeaed69569c1e4` |
| Updated PR #4515, pushed during this migration | `7dad3867982400c14ea4907329be27d24c4e845d` |
| Latest fetched `nv/main` | `7409c28e0071194af12fb8e36ca2dd7a6eb725c0` |
| Source / PR common ancestor | `81115337c62ec8e2c34ff069d0659e406da6c601` |

The source and original fetched PR have 106 and 105 exclusive commits respectively. These are
history counts, not 106 missing fixes: the upstream stack rewrites and squashes
several production changes. Main has seven commits absent from the PR head.
The new branch starts at the fetched PR head, merges those main changes, incorporates the updated PR head, and applies
the remaining production delta as a resolved squash. Both fetched PR #4515 and
main are ancestors of the resulting branch. The source snapshot is preserved
under `refs/archive/sft_v2_n4_oct6_source` in the AGA repository.

## Already represented by the PR stack

| Feature | Upstream commit | Migration treatment |
| --- | --- | --- |
| Checkpoint initialization and optimizer resume | `3124ed20d3` | Keep the stack implementation; retain extra production diagnostics and checkpoint memory release. |
| Processor/model configuration overrides | `186ff18aeb` | Keep the stack validation and override behavior; retain NM4-specific image/grid handling. |
| GTP initialization and producer data lanes | `62a195010e` | Keep the stack initialization and lane wiring; retain production tests and phase diagnostics. |
| BF16 override semantics and compact Muon layouts | `f3d82e3257` | Keep the stack implementation and tests. |
| Packing limits and padded HybridEP | `8785c3fab8` | Use the task encoder's packing configuration and current physical-row handling. |
| Balanced-knapsack bin slack | `7dad386798` | Implement and forward the production delta option in #4515 itself. Tested and pushed to the existing PR branch. |
| External encoder plugins, precomputed masks, and source templates | `842d12f975`, `f44d47c2c3` | Keep the plugin registry and source-template behavior alongside the NM4 built-ins. |

## Remaining production changes migrated

| Group | Principal files / scope | Why retained |
| --- | --- | --- |
| NM4 / legacy Nemotron Energon support | `multimodal/cookers/nemotron*.py`, `task_encoders/nemotron*.py`, registry/config wiring | Production tokenization, visual materialization, masks, cooker formats, and component options are absent from the generic PR stack. |
| Bridge / MCore integration | Bridge pin `4213458b3458d43b0841b815d62b7e76c9d97148`; nested MCore `1a216f62e797adb6ec5e97bb7d112b8b7ea136d3` | Preserve the NM4 provider, checkpoint/vision fixes, canonical zigzag CP outputs, and routing-map correction used by production. |
| Loss and model-data details | `algorithms/loss/{loss_functions,loss_input,utils}.py`, `models/megatron/{common,data,train}.py` | Answer metrics, mask/normalization details, and NM4 multimodal input handling beyond the generic stack. |
| Validation and diagnostics | SFT v2 loop, Energon worker/loader, policy placement and checkpoint helpers | Production validation behavior, exact-match measurements, loader diagnostics, and phase timing. |
| Production recipes and precision profiles | `examples/configs/sft_v2_tests/` | Preserve tracked 16/32/128-node recipes, data references, and the two committed precision ablation profiles. |
| Build/dependency support | `pyproject.toml`, `uv.lock`, Dockerfile, `scripts/build_nm4_fa4_enroot.sh` | NM4 Transformer Engine, Mamba, convolution, DeepEP, and FlashAttention-4 requirements. Keep main's newer dependencies elsewhere. |
| Focused regression tests and utilities | NM4 CP/GTP tests, cooker/visual tests, checkpoint and data-audit tools | Keep the evidence and standalone reproducers from production. |
| Existing tracked session records | `logbook/`, `session/`, `reports/` | Preserve previously committed provenance. No untracked experimental files, backup directories, or generated helpers are imported. |

## Integration changes required by the newer base

- Packing now belongs to `task_encoder.packing`; obsolete loader/controller
  packing arguments were removed. Built-in Nemotron options are validated from
  the generic plugin-compatible options dictionary.
- Loss-input preparation now lives in `algorithms/loss/loss_input.py`; the
  production metrics and tests were moved to the current public import paths.
- Keep main's per-turn preformatted tokenization; accept the legacy Nemotron
  `skip_chat_template` spelling through the corrected implementation.
- Preserve main's tracing, routed-expert data handling, row truncation,
  strict configuration validation, and layer-wise optimizer kwargs. Do not
  restore obsolete production copies of these APIs.
- Keep the upstream resume-sync helper rather than duplicate implementations.
- Restore the production balanced-knapsack delta option used by the 16-node
  recipe. Forward it through the task-owned packing API and reject it on
  other packing algorithms rather than silently ignoring it.

## Validation

| Check | Result |
| --- | --- |
| Precision / compact Muon focused tests | 16 passed |
| Checkpoint focused tests | 14 passed, 1 skipped |
| Model initialization / GTP lane setup tests | 21 passed |
| Packing / padded HybridEP focused tests | 19 passed |
| Knapsack regression tests, including balancing delta | 7 passed |
| Dependency lock freshness (`uv lock --check`) | Passed |
| Changed Python files: Ruff lint, import order, formatting | Passed |
| Data/tokenization regression tests in the training container | 168 passed |
| Runtime imports: Energon loader, Megatron setup, policy, SFT v2 | Passed with current Lens dependency in an isolated overlay |

The existing production container predates main's Lens `SpanRegistry` API.
Its original environment cannot import this newer main; a temporary overlay
with main's pinned Lens version allows the imports. No running worker
environment was changed. A fresh matching dependency environment/container is
required before a production launch. This migration does not claim distributed
GPU training or loss-parity validation on the new main base.

## Known baseline limitations

The two legacy suites `test_energon_nemotron_omni.py` and
`test_energon_nemotron_sft.py` fail all 25 cases when run directly in the
production worker environment. The same cases fail on the unchanged source
branch: missing active Energon worker/cache contexts, a null subflavors fixture,
and stale generation-prompt, lazy-audio and video-decoder expectations. They
are retained, and their failures are not counted as passing validation. A
broader data/packing run passed 121 other cases while exposing these failures.

The historical `nm4_generalist_16n.yaml` inherits
`sft_vlm_nm4_nano_energon_tp4ep4_v2.yaml`, which was already absent from the
source snapshot. The active standalone 16-node MXFP8, 32-node GTP/no-GTP and
128-node production recipes compose and validate. The historical recipe is
preserved as a source artifact, not validated as runnable.

Dependency integration retains main's newer shared packages and CUTLASS 4.6.2,
uses Quack 0.6.4 to match that CUTLASS version, and confines NM4's FA4/Quack
requirements to the Megatron extra. That extra uses the production Bridge
FlashInfer 0.6.8.post1 pair with its matching JIT cache. Other backend extras
keep their independent main dependency forks. Static metadata for the Git TE
build describes its bundled common/Torch extensions; cache-only FlashInfer
wheel metadata reuses the production lock's empty runtime dependency lists.

## GitLab lockfile size

The regenerated lock initially exceeded GitLab's 5 MiB file limit. Removing
optional whitespace around marker comparison operators reduces it to
5,196,726 bytes, with `uv lock --check` still passing. The resolved package
versions are unchanged by compaction. After future relocks, run
`uv run --no-project tools/compact_uv_lock.py`, then `uv lock --check`, before
committing. Only the unpublished migration commits were reconstructed to
exclude the oversized blob; the PR/main ancestry and source branch are preserved.

## Maintaining the branch

Use this isolated project for future main/stack updates, inspect the new delta,
and repeat dependency and regression checks. Do not reset the production
branch or replay every old source commit: much of that history is already
represented in the upstream stack. `nm4-oct6-source-commits.tsv` records the
source-only commit history and `nm4-oct6-migrated-files.tsv` records the final
file delta over PR #4515 plus fetched main.

## Additional requested EGTP2 experiment

Added `nm4_generalist_32n_mxfp8_cp64ep64_egtp2_gbs2_hybridep.yaml`: inherit
the GTP2 recipe, set dense remat to 1 and expert remat to 2, and isolate the
W&B/checkpoint names. YAML composition and Energon validation pass. The actual
production MCore config and rank-generator check pass for 128 ranks, dense DP2
and expert DP1, with 64 size-2 expert-remat groups. Added the file to the source
project as requested; existing source runtime files and branch history remain
unchanged. Submitted AGA job 1091066 against the original production runtime,
W&B run `1fbdea47`, for a comparison against existing GTP1/GTP2 runs.

The older tracked `nm4_generalist_16n.yaml` already refers to deleted
`sft_vlm_nm4_nano_energon_tp4ep4_v2.yaml` in the source branch. It is preserved as
historical material, not a validated launch recipe. The active 16/32/128-node
CP64 recipes, including the new EGTP2 variant, pass inheritance/config checks.

The migrated lock uses uv 0.12.2 (lock revision 4), which avoids older uv
expanding equivalent extra markers beyond GitLab’s 5 MiB file limit. The
main Dockerfile installer is updated to 0.12.2 and the compact lock passes
`uv 0.12.2 lock --check`. Older uv 0.11.x wants to regenerate this lock; use
the pinned build version for fresh environments.
