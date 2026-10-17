# Opt-in NM4 functional CI

For the separate 18-node MIMO / 16-node non-MIMO comparison and MIMO
save/restart loop, see [MIMO CI](mimo-ci.md). It reuses one 2+16-node
heterogeneous allocation and is selected with `--suite mimo_suite.yaml`.

These 14 recipes exercise the NM4 production architecture for 20 optimizer steps
(10 steps per restart phase). They were copied from AGA's `sft_v2_up_n4` project
to HSG's `sft_v2_n4_oct6_rebase` project. The suite runs only when explicitly
selected; it is not included in the default GB200 nightly list and does not
submit Slurm jobs automatically.

Generalist recipes inherit the data path and sequence length from
`../sft_v2_tests/nm4_generalist_16n_mxfp8_cp64ep64_gbs4.yaml`. The AGA source used
524288 tokens; the HSG parent currently uses **131072**. Always inspect the
resolved plan/config rather than assuming those settings are identical.
The CI common fragment selects `/data/models/nm4_nano_bridge`, seed 42 and Muon.
MXFP8 uses `nm4_mxfp8_tp4_precision.yaml`; HybridEP is the default dispatcher.
The checkpoint alias and CLEVR alias `/data/nemorl-datasets/clevr` must be mounted
or overridden to readable paths. Training data is not copied by this suite.

| Recipe | Nodes × GPUs | CP / EP | Dense GTP / expert GTP | GBS | Purpose |
| --- | --- | --- | --- | --- | --- |
| `generalist_16n_bf16_hybridep.yaml` | 16 × 4 | 64 / 64 | 1 / 1 | 1 | BF16 historical-loss baseline |
| `generalist_16n_mxfp8_hybridep.yaml` | 16 × 4 | 64 / 64 | 1 / 1 | 1 | MXFP8 versus BF16; dispatcher reference |
| `generalist_16n_mxfp8_alltoall.yaml` | 16 × 4 | 64 / 64 | 1 / 1 | 1 | alltoall versus HybridEP |
| `generalist_32n_mxfp8_gtp1.yaml` | 32 × 4 | 64 / 64 | 1 / 1 | 2 | GTP and Muon reference |
| `generalist_32n_mxfp8_gtp2.yaml` | 32 × 4 | 64 / 64 | 2 / 1 | 2 | Dense GTP2 parity |
| `generalist_32n_mxfp8_egtp2.yaml` | 32 × 4 | 64 / 64 | 1 / 2 | 2 | Expert GTP2 parity |
| `generalist_32n_mxfp8_muon_padded.yaml` | 32 × 4 | 64 / 64 | 1 / 1 | 2 | Explicit layer-wise padded layout |
| `generalist_32n_mxfp8_muon_compact.yaml` | 32 × 4 | 64 / 64 | 1 / 1 | 2 | Compact layout, distributed optimizer disabled |
| `clevr_2n_mxfp8_cp8ep8_2k.yaml` | 2 × 4 | 8 / 8 | 1 / 1 | 4 | CLEVR DP1 reference, 2K tokens |
| `clevr_4n_mxfp8_cp16ep16_2k.yaml` | 4 × 4 | 16 / 16 | 1 / 1 | 4 | CLEVR CP/EP parity against two-node DP1 reference |
| `clevr_16n_mxfp8_cp16ep16_2k.yaml` | 16 × 4 | 16 / 16 | 1 / 1 | 4 | CLEVR DP4 execution/finite-metric check, 2K tokens |
| `checkpoint_16n_uninterrupted.yaml` | 16 × 4 | 64 / 64 | 1 / 1 | 1 | Uninterrupted 20-step reference |
| `checkpoint_16n_save10.yaml` | 16 × 4 | 64 / 64 | 1 / 1 | 1 | Ten steps, then save weights/optimizer/loaders |
| `checkpoint_16n_resume20.yaml` | 16 × 4 | 64 / 64 | 1 / 1 | 1 | Fresh-process resume, steps 11–20 |

`_generalist_common.yaml` and `_clevr_common.yaml` are inheritance fragments.
`suite.yaml` is comparison metadata, not a training config.

## Opt-in CI runner

Start at the repository root inside a compatible GB200 training container. Provide
a shared artifact root visible at the same path on every node, and one fresh tag
for the entire suite. No W&B credentials are needed by the runner or offline checker.

```bash
export NM4_CI_RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export NM4_CI_ARTIFACTS=/data/nemorl-runs/nm4-ci
uv run --no-sync examples/configs/nm4_CI/run_ci.py plan \
  --run-tag "$NM4_CI_RUN_TAG" --artifacts "$NM4_CI_ARTIFACTS"
```

The JSON plan gives each case's allocation size, segment size, sequence limit,
checkpoint and dependency. Allocate that geometry using rem and start Ray with the
normal project wrapper. Run one case per allocation, or run cases with identical
geometry sequentially in a persistent allocation:

```bash
uv run --no-sync examples/configs/nm4_CI/run_ci.py run \
  --case generalist_16n_bf16_hybridep \
  --run-tag "$NM4_CI_RUN_TAG" --artifacts "$NM4_CI_ARTIFACTS"
```

Repeat with each case ID from `suite.yaml`. Run `checkpoint_16n_save10` to completion
before `checkpoint_16n_resume20`; the latter always starts a fresh driver process.
Use the same tag, artifact root and training overrides for both phases and their
uninterrupted reference. Case directories cannot be reused. A failed retry needs
a new tag, including both phases of a failed restart comparison.

The runner requires exactly the configured Ray GPU nodes and GPUs per node; a
misallocated case fails instead of passing as a skip. Set Slurm's domain segment
size to the plan's `segment_size` (16 for generalist, 1/4/16 for CLEVR). The runner
checks Ray GPU geometry; Slurm segment placement remains the allocation wrapper's
responsibility. Use idle allocations to avoid competing GPU workloads.

`ci_runtime.yaml` selects **auto** for language and vision attention, expandable
CUDA allocator segments and TensorBoard logging with W&B disabled. It does not
patch FA4 or remove any outer `elect_one()` wrapper. The current HSG validation
image needs the compatible **QuACK 0.6.1 / CUTLASS DSL 4.6.0** package setup for
FA4 4.0.0b19; it also needs Energon installed in the driver environment. Prepare
those dependencies with the established container setup before running these
cases. The runner does not install or modify packages.

Repeat `--override 'key=value'` to supply readable checkpoint/data/cache paths.
For example, `--override 'checkpointing.pretrained_checkpoint.path=/shared/model'`.
Apply identical training overrides to a candidate and its reference. The runner
owns log/checkpoint output paths and rejects changes to expected training steps
or failure tolerance. Changes to input data, packing, checkpoint or sequence
length require reviewing the historical BF16 baseline.

Each case writes `resolved_config.yaml`, `driver.log`, `metrics_export.log`,
`metrics.json` and `result.json` under `<artifact-root>/<tag>/<case>`. The driver
must exit zero, report the expected final controller step, and produce every
expected loss and gradient-norm scalar with finite values. Conflicting TensorBoard
records fail during export. `passed` in a case result means successful execution;
loss parity is checked separately across the whole suite:

```bash
uv run --no-sync examples/configs/nm4_CI/check_losses.py \
  --artifacts "$NM4_CI_ARTIFACTS/$NM4_CI_RUN_TAG" \
  --output "$NM4_CI_ARTIFACTS/$NM4_CI_RUN_TAG/summary.json"
```

All 14 cases and all comparison gates must pass for the suite to pass. Missing
cases, failed drivers, incomplete histories and failed comparisons produce a
nonzero checker exit status. The default driver timeout is two hours; change it
with `--timeout-seconds`. On timeout the local driver process group is stopped;
release the failed allocation to clean up any remaining Ray workers.

An external CI job can explicitly call `plan`, allocate one matrix job per case,
run each case, collect the shared artifacts, then invoke `check_losses.py` as its
final gate. Keep save/resume phases sequential on a shared filesystem. No default
GitHub workflow or nightly list is changed.

## Existing W&B comparison mode

The copied recipes can still be launched directly with `uv run --no-sync
examples/run_sft_v2.py --config ...`. Map each case to `entity/project/run_id` in
JSON and invoke:

```bash
uv run --no-sync examples/configs/nm4_CI/check_losses.py \
  --runs /path/to/nm4-ci-runs.json --output /path/to/nm4-ci-results.json
```

This mode requires finished W&B runs and exact finite loss histories using
`nemo_rl/step`, not W&B's internal `_step`. It does not prove the local driver exit
status or gradient finiteness; use the artifact runner for those execution gates.

## CPU validation

```bash
uv run --no-sync pytest tests/unit/tools/test_nm4_ci.py
```

The tests cover real recipe composition, BF16 precision clearing, parallelism,
restart prerequisites, failed/incomplete drivers, timeouts, metric failures and
strict versus inclusive loss thresholds. Fake training processes exercise runner
control flow; they do not establish GPU loss parity. GPU validation of all 14
cases is a separate opt-in run.

## Loss gates

- BF16 reference: all first 20 losses from
  [W&B run 0ge5vk10](https://wandb.ai/nvidia/rohit-sft_v2_up_n4/runs/0ge5vk10)
  are saved in `bf16_reference.json`. Step 1 was **0.9013324333** and step 20 was
  **1.0587107680**. A 10% maximum absolute relative-error envelope is used for this
  rough historical regression gate; it is configurable in `suite.yaml`.
- MXFP8 versus the newly executed BF16 baseline: fail when maximum absolute
  per-step relative loss difference is **greater than 10%**.
- GTP2, EGTP2 and both Muon variants versus the 32-node GTP1 reference:
  maximum absolute per-step relative difference must be **less than 0.5%**.
- alltoall versus the 16-node GTP1 HybridEP reference, and four-node CLEVR
  CP16/EP16 versus two-node CP8/EP8: the same **less than 0.5%** gate.
- Restart phase one versus uninterrupted steps 1–10, and restart phase two versus
  uninterrupted steps 11–20: the same **less than 0.5%** gate. Comparing step 11
  against uninterrupted step 11 detects a restart spike without confusing it with
  ordinary batch-to-batch loss variation.

The comparison is `100 * abs(candidate_loss - reference_loss) / abs(reference_loss)`.
The maximum is over individual matched steps, not a running average.

The historical BF16 run used alltoall; this clean BF16 case uses HybridEP as requested.
The saved historical losses describe the original AGA inputs. HSG's inherited
sequence length and dataset paths differ, so this historical gate must be reviewed
or replaced with a versioned baseline measured on the intended CI inputs before
claiming historical parity. The paired, newly executed reference gates still
compare matching configurations within the same suite invocation.
The production dataset contains known skipped-sample/metadata warnings. Preserve
their logs and check that compared runs use the same data; do not automatically
interpret a changed data stream as a model regression. Changes to checkpoint
contents, dataset contents or data ordering require reviewing the historical gate.

## Checkpoint restart

Run the uninterrupted case on a fresh directory. Run `checkpoint_16n_save10.yaml`
in a separate process; wait for successful completion and finalized `step_10`.
Require `training_info.json`, `sft_v2_loaders.pt`, and the saved Megatron checkpoint
with optimizer state. Then run a fresh driver process with:

```bash
env NEMO_RL_REQUIRE_RESUME=1 \
  NEMO_RL_EXPECTED_RESUME_CHECKPOINT="/data/nemorl-runs/nm4-ci/${NM4_CI_RUN_TAG}/checkpoint-16n-split/step_10" \
  uv run --no-sync examples/run_sft_v2.py \
  --config examples/configs/nm4_CI/checkpoint_16n_resume20.yaml
```

The opt-in runner sets these variables on the driver and relocates the shared
checkpoint directory under `<artifact-root>/<tag>/checkpoints/checkpoint-16n-split`.
The manual command above applies only to direct recipe launches.
Those variables must be set on the driver, not only in the worker environment.
They reject silently falling back to the pretrained model. The checker expects
resume logs at steps **11–20**, so a restart from step 1 fails. Keep the same node
geometry and do not allow remx's command template to replace the shared checkpoint
directory with a different experiment-specific directory for each phase. Explicitly
override `checkpointing.checkpoint_dir` at the end of the rendered command if needed.

The constant learning-rate schedule is unchanged between phases. Loss continuation
checks the combined behavior of weights, optimizer, scheduler and loader restoration;
it does not independently prove equality of every saved RNG/optimizer tensor.

## CLEVR context parallelism

The two-node reference uses CP8/EP8. Both the four- and sixteen-node cases use
CP16/EP16. TP1, PP1, dense/expert GTP1, GBS4, microbatch1, 2048-token sequences
and 20 optimizer steps are common to all three cases.

The two- and four-node cases have DP1 and four microbatches per step. Their
loss comparison uses matching Energon data-lane geometry and tests the combined
CP/EP change. The sixteen-node case has DP4 and one microbatch per step; it checks
execution and finite losses/gradients without claiming loss parity against the
DP1 references.

The four-node parallelism matches
`../sft_v2_tests/nm4_clevr_4n.yaml`. Its former CP4/EP4 configuration placed
128 of the 512 experts on each GPU and exhausted memory while creating Muon's
FP32 master parameters. EP16 places 32 experts on each GPU.

The historical one-node CP4/EP4 recipe has 128 experts per GPU. It failed
during Muon initialization on HSG allocation 7948255 and checkpoint dequantization
on AGA allocation 1295066 (276.60 GiB used out of 276.62 GiB). BF16 parameter
storage and disabling the quantized-weight cache still exhausted memory during
forward execution. Keep that recipe for reproducing the failure, but use the
two-node CP8/EP8 reference in the required suite: 64 experts per GPU, with the
same DP1 data lane, batch size, optimizer, precision, and 0.5% parity gate.
All CLEVR cases use `/data/nemorl-datasets/clevr` and matching 2048-token packing limits.

### Four-node GPU validation

HSG allocation 7951113 completed all 20 steps of
`clevr_4n_mxfp8_cp16ep16_2k` with exit code 0 and finite loss/gradient metrics,
using `rl-gym:pipe.72296264.sqsh`. Loss went from 10.0107 to 5.8952; mean policy
time over steps 2–20 was 12.95 seconds. This is an execution check, not a loss
parity result. The validation checkout was oct6 commit `0512eb04` plus the
replayed MR !8 fix (`242a3a7b`).

The passing allocation placed all four nodes in one NVLink domain using Slurm
segment size 4. It also used a runtime override of 16 ranks per NVLink domain;
the inherited production value is 64. Reproduce on an equivalent existing
four-node Ray allocation with a fresh run tag:

```bash
uv run --no-sync examples/configs/nm4_CI/run_ci.py run \
  --case clevr_4n_mxfp8_cp16ep16_2k \
  --run-tag clevr4n-validation-NEW_TAG \
  --artifacts /data/nm4-ci-results \
  --override cluster.segment_size=4 \
  --override policy.megatron_cfg.hybridep_num_ranks_per_nvlink_domain=16
```

A proposed four-node TP8/CP2/EP16 replacement reference failed before training:
Bridge's vision-projector config sets `num_attention_heads=1` while inheriting
TP8, triggering MCore's divisibility check. That candidate is not registered
in this suite.

## Muon layouts

Padded/GTP1 cases explicitly enable `use_layer_wise_param_layout`,
`use_distributed_optimizer` and `use_layer_wise_distributed_optimizer`. The compact
case disables the first two but keeps `use_layer_wise_distributed_optimizer: true`
for whole-matrix Muon ownership, as required by the production setup code.
All other optimizer hyperparameters, data, precision and
parallelism are identical. This checks the supported compact/padded optimizer setups;
it is not an isolated padding-only switch. No Adam substitution is made.
