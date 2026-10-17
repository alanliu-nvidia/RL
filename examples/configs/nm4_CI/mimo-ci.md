# MIMO parity and checkpoint restart

`mimo_suite.yaml` is an opt-in suite for one persistent 18-node allocation.
It runs these four processes in order, then reports three loss comparisons:

| Case | Nodes used | Optimizer steps | Comparison |
| --- | --- | --- | --- |
| `generalist_18n_mimo_mxfp8_hybridep` | 2 encoder + 16 language | 1–20 | Uninterrupted MIMO reference |
| `generalist_16n_mxfp8_hybridep_aga` | Same 16 language nodes | 1–20 | Non-MIMO versus MIMO |
| `checkpoint_18n_mimo_save10` | 2 encoder + 16 language | 1–10 | Fresh MIMO versus reference; save at step 10 |
| `checkpoint_18n_mimo_resume20` | 2 encoder + 16 language | 11–20 | Fresh-process restart versus reference |

All comparisons use the uninterrupted **MIMO** reference from the same suite run.
Each matched step must have absolute relative loss error **below 0.5%**. The
checker reports maximum and mean errors. Step 11 is compared with reference
step 11, so a restart spike cannot be hidden by averaging steps 1–20.

## Prerequisites and placement

Run on the MIMO-capable NeMo-RL/Bridge/MCore stack. MR 10 supplies heterogeneous
placement; its original base alone does not include the later MIMO training
implementation. Apply this test commit when replaying MR 10 on the current
MIMO-capable branch. This test commit does not change submodule pins or import
the runtime implementation.

The 8 encoder ranks split each pack's media in the loader, so MCore must include
nv-mistralai-megatron#557 (fan-in gradients split by the forward sizes). The
stack must also include the distributed optimizer checkpoint mapping correction
in MCore commit
`bf28746e7501228c0a07eb10a3f10e35d3ceebf9` (PR 698), or equivalent fixes.
The latter corrects optimizer parameter indexing for mixed FP32/BF16 groups.
The suite keeps `use_distributed_optimizer: true` and saves optimizer state.

Allocate two Slurm heterogeneous components in this exact order:

| Component / Ray node block | Nodes | GPUs/node | Slurm segment constraint | MIMO ranks |
| --- | --- | --- | --- | --- |
| 0: encoder | 2 | 4 | Omit the segment flag | 0–7 |
| 1: language | 16 | 4 | `--segment=16` | 8–71 |

Use the project's container, mounts, and normal `ray.sub` wrapper. The wrapper
must expose `node_block_0` and `node_block_1` resources on the corresponding Ray
nodes. Keep `cluster.segment_size: null` in the recipes: segment constraints
are per Slurm component. The runner verifies the 2+16 node counts and 4 GPUs
per selected node; the Slurm wrapper enforces NVLink segment placement.

The non-MIMO case uses `run_sft_on_node_blocks.py` to call the ordinary SFT
entry point with `node_block_constraints(..., blocks=[1])`. Thus it uses the
same 16 language nodes after MIMO finishes. The encoder nodes stay allocated.
This selection is confined to that test process; production SFT code is unchanged.

## Matching inputs

Both arms inherit `generalist_16n_mxfp8_hybridep.yaml`, including language
CP64/EP64, GBS1, MXFP8, HybridEP, Muon, seed 42, and sequence capacity. Both
use packing buffer **500** and the same AGA-accessible blend:
`/opt/nemo-rl/examples/configs/sft_v2_tests/sft_super35_generalist_full_cmh.yaml`.
Despite its filename, this blend uses `/data/super3p5vl-textdatasets` and DSS
aliases available through the AGA project mounts. Mount the checkout at
`/opt/nemo-rl` on every node, or override `data.train.path` for all cases.
The inherited DSS cache directory must also be mounted on every node.

Non-MIMO loads `/data/models/nm4_nano_bridge`. Set `NM4_CI_MIMO_CHECKPOINT` to a
MIMO checkpoint converted from those **same weights**. A metadata/key-prefix
view sharing the underlying tensor shards is suitable. Do not compare unrelated
checkpoints. The default alias is `/data/models/nm4_nano_bridge_mimo`.

## Run in the existing head container

Start from the repository root, with Ray running and the allocation idle:

```bash
export NM4_CI_RUN_TAG="mimo-$(date -u +%Y%m%dT%H%M%SZ)"
export NM4_CI_ARTIFACTS=/data/nemorl-runs/nm4-ci
export NM4_CI_MIMO_CHECKPOINT=/data/models/nm4_nano_bridge_mimo_pr10

uv run --no-sync examples/configs/nm4_CI/run_ci.py plan \
  --suite examples/configs/nm4_CI/mimo_suite.yaml \
  --run-tag "$NM4_CI_RUN_TAG" --artifacts "$NM4_CI_ARTIFACTS"

bash examples/configs/nm4_CI/run_mimo_ci.sh
```

The shell runner passes additional arguments, such as `--override 'sft.seed=42'`
or `--timeout-seconds 7200`, to all four cases. Keep training overrides identical
across cases. A global pretrained-checkpoint override would replace the MIMO
checkpoint too; use the MIMO environment variable to select its converted view.
Artifact paths must be shared across all nodes. No W&B credentials are needed.

Each case records its resolved config, driver log, metrics and execution result.
The save and restart cases share one checkpoint directory. Restart requires a
successful save result, complete steps 1–10 and finalized step-10 artifacts.
It sets `NEMO_RL_REQUIRE_RESUME=1` and the exact expected checkpoint, so an
accidental fresh run cannot pass. Restart metrics must contain exactly 11–20.

The loop stops on a failed case. A failed retry requires a fresh run tag. It
does not submit or cancel allocations. On driver failure, inspect and clean up
any remaining Ray actors before retrying. After all four cases complete, the
checker writes `<artifact-root>/<tag>/mimo_summary.json` and exits nonzero if
any of the three comparisons fails. Case `status: passed` alone only establishes
successful execution; the summary establishes loss parity.

For an individual phase, use `run_ci.py run --suite ... --case ...` with the same
tag and artifact root. For offline rechecking:

```bash
uv run --no-sync examples/configs/nm4_CI/check_losses.py \
  --suite examples/configs/nm4_CI/mimo_suite.yaml \
  --artifacts "$NM4_CI_ARTIFACTS/$NM4_CI_RUN_TAG" \
  --output "$NM4_CI_ARTIFACTS/$NM4_CI_RUN_TAG/mimo_summary.json"
```

CPU harness tests are in `tests/unit/tools/test_nm4_ci.py`. They verify config
composition, placement selection, restart prerequisites, failure handling and
comparison gates. GPU loss parity requires running the suite above.
