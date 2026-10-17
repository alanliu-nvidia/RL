# NM4 MIMO

| Config | Node blocks | Encoder + language GPUs | CP / DP / EP | GBS / MBS / GA | Tokens | Steps |
|---|---|---:|---|---|---:|---:|
| [Smoke](vlm_sft-mimo-nm4-5n4g-2k-smoke.yaml) | `1,4` | 4 + 16 | 4 / 4 / 16 | 16 / 1 / 4 | 2,048 | 20 |
| [18 nodes](vlm_sft-mimo-nm4-18n4g-512k.yaml) | `2,16` | 8 + 64 | 64 / 1 / 64 | 4 / 1 / 4 | 524,288 | 500 |
| [512 language GPUs](vlm_sft-mimo-nm4-144n4g-512llm-512k.yaml) | `16` x 9 | 64 + 512 | 64 / 8 / 64 | 32 / 1 / 4 | 524,288 | 500 |

## Launch

Run on a Slurm login node (`launch.sh` calls `sbatch`), from the root of a clone with submodules
(`git submodule update --init --recursive`):

```bash
export CONTAINER=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_ci/nemo_rl_ci/sqsh_files/rl-gym.73975125.sqsh
export MOUNTS="/lustre/fsw:/lustre/fsw,/scratch/fsw:/scratch/fsw,$PWD:/opt/nemo-rl"
export GPUS_PER_NODE=4 SBATCH_ACCOUNT=<account> SBATCH_PARTITION=batch SBATCH_TIMELIMIT=02:00:00
CONFIGS=examples/configs/sft_v2_tests/mimo
bash "$CONFIGS/launch.sh" "$CONFIGS/vlm_sft-mimo-nm4-5n4g-2k-smoke.yaml" 1,4
bash "$CONFIGS/launch.sh" "$CONFIGS/vlm_sft-mimo-nm4-18n4g-512k.yaml" 2,16
bash "$CONFIGS/launch.sh" "$CONFIGS/vlm_sft-mimo-nm4-144n4g-512llm-512k.yaml" 16,16,16,16,16,16,16,16,16
```

Extra arguments are Hydra overrides, for example `sft.max_num_steps=100 logger.wandb_enabled=false`.

## Starting checkpoint

The shared checkpoint views on AGA are under
`/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/ykarnati/nm4_sft`:

| Version | Directory under the shared root | Starting weights |
|---|---|---|
| Pretraining | `pretraining/iter_0568125` | Original pretraining weights, including the original embedding and output rows. |
| Embedding reinitialized | `embedding_reinitialized/iter_0568125` | Production starting weights: 2,026 rows in both the input embedding and output layer were reinitialized. |

Both are MIMO-compatible metadata views of iteration 568125, with direct symlinks to the original
tensor files. The original files must remain available. These views contain model weights for SFT
initialization, not optimizer/RNG state for resuming a training run. Access requires the existing
Nemotron project permissions; `PROVENANCE.json` in each view records its sources and metadata hashes.

`launch.sh` pins the **embedding-reinitialized version**, preserving the previous recipe's starting
weights while using the per-expert checkpoint chunks. The two production YAMLs set the same
`checkpointing.pretrained_checkpoint.path` for direct `run_sft_v2.py` invocations; smoke inherits it.
To select the pretraining version explicitly:

```bash
export MIMO_PRETRAINED_CHECKPOINT=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/ykarnati/nm4_sft/pretraining/iter_0568125
bash "$CONFIGS/launch.sh" "$CONFIGS/vlm_sft-mimo-nm4-18n4g-512k.yaml" 2,16
```

Precedence is: an explicit `checkpointing.pretrained_checkpoint.path=...` command-line override,
then `MIMO_PRETRAINED_CHECKPOINT`, then the launcher's pinned default. This controls initialization;
an existing saved run under `checkpointing.checkpoint_dir` takes precedence for automatic resume.
Use a fresh output directory when comparing starting checkpoint versions.

## Notes

- The container is a frozen environment: `launch.sh` runs `uv run --no-sync`. NeMo RL prints a
  "Container/Code Version Mismatch" warning; this is expected.
- `MOUNTS` maps the repository to `/opt/nemo-rl` (Ray workers import NeMo RL, Bridge and Core from there) and adds
  `/lustre/fsw` and `/scratch/fsw` (checkpoint, model, tokenizer and data paths in the recipes).
- W&B is on (base recipe). Without W&B credentials, add `logger.wandb_enabled=false`.
- The 18-node and 144-node recipes save a checkpoint to `/data/nemorl-runs/...` every 100 steps. Mount that path, or add
  `checkpointing.checkpoint_dir=<path>` or `checkpointing.enabled=false`. The smoke does not save.
- Logs: `<repository>/<job id>-logs/`. TensorBoard: `logs/<recipe name>/exp_*/tensorboard/`.
- The shared views retain per-expert chunks from pretraining, avoiding the old `rf2` view's full-tensor
  reads for each expert. Their source views passed 5-node loading/training smoke tests (jobs 1274195
  for pretraining and 1274045 for embedding reinitialized). The published paths were verified by
  metadata hashes and tensor-link targets; GPU tests were not repeated after publication.
  See the [checkpoint study](http://10.57.213.224:8765/reports/nm4-vlm/post_training/pretrain_ckpt_load.md).
  Loading at 144 nodes has not been measured with these views.
- Reference (AGA, rl-gym.73975125): smoke 20/20, about 9.2 s per update; 18 nodes with `sft.max_num_steps=100`:
  10.12 s per update (mean of updates 11-100).
