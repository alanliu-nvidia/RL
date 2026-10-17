#!/usr/bin/env bash
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

set -euo pipefail
config=${1:?Usage: bash launch.sh CONFIG NODE_BLOCKS [HYDRA_OVERRIDE ...]}
IFS=, read -r -a blocks <<< "${2:?Set node blocks, for example 2,16}"
shift 2
: "${CONTAINER:?}" "${MOUNTS:?}" "${GPUS_PER_NODE:?}"
# Pin the starting weights; use the pretraining sibling for the original embedding/output rows.
# Explicit Hydra overrides below take precedence over this launcher default.
MIMO_PRETRAINED_CHECKPOINT=${MIMO_PRETRAINED_CHECKPOINT:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/ykarnati/nm4_sft/embedding_reinitialized/iter_0568125}
printf -v COMMAND '%q ' uv run --no-sync examples/run_sft_v2.py --config "$config" \
  "checkpointing.pretrained_checkpoint.path=$MIMO_PRETRAINED_CHECKPOINT" "$@"
export COMMAND
args=()
for nodes in "${blocks[@]}"; do
  ((${#args[@]} == 0)) || args+=(:)
  args+=(--nodes="$nodes" --segment="$nodes" --gpus-per-node="$GPUS_PER_NODE" --exclusive)
done
exec sbatch "${args[@]}" ray.sub
