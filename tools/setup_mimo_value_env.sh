#!/bin/bash
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

# Compatibility for pinned images built before the MIMO critic was registered.
# Runs inside each job's named container before Ray starts; no shared env edits.
set -euo pipefail
PPO_VENV_ROOT="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
PPO_CRITIC_VENV="${PPO_VENV_ROOT}/nemo_rl.models.value.workers.megatron_mimo_value_worker.MegatronMIMOValueWorker"
if [[ -x "${PPO_CRITIC_VENV}/bin/python" ]]; then
  exit 0
fi
for ppo_existing_venv in \
  "${PPO_VENV_ROOT}/nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker" \
  "${PPO_VENV_ROOT}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"; do
  if [[ -x "${ppo_existing_venv}/bin/python" ]]; then
    ln -s "${ppo_existing_venv}" "${PPO_CRITIC_VENV}"
    echo "MIMO critic reuses existing backend: ${ppo_existing_venv}"
    exit 0
  fi
done
echo "ERROR: pinned image has no existing Megatron backend in ${PPO_VENV_ROOT}" >&2
exit 1
