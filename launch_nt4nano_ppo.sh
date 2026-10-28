#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Candidate launcher: independent MIMO critic initialized from the actor backbone.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PPO_SCALE="${SCALE:-8}"
PPO_TAG="${RUN_TAG:-ppo-${PPO_SCALE}n}"
PPO_TIME="${WALLTIME:-}"
source "${NT4_PPO_ENV_FILE:-${PROJECT_ROOT}/../nt4-smoke.env}"
export SCALE="${PPO_SCALE}" RUN_TAG="${PPO_TAG}"
case "${SCALE}" in
  8) export CONFIG=examples/configs/recipes/vlm/vlm_ppo-nt4nano-clevr-mimo-8n4g-smoke.yaml ;;
  64) export CONFIG=examples/configs/recipes/vlm/vlm_ppo-nt4nano-clevr-mimo-64n4g.yaml ;;
  *) echo "ERROR: SCALE must be 8 or 64" >&2; exit 1 ;;
esac
export WALLTIME="${PPO_TIME:-$([[ ${SCALE} == 64 ]] && echo 4:00:00 || echo 1:30:00)}"
# The pinned image predates this critic class; reuse its existing mcore env.
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export SETUP_COMMAND="${SETUP_COMMAND:+${SETUP_COMMAND}
}bash /opt/nemo-rl/tools/setup_mimo_value_env.sh"
"${NT4_PPO_CHECK_PYTHON:-${PROJECT_ROOT}/../.venv-wandb/bin/python}" \
  "${PROJECT_ROOT}/../ppo-tools/preflight_nt4nano_ppo.py" "${PROJECT_ROOT}/${CONFIG}"
exec "${PROJECT_ROOT}/launch_clevr_vlm.sh"
