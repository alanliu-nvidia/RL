#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# Run inside the head container of an existing 2+16-node heterogeneous Ray allocation.
set -euo pipefail
: "${NM4_CI_RUN_TAG:?Set a fresh run tag}"
: "${NM4_CI_ARTIFACTS:?Set a shared artifact directory}"
ci_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for case in generalist_18n_mimo_mxfp8_hybridep generalist_16n_mxfp8_hybridep_aga \
  checkpoint_18n_mimo_save10 checkpoint_18n_mimo_resume20; do
  uv run --no-sync "${ci_dir}/run_ci.py" run --suite "${ci_dir}/mimo_suite.yaml" \
    --case "${case}" --run-tag "${NM4_CI_RUN_TAG}" --artifacts "${NM4_CI_ARTIFACTS}" "$@"
done
uv run --no-sync "${ci_dir}/check_losses.py" --suite "${ci_dir}/mimo_suite.yaml" \
  --artifacts "${NM4_CI_ARTIFACTS}/${NM4_CI_RUN_TAG}" \
  --output "${NM4_CI_ARTIFACTS}/${NM4_CI_RUN_TAG}/mimo_summary.json"
