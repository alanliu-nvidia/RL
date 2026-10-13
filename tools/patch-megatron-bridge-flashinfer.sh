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
#
# Repoints Megatron-Bridge's own flashinfer-python/flashinfer-cubin
# [tool.uv.sources] entries from the public "flashinfer" index to
# gitlab-internal-pypi, matching the vllm/mcore extras' pin in the root
# pyproject.toml.
#
# Why this is needed: uv requires a single, workspace-global index per
# package name -- the check is not fork-aware, so even a source scoped to an
# extra that never actually resolves alongside Megatron-Bridge still
# conflicts with Megatron-Bridge's own unconditional "flashinfer" claim for
# the same package (verified empirically; see git history for the
# investigation). There is no way to keep Megatron-Bridge on a different
# flashinfer-python/-cubin source than the rest of the workspace, so this
# patch exists to make them agree rather than conflict.
#
# This is a plain, uncommitted file edit inside the build context -- same
# pattern as build-custom-mcore.sh's source swap. Re-run after every
# Megatron-Bridge bump; it does not persist across `git submodule update`.

set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"
MEGATRON_BRIDGE_PYPROJECT="$REPO_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/pyproject.toml"

if [[ ! -f "$MEGATRON_BRIDGE_PYPROJECT" ]]; then
  echo "[ERROR] $MEGATRON_BRIDGE_PYPROJECT not found." >&2
  exit 1
fi

sed -i \
  -e 's/^flashinfer-python = { index = "flashinfer" }$/flashinfer-python = { index = "gitlab-internal-pypi" }/' \
  -e 's/^flashinfer-cubin = { index = "flashinfer" }$/flashinfer-cubin = { index = "gitlab-internal-pypi" }/' \
  "$MEGATRON_BRIDGE_PYPROJECT"

if ! grep -q 'name = "gitlab-internal-pypi"' "$MEGATRON_BRIDGE_PYPROJECT"; then
  cat >>"$MEGATRON_BRIDGE_PYPROJECT" <<'EOF'

[[tool.uv.index]]
name = "gitlab-internal-pypi"
url = "https://gitlab-master.nvidia.com/api/v4/projects/65523/packages/pypi/simple"
explicit = true
authenticate = "always"
EOF
fi

echo "[INFO] Patched $MEGATRON_BRIDGE_PYPROJECT to resolve flashinfer-python/-cubin from gitlab-internal-pypi."
