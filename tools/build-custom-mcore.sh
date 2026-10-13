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
# Replaces the Megatron-LM checkout nested inside the Megatron-Bridge
# submodule with a fresh clone of another repo/ref, then re-locks so
# megatron-core's resolved metadata matches the swapped-in source.
#
# megatron-core is installed as an editable path dependency from
# 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
# (declared in Megatron-Bridge's own pyproject.toml), so no pyproject.toml
# edit is needed here -- only the checkout on disk changes, and `uv lock`
# re-reads its metadata from that path.
#
# Authentication (e.g. for a private repo) must already be configured by the
# caller (git credential helper, SSH agent, etc.) before invoking this script.

set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"

GIT_URL=${1:-https://github.com/NVIDIA-NeMo/nv-mistralai-megatron.git}
GIT_REF=${2:-main}

MCORE_DIR="$REPO_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

echo "Replacing Megatron-LM (megatron-core) at $MCORE_DIR with:"
echo "  Git URL: $GIT_URL"
echo "  Git ref: $GIT_REF"

rm -rf "$MCORE_DIR"
git clone "$GIT_URL" "$MCORE_DIR"
cd "$MCORE_DIR"
git checkout "$GIT_REF"

cd "$REPO_ROOT"
# Skippable for the Docker build specifically: the TensorRT-LLM workspace
# path-dependency isn't copied into that build context until a later layer,
# so `uv lock` here would fail trying to build its metadata from an absent
# directory. The Dockerfile sets SKIP_UV_LOCK=1 and runs the real `uv lock`
# itself once the full workspace is present.
if [[ -n "${SKIP_UV_LOCK:-}" ]]; then
  echo "SKIP_UV_LOCK is set; leaving uv.lock for the caller to regenerate."
else
  echo "Updating uv.lock for the replaced megatron-core source..."
  uv lock
fi

cat <<EOF
[INFO] megatron-core now resolves from the local clone at $MCORE_DIR (git ref: $GIT_REF).
[WARNING] This is a plain directory swap, not a submodule update: the checkout
          no longer matches the remote registered in Megatron-Bridge's
          .gitmodules. Do not commit this change from inside a checked-out
          working tree -- it is intended for one-off/CI image builds only.
EOF
