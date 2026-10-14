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
"""Diagnose a mixed ("hybrid") nvidia-cutlass-dsl tree in a virtualenv.

nvidia-cutlass-dsl-internal ships its own copy of the cutlass DSL under the same
nvidia_cutlass_dsl/ paths as the PyPI nvidia-cutlass-dsl-libs-* wheels. When both
are installed, whichever wheel is unpacked last wins per shared file. This prints,
for every cutlass distribution in the venv:

  * how many of its files no longer match the hash in its own RECORD (i.e. were
    overwritten by another wheel), and which wheel's content they now hold;
  * how many paths are shipped by more than one wheel;
  * where ``DSLUserCodeRuntimeError`` is defined and which wheel ships that file.

Usage: python3 tools/diagnose_cutlass_tree.py <venv-path>

Uses only the standard library so it can run before any dependency is usable.
"""

import base64
import glob
import hashlib
import os
import sys


def _digest(path: str) -> str:
    with open(path, "rb") as fh:
        raw = hashlib.sha256(fh.read()).digest()
    return "sha256=" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _read_records(site_packages: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for record in glob.glob(f"{site_packages}/*.dist-info/RECORD"):
        dist = os.path.basename(os.path.dirname(record))[: -len(".dist-info")]
        if "cutlass" not in dist.lower():
            continue
        with open(record) as fh:
            rows = (line.rstrip("\n").rsplit(",", 2) for line in fh)
            records[dist] = {r[0]: r[1] for r in rows if len(r) == 3 and r[1]}
    return records


def main(venv: str) -> int:
    matches = glob.glob(f"{venv}/lib/python*/site-packages")
    if not matches:
        print(f"[cutlass-diagnose] no site-packages under {venv}")
        return 2
    site_packages = matches[0]
    records = _read_records(site_packages)
    print(f"[cutlass-diagnose] {venv}")
    print("  cutlass distributions:", ", ".join(sorted(records)) or "(none)")

    owners: dict[tuple[str, str], list[str]] = {}
    shipped_by: dict[str, list[str]] = {}
    for dist, files in records.items():
        for path, file_hash in files.items():
            owners.setdefault((path, file_hash), []).append(dist)
            shipped_by.setdefault(path, []).append(dist)

    total_mismatched = 0
    for dist, files in sorted(records.items()):
        mismatched = []
        for path in files:
            full = os.path.join(site_packages, path)
            if os.path.isfile(full) and _digest(full) != files[path]:
                mismatched.append((path, _digest(full)))
        total_mismatched += len(mismatched)
        print(f"  {dist}: {len(files)} files, {len(mismatched)} no longer match its RECORD")
        for path, now_hash in sorted(mismatched)[:5]:
            holder = owners.get((path, now_hash)) or ["(no cutlass wheel has this content)"]
            print(f"    {path}\n      now holds content from: {', '.join(holder)}")

    shared = sorted(p for p, dists in shipped_by.items() if len(dists) > 1)
    print(f"  paths shipped by more than one wheel: {len(shared)}")

    needle = "class DSLUserCodeRuntimeError"
    for root, _, names in os.walk(os.path.join(site_packages, "nvidia_cutlass_dsl")):
        for name in names:
            if not name.endswith(".py"):
                continue
            full = os.path.join(root, name)
            with open(full, errors="ignore") as fh:
                if needle in fh.read():
                    rel = os.path.relpath(full, site_packages)
                    print(f"  {needle.split()[-1]} defined in {rel} (shipped by: {', '.join(shipped_by.get(rel, ['?']))})")

    print("  ->", "HYBRID tree (files overwritten between wheels)" if total_mismatched else "consistent with every wheel's RECORD")
    return 1 if total_mismatched else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
