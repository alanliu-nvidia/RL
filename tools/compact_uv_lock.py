# SPDX-License-Identifier: Apache-2.0
"""Remove optional marker whitespace so the NM4 lock fits GitLab's file limit.

Run after ``uv lock``, then verify with ``uv lock --check``. Versions, URLs,
hashes, and marker operators are unchanged.
"""

import sys
import tomllib
from pathlib import Path


def main() -> None:
    """Compact the requested lockfile, defaulting to uv.lock."""
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "uv.lock")
    content = path.read_text()
    for operator in ("==", "!=", ">=", "<=", "<", ">"):
        content = content.replace(f" {operator} ", operator)
    tomllib.loads(content)
    path.write_text(content)
    print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
