#!/usr/bin/env python3
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

"""Count recorded NM4 sample rejections without Ray log deduplication."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts: Counter[tuple[str, str, str]] = Counter()
    reasons: dict[tuple[str, str, str], str] = {}
    total = 0
    for path in sorted(args.directory.glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                # A monitor can observe an unfinished final append.
                if not line.endswith("\n"):
                    continue
                row = json.loads(line)
                key = (
                    row["stage"],
                    row["subflavors"].get("cook", "unknown"),
                    row["error_type"],
                )
                counts[key] += 1
                reasons.setdefault(key, row["error"][:1000])
                total += 1
    summary = {
        "drop_attempts": total,
        "groups": [
            {
                "stage": key[0],
                "cooker": key[1],
                "error_type": key[2],
                "count": count,
                "example_reason": reasons[key],
            }
            for key, count in counts.most_common()
        ],
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
