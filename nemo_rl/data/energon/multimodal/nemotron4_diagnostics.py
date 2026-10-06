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

"""Optional per-process records of rejected NM4 samples."""

import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def record_sample_drop(stage: str, sample: Any, exception: Exception) -> None:
    directory = os.environ.get("NRL_NM4_SAMPLE_DROP_DIR")
    if not directory:
        return
    if isinstance(sample, dict):
        key = sample.get("__key__")
        subflavors = sample.get("__subflavors__", {})
        sources = sample.get("__sources__", [])
    else:
        key = getattr(sample, "sample_key", getattr(sample, "__key__", None))
        subflavors = getattr(sample, "__subflavors__", {})
        sources = getattr(sample, "__sources__", [])
    row = {
        "timestamp": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "stage": stage,
        "sample_key": key,
        "subflavors": subflavors,
        "sources": [str(source) for source in sources or []],
        "error_type": type(exception).__name__,
        "error": str(exception),
    }
    try:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / f"{row['host']}.{row['pid']}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
    except OSError:
        logger.exception("Could not record rejected NM4 sample %r", key)
