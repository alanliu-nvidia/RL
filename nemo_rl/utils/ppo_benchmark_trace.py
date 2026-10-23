# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in benchmark spans around blocking driver RPCs (not GPU kernel spans)."""

import functools
import hashlib
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def benchmark_span(phase, instance, meta=None):
    directory = os.environ.get("NRL_PPO_BENCHMARK_TRACE_DIR")
    if not directory:
        yield
        return
    host, pid = socket.gethostname(), os.getpid()
    record = {
        "phase": phase,
        "host": host,
        "pid": pid,
        "tid": threading.get_ident(),
        "start_ns": time.monotonic_ns(),
        "wall_start_ns": time.time_ns(),
        "status": "ok",
    }
    if hasattr(instance, "_train_steps"):
        record["step"] = instance._train_steps + 1
    if meta is not None and hasattr(meta, "sample_ids"):
        record["samples"] = len(meta.sample_ids)
        record["sample_ids_sha256"] = hashlib.sha256(
            json.dumps(meta.sample_ids, sort_keys=True).encode()
        ).hexdigest()
    try:
        yield
    except BaseException as error:
        record["status"] = type(error).__name__
        raise
    finally:
        record["end_ns"] = time.monotonic_ns()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        # One O_APPEND write per complete span. Every file belongs to one process;
        # all comparisons proving concurrency use the same monotonic clock.
        payload = (json.dumps(record) + "\n").encode()
        fd = os.open(
            root / f"spans-{host}-{pid}.jsonl",
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            if os.write(fd, payload) != len(payload):
                raise OSError("Incomplete benchmark span write")
        finally:
            os.close(fd)


def trace_benchmark_rpc(phase):
    """Record only metadata and elapsed RPC time; never copy training tensors."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(self, *args, **kwargs):
            meta = kwargs.get("meta", args[0] if args else None)
            with benchmark_span(phase, self, meta):
                return function(self, *args, **kwargs)

        return wrapped

    return decorate


def record_initial_value_head(weight):
    """Hash only the newly initialized scalar head, before optimizer updates."""
    directory = os.environ.get("NRL_PPO_BENCHMARK_TRACE_DIR")
    if not directory:
        return
    import torch

    payload = weight.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    record = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": torch.distributed.get_rank(),
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "point": "fresh scalar-head construction before DDP/checkpoint load",
    }
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"initial-head-{record['host']}-{record['pid']}.json").write_text(
        json.dumps(record) + "\n"
    )
