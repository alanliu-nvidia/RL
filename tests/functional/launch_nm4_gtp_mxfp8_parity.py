#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Launch the standalone parity test inside an existing four-node rem allocation."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["both", "gdp", "attention"], default="both")
    parser.add_argument(
        "--test-file",
        choices=["nm4_full_gtp_mxfp8_parity.py", "nm4_gtp_mxfp8_parity.py"],
        default="nm4_full_gtp_mxfp8_parity.py",
    )
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--reuse-baseline", action="store_true")
    args = parser.parse_args()
    mount_config = Path.home() / ".cache/nemorl/last-mounts.env"
    values = dict(
        line.split("=", 1)
        for line in mount_config.read_text().splitlines()
        if "=" in line
    )
    job = json.loads(
        subprocess.check_output(
            ["scontrol", "show", "job", args.job_id, "--json"], text=True
        )
    )["jobs"][0]
    nodes = sorted(
        subprocess.check_output(
            ["scontrol", "show", "hostnames", job["nodes"]], text=True
        ).split()
    )
    if len(nodes) != 4:
        raise ValueError(f"Expected four allocated nodes, got {nodes}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    container_output = "/data/" + str(args.output_dir.relative_to(Path.home() / "data"))
    rank_mapping = " ".join(
        f"{shlex.quote(node)}) test_node_rank={index} ;;"
        for index, node in enumerate(nodes)
    )
    command = (
        f'case "$SLURMD_NODENAME" in {rank_mapping} *) exit 2 ;; esac; '
        "cd /opt/nemo-rl && "
        "UV_PROJECT_ENVIRONMENT=/opt/ray_venvs/nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker "
        "uv run --no-sync python -m torch.distributed.run "
        '--nnodes=4 --nproc-per-node=4 --node-rank="$test_node_rank" '
        f"--master-addr={shlex.quote(nodes[0])} --master-port=29641 "
        f"tests/functional/{args.test_file} "
        f"--output {shlex.quote(container_output + '/parity.json')}"
    )
    if args.test_file == "nm4_gtp_mxfp8_parity.py":
        command += f" --kind {args.kind}"
    if args.baseline_dir is not None:
        baseline_path = "/data/" + str(
            args.baseline_dir.relative_to(Path.home() / "data")
        )
        command += f" --baseline-dir {shlex.quote(baseline_path)}"
    if args.reuse_baseline:
        command += " --reuse-baseline"
    invocation = [
        "srun",
        "--jobid",
        args.job_id,
        "--overlap",
        "--nodes=4",
        "--ntasks=4",
        "--ntasks-per-node=1",
        "--gres=gpu:4",
        "--kill-on-bad-exit=1",
        "--no-container-mount-home",
        "--container-image=" + values["CONTAINER"],
        "--container-mounts=" + values["MOUNTS"],
        "bash",
        "-c",
        command,
    ]
    (args.output_dir / "launch.json").write_text(
        json.dumps(
            {"job_id": args.job_id, "nodes": nodes, "command": command}, indent=2
        )
        + "\n"
    )
    with (args.output_dir / "test.log").open("w") as logfile:
        result = subprocess.run(
            invocation, stdout=logfile, stderr=subprocess.STDOUT, check=False
        )
    (args.output_dir / "exit-code.txt").write_text(str(result.returncode) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
