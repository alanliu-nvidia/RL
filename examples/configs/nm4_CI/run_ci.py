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

"""Plan or run explicitly selected NM4 cases in an existing Ray allocation."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
from pathlib import Path

from check_losses import read_local_result, validate_metrics
from omegaconf import DictConfig, OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parents[2]


def compose_case(
    case: str, *, suite: DictConfig, run_dir: Path, overrides: list[str]
) -> DictConfig:
    """Use training's config loader, keeping per-case artifacts and resume paths separate."""
    register_omegaconf_resolvers()
    config = load_config(SUITE_DIR / suite.cases[case].config)
    config = OmegaConf.merge(config, OmegaConf.load(SUITE_DIR / "ci_runtime.yaml"))
    config = parse_hydra_overrides(config, overrides)
    # Artifact locations are owned by the runner so partial/old histories cannot pass.
    logs = str(run_dir / case / "logs")
    config.logger.log_dir = logs
    config.logger.tensorboard.log_dir = logs
    checkpoint_name = (
        "checkpoint-16n-split"
        if case in ("checkpoint_16n_save10", "checkpoint_16n_resume20")
        else case
    )
    config.checkpointing.checkpoint_dir = str(run_dir / "checkpoints" / checkpoint_name)
    if config.sft.max_num_steps != suite.cases[case].steps[-1]:
        raise ValueError(
            "Changing max_num_steps would invalidate the suite's expected steps."
        )
    if config.sft.max_consecutive_step_failures != 0:
        raise ValueError("CI must fail on a failed training step.")
    OmegaConf.resolve(config)
    return config


def require_allocation(config: DictConfig) -> None:
    """Fail instead of silently skipping a case on an undersized/mismatched cluster."""
    # Ray is unnecessary for CPU planning and artifact checking.
    import ray

    ray.init(address="auto", log_to_driver=False)
    try:
        gpu_nodes = [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) > 0
        ]
        expected = config.cluster
        if len(gpu_nodes) != expected.num_nodes or any(
            node["Resources"]["GPU"] != expected.gpus_per_node for node in gpu_nodes
        ):
            raise ValueError(
                f"Expected {expected.num_nodes} nodes × {expected.gpus_per_node} GPUs; "
                f"Ray reports {[node['Resources']['GPU'] for node in gpu_nodes]}."
            )
    finally:
        ray.shutdown()


def resume_environment(
    run_dir: Path, *, case: str, config: DictConfig
) -> dict[str, str]:
    """Require the completed save phase; the training preflight verifies optimizer state."""
    environment = os.environ.copy()
    # An inherited resume request must not contaminate a fresh reference case.
    environment.pop("NEMO_RL_REQUIRE_RESUME", None)
    environment.pop("NEMO_RL_EXPECTED_RESUME_CHECKPOINT", None)
    if case == "checkpoint_16n_resume20":
        read_local_result(
            run_dir / "checkpoint_16n_save10",
            case="checkpoint_16n_save10",
            steps=list(range(1, 11)),
        )
        checkpoint = Path(config.checkpointing.checkpoint_dir) / "step_10"
        for relative in ("training_info.json", "sft_v2_loaders.pt", "policy/weights"):
            if not (checkpoint / relative).exists():
                raise ValueError(
                    f"Missing finalized resume artifact: {checkpoint / relative}"
                )
        environment["NEMO_RL_REQUIRE_RESUME"] = "1"
        environment["NEMO_RL_EXPECTED_RESUME_CHECKPOINT"] = str(checkpoint)
    elif Path(config.checkpointing.checkpoint_dir).exists():
        raise ValueError(
            "Fresh CI cases must not reuse an existing checkpoint directory."
        )
    return environment


def run_case(
    case: str,
    *,
    suite: DictConfig,
    config: DictConfig,
    run_dir: Path,
    timeout_seconds: int,
) -> None:
    """Run one fresh driver and write success only after controller and metric checks."""
    directory = run_dir / case
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / "resolved_config.yaml"
    OmegaConf.save(config, config_path)
    command = [
        "uv",
        "run",
        "--no-sync",
        "examples/run_sft_v2.py",
        "--config",
        str(config_path),
    ]
    result = {
        "case": case,
        "status": "failed",
        "exit_code": None,
        "total_steps": None,
        "command": command,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "error": None,
    }
    try:
        require_allocation(config)
        environment = resume_environment(run_dir, case=case, config=config)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTORCH_CUDA_ALLOC_CONF"] = (
            config.policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF
        )
        with (directory / "driver.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                result["exit_code"] = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise ValueError(
                    f"Driver timed out after {timeout_seconds}s."
                ) from None
        if result["exit_code"] != 0:
            raise ValueError(
                f"Driver failed with exit code {result['exit_code']}; see driver.log."
            )
        completed = re.findall(
            r"'total_steps':\s*(\d+)", (directory / "driver.log").read_text()
        )
        result["total_steps"] = int(completed[-1]) if completed else None
        if result["total_steps"] != suite.cases[case].steps[-1]:
            raise ValueError("Missing expected controller completion result.")
        with (directory / "metrics_export.log").open("w") as log:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "tests/json_dump_tb_logs.py",
                    str(directory / "logs"),
                    "--output_path",
                    str(directory / "metrics.json"),
                    "--error-on-conflicts",
                ],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        validate_metrics(
            json.loads((directory / "metrics.json").read_text()),
            steps=list(suite.cases[case].steps),
        )
        result["status"] = "passed"
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    """Print the allocation plan or execute one opt-in case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument(
        "--case", choices=list(OmegaConf.load(SUITE_DIR / "suite.yaml").cases)
    )
    parser.add_argument(
        "--run-tag", required=True, help="Fresh tag shared by all cases"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Shared path visible inside all nodes",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override; repeat as needed",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error(
            "Run tag must contain only letters, digits, underscores or hyphens."
        )
    if args.timeout_seconds <= 0:
        parser.error("Timeout must be positive.")
    suite = OmegaConf.load(SUITE_DIR / "suite.yaml")
    os.environ[suite.run_tag_env] = args.run_tag
    run_dir = args.artifacts.resolve() / args.run_tag
    if args.action == "run":
        if not args.case:
            parser.error("run requires --case; this suite never executes by default.")
        config = compose_case(
            args.case, suite=suite, run_dir=run_dir, overrides=args.override
        )
        run_case(
            args.case,
            suite=suite,
            config=config,
            run_dir=run_dir,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        plan = []
        for case in [args.case] if args.case else suite.cases:
            config = compose_case(
                case, suite=suite, run_dir=run_dir, overrides=args.override
            )
            plan.append(
                {
                    "case": case,
                    "nodes": config.cluster.num_nodes,
                    "gpus_per_node": config.cluster.gpus_per_node,
                    "segment_size": config.cluster.segment_size,
                    "steps": list(suite.cases[case].steps),
                    "depends_on": "checkpoint_16n_save10"
                    if case == "checkpoint_16n_resume20"
                    else None,
                    "sequence_length": config.policy.max_total_sequence_length,
                    "checkpoint": config.checkpointing.pretrained_checkpoint.path,
                }
            )
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
