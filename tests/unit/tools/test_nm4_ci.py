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

"""CPU checks for the opt-in NM4 runner's failure and parity gates."""

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

SUITE_DIR = Path(__file__).resolve().parents[3] / "examples/configs/nm4_CI"
sys.path.insert(0, str(SUITE_DIR))
import check_losses  # noqa: E402
import run_ci  # noqa: E402


@pytest.fixture
def suite(monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    monkeypatch.setenv("NM4_CI_RUN_TAG", "unit-test")
    return OmegaConf.load(SUITE_DIR / "suite.yaml")


def write_result(directory: Path, *, case: str, steps: list[int]) -> None:
    """Create a complete local result for artifact-validation tests."""
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(
        json.dumps(
            {"case": case, "status": "passed", "exit_code": 0, "total_steps": steps[-1]}
        )
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "loss": {str(step): 1.0 for step in steps},
                "grad_norm": {str(step): 2.0 for step in steps},
            }
        )
    )


def test_case_geometry_and_resume_paths(suite: DictConfig, tmp_path: Path) -> None:
    configs = {
        case: run_ci.compose_case(case, suite=suite, run_dir=tmp_path, overrides=[])
        for case in suite.cases
    }
    assert len(configs) == 14
    for case, config in configs.items():
        expected_nodes = 32 if "32n" in case else 16
        if case.startswith("clevr_1n_"):
            expected_nodes = 1
        elif case.startswith("clevr_4n_"):
            expected_nodes = 4
        assert config.cluster.num_nodes == expected_nodes
        assert config.cluster.gpus_per_node == 4
        assert not config.logger.wandb_enabled
        assert config.logger.tensorboard_enabled
        assert config.policy.megatron_cfg.attention_backend == "auto"
        assert "saved_args" not in (config.policy.megatron_cfg.model_overrides or {})
        assert config.sft.max_num_steps == suite.cases[case].steps[-1]
    bf16 = configs["generalist_16n_bf16_hybridep"].policy.megatron_cfg
    assert not bf16.fp8_cfg.enabled
    assert bf16.model_overrides.fp8 is None
    assert bf16.model_overrides.fp8_recipe is None
    assert not bf16.model_overrides.fp8_param
    assert (
        configs[
            "generalist_32n_mxfp8_gtp2"
        ].policy.megatron_cfg.model_overrides.gtp_weight_remat_size
        == 2
    )
    assert (
        configs[
            "generalist_32n_mxfp8_egtp2"
        ].policy.megatron_cfg.model_overrides.expert_gtp_weight_remat_size
        == 2
    )
    for nodes, cp in ((1, 1), (4, 4), (16, 16)):
        config = configs[f"clevr_{nodes}n_mxfp8_cp{cp}_2k"]
        assert config.policy.max_total_sequence_length == 2048
        assert config.policy.megatron_cfg.context_parallel_size == cp
        assert config.policy.train_global_batch_size == 4
        assert config.cluster.segment_size == nodes
    assert (
        configs["checkpoint_16n_save10"].checkpointing.checkpoint_dir
        == configs["checkpoint_16n_resume20"].checkpointing.checkpoint_dir
    )
    assert (
        configs["checkpoint_16n_uninterrupted"].checkpointing.checkpoint_dir
        != configs["checkpoint_16n_save10"].checkpointing.checkpoint_dir
    )


@pytest.mark.parametrize(
    "override", ["sft.max_num_steps=1", "sft.max_consecutive_step_failures=2"]
)
def test_invalid_step_overrides(
    suite: DictConfig, tmp_path: Path, override: str
) -> None:
    with pytest.raises(ValueError):
        run_ci.compose_case(
            "generalist_16n_mxfp8_hybridep",
            suite=suite,
            run_dir=tmp_path,
            overrides=[override],
        )


@pytest.mark.parametrize(
    "strict,limit,value,passes",
    [
        (True, 0.5, 100.49, True),
        (True, 0.5, 100.5, False),
        (False, 10.0, 110.0, True),
        (False, 10.0, 110.01, False),
    ],
)
def test_parity_boundaries(
    strict: bool, limit: float, value: float, passes: bool
) -> None:
    arguments = dict(steps=[1], max_relative_loss_pct=limit, strict_threshold=strict)
    if passes:
        check_losses.compare_losses({1: value}, {1: 100.0}, **arguments)
    else:
        with pytest.raises(ValueError):
            check_losses.compare_losses({1: value}, {1: 100.0}, **arguments)


@pytest.mark.parametrize(
    "tag,values",
    [
        ("loss", {"1": 1.0}),
        ("loss", {"1": 1.0, "2": float("nan")}),
        ("grad_norm", {"1": 1.0, "2": float("inf")}),
        ("grad_norm", {"1": 1.0, "2": -1.0}),
        ("loss", {"1": 1.0, "2": 1.0, "3": 1.0}),
    ],
)
def test_reject_invalid_metrics(tag: str, values: dict[str, float]) -> None:
    metrics = {"loss": {"1": 1.0, "2": 1.0}, "grad_norm": {"1": 2.0, "2": 2.0}}
    metrics[tag] = values
    with pytest.raises(ValueError):
        check_losses.validate_metrics(metrics, steps=[1, 2])


@pytest.mark.parametrize(
    "field,value",
    [("exit_code", 1), ("total_steps", 1), ("status", "failed"), ("case", "other")],
)
def test_complete_metrics_cannot_hide_failed_driver(
    tmp_path: Path, field: str, value: int | str
) -> None:
    directory = tmp_path / "case"
    write_result(directory, case="case", steps=[1, 2])
    result = json.loads((directory / "result.json").read_text())
    result[field] = value
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        check_losses.read_local_result(directory, case="case", steps=[1, 2])


def test_resume_rejects_missing_save_phase_and_incomplete_checkpoint(
    suite: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = run_ci.compose_case(
        "checkpoint_16n_resume20", suite=suite, run_dir=tmp_path, overrides=[]
    )
    with pytest.raises(FileNotFoundError):
        run_ci.resume_environment(
            tmp_path, case="checkpoint_16n_resume20", config=config
        )
    write_result(
        tmp_path / "checkpoint_16n_save10",
        case="checkpoint_16n_save10",
        steps=list(range(1, 11)),
    )
    with pytest.raises(ValueError, match="Missing finalized"):
        run_ci.resume_environment(
            tmp_path, case="checkpoint_16n_resume20", config=config
        )
    checkpoint = Path(config.checkpointing.checkpoint_dir) / "step_10"
    (checkpoint / "policy/weights").mkdir(parents=True)
    (checkpoint / "training_info.json").write_text("{}")
    (checkpoint / "sft_v2_loaders.pt").write_bytes(b"fixture")
    env = run_ci.resume_environment(
        tmp_path, case="checkpoint_16n_resume20", config=config
    )
    assert env["NEMO_RL_REQUIRE_RESUME"] == "1"
    assert env["NEMO_RL_EXPECTED_RESUME_CHECKPOINT"] == str(checkpoint)
    monkeypatch.setenv("NEMO_RL_REQUIRE_RESUME", "1")
    monkeypatch.setenv("NEMO_RL_EXPECTED_RESUME_CHECKPOINT", "wrong")
    fresh = run_ci.compose_case(
        "generalist_16n_mxfp8_hybridep", suite=suite, run_dir=tmp_path, overrides=[]
    )
    assert "NEMO_RL_REQUIRE_RESUME" not in run_ci.resume_environment(
        tmp_path, case="generalist_16n_mxfp8_hybridep", config=fresh
    )


def test_allocation_failure_writes_failed_result(
    suite: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real run must reject undersized Ray resources, not return a successful skip.
    fake_ray = types.SimpleNamespace(
        init=lambda **kw: None,
        shutdown=lambda: None,
        nodes=lambda: [{"Alive": True, "Resources": {"GPU": 4}}],
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    case = "generalist_16n_mxfp8_hybridep"
    config = run_ci.compose_case(case, suite=suite, run_dir=tmp_path, overrides=[])
    with pytest.raises(ValueError, match="Expected 16 nodes"):
        run_ci.run_case(
            case, suite=suite, config=config, run_dir=tmp_path, timeout_seconds=10
        )
    result = json.loads((tmp_path / case / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["exit_code"] is None
    assert "--no-sync" in result["command"]
    with pytest.raises(FileExistsError):
        run_ci.run_case(
            case, suite=suite, config=config, run_dir=tmp_path, timeout_seconds=10
        )


def test_offline_checker_cli(tmp_path: Path) -> None:
    for case in ("reference", "candidate"):
        write_result(tmp_path / case, case=case, steps=[1, 2])
    manifest = {
        "cases": {case: {"steps": [1, 2]} for case in ("reference", "candidate")},
        "checks": [
            {
                "name": "parity",
                "candidate": "candidate",
                "reference_case": "reference",
                "steps": [1, 2],
                "max_relative_loss_pct": 0.5,
                "strict_threshold": True,
            }
        ],
    }
    path = tmp_path / "suite.yaml"
    OmegaConf.save(OmegaConf.create(manifest), path)
    command = [
        sys.executable,
        str(SUITE_DIR / "check_losses.py"),
        "--suite",
        str(path),
        "--artifacts",
        str(tmp_path),
        "--output",
        str(tmp_path / "summary.json"),
    ]
    assert subprocess.run(command, capture_output=True).returncode == 0
    metrics = json.loads((tmp_path / "candidate/metrics.json").read_text())
    metrics["loss"]["2"] = 1.02
    (tmp_path / "candidate/metrics.json").write_text(json.dumps(metrics))
    assert subprocess.run(command, capture_output=True).returncode == 1
    assert "parity" in json.loads((tmp_path / "summary.json").read_text())["errors"]


@pytest.mark.parametrize(
    "failure",
    [None, "driver", "completion", "export", "nan", "missing_steps", "timeout"],
)
def test_driver_and_export_gates(
    suite: DictConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    """Exercise execution and artifact writing with fake training, without GPUs."""
    case = "generalist_16n_mxfp8_hybridep"
    config = run_ci.compose_case(case, suite=suite, run_dir=tmp_path, overrides=[])
    monkeypatch.setattr(run_ci, "require_allocation", lambda config: None)
    terminated = []
    monkeypatch.setattr(run_ci.os, "killpg", lambda pid, sig: terminated.append(pid))

    class FakeDriver:
        pid = 123456

        def __init__(self, command: list[str], **kwargs: object) -> None:
            assert command[:4] == ["uv", "run", "--no-sync", "examples/run_sft_v2.py"]
            kwargs["stdout"].write(
                "{}" if failure == "completion" else "{'total_steps': 20}"
            )
            self.waits = 0

        def wait(self, *, timeout: int) -> int:
            self.waits += 1
            if failure == "timeout" and self.waits == 1:
                raise subprocess.TimeoutExpired("fake driver", timeout)
            return 1 if failure == "driver" else 0

    def export(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert "--error-on-conflicts" in command
        if failure == "export":
            raise subprocess.CalledProcessError(1, command)
        steps = range(1, 20 if failure == "missing_steps" else 21)
        metrics = {
            "loss": {str(step): 1.0 for step in steps},
            "grad_norm": {str(step): 2.0 for step in steps},
        }
        if failure == "nan":
            metrics["grad_norm"]["20"] = float("nan")
        (tmp_path / case / "metrics.json").write_text(json.dumps(metrics))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_ci.subprocess, "Popen", FakeDriver)
    monkeypatch.setattr(run_ci.subprocess, "run", export)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            run_ci.run_case(
                case, suite=suite, config=config, run_dir=tmp_path, timeout_seconds=10
            )
    else:
        run_ci.run_case(
            case, suite=suite, config=config, run_dir=tmp_path, timeout_seconds=10
        )
        assert check_losses.read_local_result(
            tmp_path / case, case=case, steps=list(range(1, 21))
        )
    result = json.loads((tmp_path / case / "result.json").read_text())
    assert result["status"] == ("failed" if failure else "passed")
    assert bool(result["error"]) == bool(failure)
    if failure == "timeout":
        assert terminated == [FakeDriver.pid]
