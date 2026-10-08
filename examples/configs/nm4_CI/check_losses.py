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

"""Check opt-in NM4 CI artifacts or W&B runs against suite.yaml loss gates."""

import argparse
import json
import math
from pathlib import Path

import yaml


def compare_losses(
    candidate: dict[int, float],
    reference: dict[int, float],
    *,
    steps: list[int],
    max_relative_loss_pct: float,
    strict_threshold: bool,
) -> dict[str, float]:
    """Reject missing/nonfinite losses or relative errors outside the chosen gate."""
    differences = []
    for step in steps:
        actual, expected = candidate[step], reference[step]
        if not math.isfinite(actual) or not math.isfinite(expected) or expected <= 0:
            raise ValueError(f"Invalid loss at step {step}: {actual=}, {expected=}")
        differences.append(100 * abs(actual - expected) / abs(expected))
    maximum = max(differences)
    failed = (
        maximum >= max_relative_loss_pct
        if strict_threshold
        else maximum > max_relative_loss_pct
    )
    if failed:
        raise ValueError(
            f"Maximum relative loss difference {maximum:.6f}% violates "
            f"{'<' if strict_threshold else '<='} {max_relative_loss_pct}%"
        )
    return {
        "max_relative_loss_pct": maximum,
        "mean_absolute_relative_loss_pct": sum(differences) / len(differences),
    }


def validate_metrics(
    metrics: dict[str, dict[str, float]], *, steps: list[int]
) -> dict[int, float]:
    """Require exact training steps and finite loss/gradient measurements."""
    for tag in ("loss", "grad_norm"):
        values = {int(step): float(value) for step, value in metrics[tag].items()}
        if sorted(values) != steps:
            raise ValueError(f"Expected {tag} steps {steps}; got {sorted(values)}.")
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Nonfinite {tag}.")
        if tag == "grad_norm" and any(value < 0 for value in values.values()):
            raise ValueError("Negative gradient norm.")
    return {int(step): float(value) for step, value in metrics["loss"].items()}


def read_local_result(
    directory: Path, *, case: str, steps: list[int]
) -> dict[int, float]:
    """Reject a failed/incomplete driver even if it left complete metric files."""
    result = json.loads((directory / "result.json").read_text())
    if (
        result["case"] != case
        or result["status"] != "passed"
        or result["exit_code"] != 0
    ):
        raise ValueError(f"Case did not finish successfully: {result}")
    if result["total_steps"] != steps[-1]:
        raise ValueError("Controller did not complete the expected optimizer steps.")
    return validate_metrics(
        json.loads((directory / "metrics.json").read_text()), steps=steps
    )


def main() -> None:
    """Download mapped W&B runs and exit nonzero if any required case/check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=Path(__file__).with_name("suite.yaml")
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--runs",
        type=Path,
        help="JSON mapping case IDs to entity/project/run_id",
    )
    sources.add_argument(
        "--artifacts",
        type=Path,
        help="Suite run directory containing one directory per case",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    mappings = None
    run_errors = (ValueError, KeyError, TypeError, OSError)
    if args.runs:
        # W&B is optional for offline CI artifact checks.
        import wandb

        mappings = json.loads(args.runs.read_text())
        if set(mappings) != set(suite["cases"]):
            raise ValueError(
                "The run map must contain exactly the case IDs in suite.yaml."
            )
        api = wandb.Api(timeout=60)
        run_errors += (wandb.errors.Error,)
    losses = {}
    results = {"runs": {}, "checks": {}, "errors": {}}
    for case, spec in suite["cases"].items():
        try:
            if args.artifacts:
                values = read_local_result(
                    args.artifacts / case, case=case, steps=spec["steps"]
                )
                losses[case] = values
                results["runs"][case] = {
                    "path": str(args.artifacts / case),
                    "losses": values,
                }
                continue
            run = api.run(mappings[case])
            if suite["require_finished"] and run.state != "finished":
                raise ValueError(f"Run state is {run.state!r}; expected 'finished'.")
            values = {}
            for row in run.scan_history(
                keys=[suite["step_metric"], suite["metric"]], page_size=1000
            ):
                step, value = row[suite["step_metric"]], row[suite["metric"]]
                if step is None or value is None:
                    continue
                step, value = int(step), float(value)
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite loss at step {step}: {value}")
                if step in values and values[step] != value:
                    raise ValueError(f"Conflicting loss records for step {step}.")
                values[step] = value
            if sorted(values) != spec["steps"]:
                raise ValueError(
                    f"Expected training steps {spec['steps']}; got {sorted(values)}."
                )
            losses[case] = values
            results["runs"][case] = {"path": mappings[case], "losses": values}
        except run_errors as error:
            results["errors"][case] = f"{type(error).__name__}: {error}"
    for check in suite["checks"]:
        try:
            if "reference_file" in check:
                saved = json.loads(
                    (args.suite.parent / check["reference_file"]).read_text()
                )
                reference = {int(s): float(v) for s, v in saved["losses"].items()}
            else:
                reference = losses[check["reference_case"]]
            results["checks"][check["name"]] = compare_losses(
                losses[check["candidate"]],
                reference,
                steps=check["steps"],
                max_relative_loss_pct=check["max_relative_loss_pct"],
                strict_threshold=check["strict_threshold"],
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            results["errors"][check["name"]] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if results["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
