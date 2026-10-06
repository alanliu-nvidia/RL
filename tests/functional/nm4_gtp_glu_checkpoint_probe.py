#!/usr/bin/env python3
"""Identify GLU gate/up quarter permutations in the full NM4 weight audit."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.run_dir / "rank-0.parity.json").read_text())
    matches = []
    reference_weight = None
    incorrect_weight = None
    for name in audit["weight_hash_mismatches"]:
        key = hashlib.sha256(name.encode()).hexdigest()
        original = torch.load(
            args.run_dir / "baseline" / "rank-0" / f"{key}.weight.pt", weights_only=True
        )
        quarters = original.chunk(4, dim=0)
        permuted = torch.cat(
            [quarters[0], quarters[2], quarters[1], quarters[3]], dim=0
        )
        checksum = hashlib.sha256(
            permuted.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        matches.append(
            {
                "name": name,
                "matches_gate_up_quarter_permutation": checksum
                == audit["parameters"][name]["weight_sha256"],
            }
        )
        if reference_weight is None:
            reference_weight = original
            incorrect_weight = permuted
    assert matches and all(
        item["matches_gate_up_quarter_permutation"] for item in matches
    )
    assert reference_weight.shape == (12288, 2048)
    torch.manual_seed(42)
    inputs = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16)
    targets = torch.randn(32, 6144, device="cuda", dtype=torch.bfloat16)
    results = []
    for weights in (reference_weight, incorrect_weight):
        parameter = weights.cuda().bfloat16().requires_grad_()
        gate, up = F.linear(inputs, parameter).chunk(2, dim=-1)
        output = F.silu(gate) * up
        loss = (output.float() - targets.float()).square().mean()
        loss.backward()
        results.append(
            {
                "loss": float(loss),
                "output": output.detach().float().cpu(),
                "gradient": parameter.grad.detach().float().cpu(),
            }
        )
    baseline, candidate = results
    output_error = float(
        (candidate["output"] - baseline["output"]).norm() / baseline["output"].norm()
    )
    grad_error = float(
        (candidate["gradient"] - baseline["gradient"]).norm()
        / baseline["gradient"].norm()
    )
    args.output.write_text(
        json.dumps(
            {
                "all_mismatches_explained_by_gate_up_permutation": True,
                "weight_audits": matches,
                "fc1_geometry": [12288, 2048],
                "reference_loss": baseline["loss"],
                "permuted_loss": candidate["loss"],
                "output_relative_l2": output_error,
                "gradient_relative_l2": grad_error,
                "scope": "Actual audited vision FC1 checkpoint weights; isolated SwiGLU algebra, no optimizer updates",
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output.read_text(), flush=True)


if __name__ == "__main__":
    main()
