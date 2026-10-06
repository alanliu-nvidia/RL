#!/usr/bin/env python3
"""Frozen-input GTP1/GTP2 MXFP8 forward/gradient parity; standalone torchrun.

Run on four nodes with four ranks per node and --cp-size 4. Each CP group
consumes one of four distinct mock sequences. GTP changes weight storage,
not the logical data lanes. No optimizer is constructed or stepped.

Covers actual MCore GDP (contiguous chunkwise CP, padded in-projection) and
attention (zigzag CP), native MXFP8 weights, Bridge GTP configuration and
classification, DDP gradient reduction and replicated-GTP gradient finalize.
This reduced layer test does not cover NM4's provider/LM head, MTP, MoE,
HybridEP, data loader, or distributed optimizer parameter gathering.
"""

import argparse
import faulthandler
import gc
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.bridge.training.gtp import classify_gtp_remat_chains, configure_gtp_remat
from megatron.core import parallel_state as ps
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_replicated_grads_over_gtp_remat_group,
)
from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.generalized_tensor_parallelism import reset_gtp_state
from megatron.core.tensor_parallel.gtp_api import (
    gtp_native_fp8_load_context,
    is_gtp_param,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import fp8_autocast, fp8_model_init
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


def make_config(args, *, gtp_size):
    return TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=512,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        bias_dropout_fusion=False,
        normalization="RMSNorm",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=args.cp_size,
        gtp_weight_remat_size=gtp_size,
        expert_gtp_weight_remat_size=1,
        calculate_per_token_loss=True,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        mamba_num_heads=8,
        mamba_head_dim=32,
        mamba_num_groups=2,
        mamba_state_dim=128,
        gdp_num_householder=3,
        gdp_cutedsl_kernel=False,
        linear_cp_mode="chunkwise",
        linear_cp_layout="contiguous",
        attention_cp_layout="zigzag",
        cp_comm_type="p2p",
    )


def make_layer(config, *, kind):
    groups = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "cp", "gtp_remat"]
    )
    spec = (
        gated_delta_product_stack_spec.submodules.mamba_layer
        if kind == "gdp"
        else get_gpt_layer_with_transformer_engine_spec()
    )
    return build_module(
        spec, config=config, layer_number=1, pg_collection=groups
    ).cuda()


def setup_parallel(args, *, gtp_size):
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=args.cp_size,
        gtp_remat_size=gtp_size,
    )
    model_parallel_cuda_manual_seed(42)


def teardown_parallel():
    dist.barrier()
    ps.destroy_model_parallel()
    reset_gtp_state()
    FP8GlobalStateManager.reset()
    gc.collect()
    torch.cuda.empty_cache()


def mock_batch(args, *, kind, trial):
    gtp_size = ps.get_gtp_weight_remat_group().size()
    lane = ps.get_data_parallel_rank(with_gtp_remat=False) * gtp_size
    lane += ps.get_gtp_weight_remat_group().rank()
    cp_rank = ps.get_context_parallel_rank()
    generator = torch.Generator(device="cpu").manual_seed(1000 + 100 * trial + lane)
    full_x = torch.randn(args.sequence_length, 1, 256, generator=generator).bfloat16()
    full_target = torch.randn(
        args.sequence_length, 1, 256, generator=generator
    ).bfloat16()
    # Unequal token counts test global-token normalization, rather than mean-of-means.
    full_mask = (torch.arange(args.sequence_length) % (lane + 2) != 0).float()
    if kind == "gdp":
        positions = torch.arange(args.sequence_length).chunk(args.cp_size)[cp_rank]
    else:
        chunks = torch.arange(args.sequence_length).chunk(2 * args.cp_size)
        positions = torch.cat([chunks[cp_rank], chunks[2 * args.cp_size - cp_rank - 1]])
    batch = (full_x[positions], full_target[positions], full_mask[positions])
    digest = hashlib.sha256()
    for value in batch:
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return tuple(value.cuda() for value in batch), digest.hexdigest(), lane


def full_gradients(layer, canonical):
    gradients = {}
    for name, param in layer.named_parameters():
        gradient = param.main_grad.detach().float()
        if is_gtp_param(param):
            group = ps.get_gtp_weight_remat_group()
            shards = [torch.empty_like(gradient) for _ in range(group.size())]
            dist.all_gather(shards, gradient.contiguous(), group=group)
            gradient = torch.cat(shards, dim=0)[: canonical[name].shape[0]]
        assert gradient.shape == canonical[name].shape, (name, gradient.shape)
        gradients[name] = gradient.cpu().clone()
    return gradients


def run_phase(args, *, kind, gtp_size, canonical):
    setup_parallel(args, gtp_size=gtp_size)
    config = make_config(args, gtp_size=gtp_size)
    configure_gtp_remat(config)
    recipe = MXFP8BlockScaling()
    with fp8_model_init(enabled=True, recipe=recipe):
        layer = make_layer(config, kind=kind)
    shard_rank = ps.get_gtp_weight_remat_group().rank()
    with torch.no_grad(), gtp_native_fp8_load_context(layer):
        for name, param in layer.named_parameters():
            value = canonical[name].cuda()
            if is_gtp_param(param):
                padded_rows = param.shape[0] * gtp_size
                if padded_rows > value.shape[0]:
                    value = torch.cat(
                        [
                            value,
                            value.new_zeros(
                                (padded_rows - value.shape[0], *value.shape[1:])
                            ),
                        ],
                        dim=0,
                    )
                value = value.narrow(0, shard_rank * param.shape[0], param.shape[0])
            param.copy_(value)
    native_count = sum(is_mxfp8tensor(param) for param in layer.parameters())
    sharded_count = sum(is_gtp_param(param) for param in layer.parameters())
    assert native_count > 0, "Test must execute native MXFP8 weights"
    assert (sharded_count > 0) == (gtp_size > 1), "GTP must actually shard weights"
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            grad_reduce_in_fp32=True,
        ),
        layer,
    )
    classify_gtp_remat_chains([ddp], config)
    results = []
    for trial in range(args.trials):
        if dist.get_rank() == 0:
            print(f"{kind} GTP={gtp_size} trial={trial} starting forward", flush=True)
        ddp.zero_grad_buffer()
        (inputs, targets, mask), digest, lane = mock_batch(args, kind=kind, trial=trial)
        token_count = mask.sum()
        dist.all_reduce(token_count)
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            output = ddp(inputs, attention_mask=None)
            if isinstance(output, tuple):
                output = output[0]
            local_sum = (
                (output.float() - targets.float()).square().mean(dim=-1).squeeze(-1)
                * mask
            ).sum()
            backward_loss = local_sum / token_count
        if dist.get_rank() == 0:
            print(f"{kind} GTP={gtp_size} trial={trial} starting backward", flush=True)
        backward_loss.backward()
        if dist.get_rank() == 0:
            print(f"{kind} GTP={gtp_size} trial={trial} syncing gradients", flush=True)
        ddp.finish_grad_sync()
        _allreduce_replicated_grads_over_gtp_remat_group(
            [ddp],
            ps.get_gtp_weight_remat_group(),
            ps.get_expert_gtp_weight_remat_group(check_initialized=False),
            calculate_per_token_loss=True,
        )
        global_sum = local_sum.detach().clone()
        dist.all_reduce(global_sum)
        results.append(
            {
                "loss": float(global_sum / token_count),
                "valid_tokens": int(token_count),
                "input_sha256": digest,
                "lane": lane,
                "output": output.detach().float().cpu().clone(),
                "gradients": full_gradients(layer, canonical),
            }
        )
        if dist.get_rank() == 0:
            print(
                f"{kind} GTP={gtp_size} trial={trial} loss={results[-1]['loss']:.8f}",
                flush=True,
            )
    del ddp, layer
    teardown_parallel()
    return results


def compare(args, *, kind, baseline, sharded):
    reports = []
    for trial, (reference, candidate) in enumerate(zip(baseline, sharded, strict=True)):
        assert reference["input_sha256"] == candidate["input_sha256"], (
            "Inputs changed across GTP"
        )
        assert reference["lane"] == candidate["lane"]
        assert reference["valid_tokens"] == candidate["valid_tokens"]
        loss_relative = abs(candidate["loss"] - reference["loss"]) / abs(
            reference["loss"]
        )
        output_relative = float(
            (candidate["output"] - reference["output"]).norm()
            / reference["output"].norm().clamp_min(1e-12)
        )
        parameters = {}
        total_norm_squared = 0.0
        for name, base_gradient in reference["gradients"].items():
            gradient = candidate["gradients"][name]
            difference = gradient - base_gradient
            norm = float(base_gradient.norm())
            relative_l2 = float(difference.norm()) / max(norm, 1e-12)
            ratio = float(gradient.norm()) / max(norm, 1e-12)
            finite = bool(
                torch.isfinite(gradient).all() and torch.isfinite(base_gradient).all()
            )
            passed = finite and (relative_l2 <= args.grad_rtol)
            total_norm_squared += norm * norm
            parameters[name] = {
                "relative_l2": relative_l2,
                "norm_ratio": ratio,
                "max_absolute_error": float(difference.abs().max()),
                "passed": passed,
            }
        passed = (
            loss_relative <= args.loss_rtol
            and output_relative <= args.output_rtol
            and total_norm_squared > 0
            and all(item["passed"] for item in parameters.values())
        )
        reports.append(
            {
                "kind": kind,
                "trial": trial,
                "passed": passed,
                "baseline_loss": reference["loss"],
                "gtp2_loss": candidate["loss"],
                "loss_relative": loss_relative,
                "output_relative_l2": output_relative,
                "valid_tokens": reference["valid_tokens"],
                "parameters": parameters,
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["gdp", "attention", "both"], default="both")
    parser.add_argument("--cp-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--loss-rtol", type=float, default=1e-3)
    parser.add_argument("--output-rtol", type=float, default=1e-2)
    parser.add_argument("--grad-rtol", type=float, default=2e-2)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_file = (args.output.parent / f"rank-{os.environ['RANK']}.stack.log").open("w")
    faulthandler.dump_traceback_later(120, repeat=True, file=trace_file)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    assert dist.get_world_size() == 16, "Launch four nodes with four GPUs each"
    assert args.sequence_length % (2 * args.cp_size * 32) == 0
    rank = dist.get_rank()
    all_reports = []
    for kind in ["gdp", "attention"] if args.kind == "both" else [args.kind]:
        setup_parallel(args, gtp_size=1)
        with fp8_model_init(enabled=False):
            unquantized = make_layer(make_config(args, gtp_size=1), kind=kind)
        canonical = {}
        for name, param in unquantized.named_parameters():
            dist.broadcast(param.data, src=0)
            canonical[name] = param.detach().cpu().clone()
        del unquantized
        teardown_parallel()
        baseline = run_phase(args, kind=kind, gtp_size=1, canonical=canonical)
        sharded = run_phase(args, kind=kind, gtp_size=2, canonical=canonical)
        all_reports.extend(compare(args, kind=kind, baseline=baseline, sharded=sharded))
    rank_reports = [None] * dist.get_world_size()
    dist.all_gather_object(rank_reports, all_reports)
    passed = all(item["passed"] for reports in rank_reports for item in reports)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "passed": passed,
                    "world_size": 16,
                    "cp_size": args.cp_size,
                    "optimizer_updates": 0,
                    "precision": "native MXFP8 weights and MXFP8 compute",
                    "rank_reports": rank_reports,
                    "tolerances": {
                        "loss": args.loss_rtol,
                        "output": args.output_rtol,
                        "gradient": args.grad_rtol,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"PARITY {'PASS' if passed else 'FAIL'}: {args.output}", flush=True)
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    trace_file.close()
    if not passed:
        raise AssertionError("MXFP8 GTP1/GTP2 parity failed; inspect JSON report")


if __name__ == "__main__":
    main()
