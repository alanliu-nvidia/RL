#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Standalone full-architecture NM4 checkpoint GTP1/GTP2 parity on 16 GPUs.

Uses the production NeMo-RL provider/configuration path and original checkpoint.
Changes only runtime topology (CP8/EP8), logical global batch two, and sequence
length. All 72 language layers, 512 experts, MTP, vision modules, dimensions,
and native MXFP8 precision policy are retained. Text-only mock input exercises
the language path and MTP; vision architecture is present but receives no media.
No optimizer is constructed or stepped. Full reduced gradients are compared
one parameter at a time, using disk for the baseline to bound host memory.

From the project checkout on the login node, launch on an existing allocation:
    python3 tests/functional/launch_nm4_gtp_mxfp8_parity.py \
        --job-id JOB_ID --output-dir /home/USER/data/tmp/nm4-gtp-parity

The launcher runs this file with uv run --no-sync and torchrun on four nodes.
Reports contain per-rank weight hashes and complete gradient error metrics.
Baseline tensors are large and are saved outside the checkout. Prefer paired
phases; separate-process baseline reuse has not established gradient parity.
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


def digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def make_config(args, *, gtp_size):
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.sft_v2 import MasterConfig
    from nemo_rl.models.megatron.setup import setup_model_config
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(args.recipe), resolve=True)
    resolved["cluster"]["num_nodes"] = 4
    resolved["cluster"]["segment_size"] = 2
    policy = resolved["policy"]
    policy["train_global_batch_size"] = 16 // args.cp_size
    parallel = policy["megatron_cfg"]
    parallel["context_parallel_size"] = args.cp_size
    parallel["expert_model_parallel_size"] = args.ep_size
    parallel["model_overrides"]["gtp_weight_remat_size"] = gtp_size
    parallel["model_overrides"]["expert_gtp_weight_remat_size"] = 1
    # Allocation 977113 has two NVLink domains of two nodes each.
    parallel["hybridep_num_ranks_per_nvlink_domain"] = args.ranks_per_nvlink_domain
    master = MasterConfig.model_validate(resolved)
    policy = master.policy
    policy["pretrained_checkpoint"] = master.checkpointing["pretrained_checkpoint"]
    for key, value in policy["megatron_cfg"]["env_vars"].items():
        os.environ[key] = str(value)
    config, provider = setup_model_config(
        policy,
        int(os.environ.get("RANK", "0")),
        torch.bfloat16,
        policy["model_name"],
        policy["pretrained_checkpoint"]["path"],
    )
    provider.calculate_per_token_loss = True
    assert provider.num_layers == 72 and provider.hidden_size == 3072
    assert provider.num_moe_experts == 512
    assert (
        provider.mamba_num_heads,
        provider.mamba_head_dim,
        provider.mamba_num_groups,
        provider.mamba_state_dim,
    ) == (24, 64, 24, 128)
    assert provider.mtp_num_layers == 2
    assert provider.linear_cp_layout == "contiguous"
    assert provider.fp8_recipe == "mxfp8" and provider.fp8_param
    return config, provider, policy["pretrained_checkpoint"]["path"]


def architecture(provider):
    fields = [
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "kv_channels",
        "num_moe_experts",
        "moe_ffn_hidden_size",
        "moe_latent_size",
        "hybrid_layer_pattern",
        "mamba_num_heads",
        "mamba_head_dim",
        "mamba_num_groups",
        "mamba_state_dim",
        "gdp_num_householder",
        "mtp_num_layers",
        "mtp_use_repeated_layer",
        "vocab_size",
        "linear_cp_layout",
        "attention_cp_layout",
    ]
    return {name: getattr(provider, name) for name in fields}


def tensor_path(root, *, name, expert, rank, kind):
    owner = rank if expert else 0
    return (
        root
        / f"rank-{owner}"
        / f"{hashlib.sha256(name.encode()).hexdigest()}.{kind}.pt"
    )


def full_tensor(param, value, expected_shape):
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.gtp_api import is_gtp_param

    if is_gtp_param(param):
        group = ps.get_gtp_weight_remat_group()
        shards = [torch.empty_like(value) for _ in range(group.size())]
        dist.all_gather(shards, value.contiguous(), group=group)
        value = torch.cat(shards, dim=0)[: expected_shape[0]]
    assert tuple(value.shape) == tuple(expected_shape), (value.shape, expected_shape)
    return value


def mock_batch(args, provider):
    from megatron.core import parallel_state as ps
    from megatron.core.packed_seq_params import PackedSeqParams

    group = ps.get_gtp_weight_remat_group()
    lane = ps.get_data_parallel_rank(with_gtp_remat=False) * group.size() + group.rank()
    generator = torch.Generator().manual_seed(42000 + lane)
    tokens = torch.randint(
        32, provider.vocab_size, (1, args.sequence_length), generator=generator
    )
    labels = torch.roll(tokens, shifts=-1, dims=1)
    positions = torch.arange(args.sequence_length).unsqueeze(0)
    mask = (positions % (lane + 2) != 0).float()
    mask[:, -3:] = 0
    identity = hashlib.sha256()
    for tensor in (tokens, labels, positions, mask):
        identity.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    boundaries = torch.tensor(
        [0, args.sequence_length], dtype=torch.int32, device="cuda"
    )
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries,
        cu_seqlens_q_padded=boundaries,
        cu_seqlens_kv_padded=boundaries,
        max_seqlen_q=args.sequence_length,
        max_seqlen_kv=args.sequence_length,
        seq_idx=torch.zeros_like(tokens, dtype=torch.int32, device="cuda"),
    )
    batch = {
        "input_ids": tokens.cuda(),
        "labels": labels.cuda(),
        "position_ids": positions.cuda(),
        "loss_mask": mask.cuda(),
        "packed_seq_params": packed,
    }
    # Inputs are replicated over CP, so count each logical lane only once.
    count = (
        mask.sum().cuda()
        if ps.get_context_parallel_rank() == 0
        else mask.new_zeros(()).cuda()
    )
    dist.all_reduce(count)
    return batch, identity.hexdigest(), count, lane


def run_phase(args, *, gtp_size):
    from megatron.bridge.models.model_provider import get_model
    from megatron.bridge.training.checkpointing import (
        _load_model_weights_from_checkpoint,
    )
    from megatron.bridge.training.gtp import (
        classify_gtp_remat_chains,
        configure_gtp_remat,
    )
    from megatron.core import parallel_state as ps
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.fp8_utils import dequantize_fp8_tensor, is_mxfp8tensor
    from megatron.core.num_microbatches_calculator import (
        destroy_num_microbatches_calculator,
        init_num_microbatches_calculator,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        dequantize_gtp_native_fp8,
        reset_gtp_state,
    )
    from megatron.core.tensor_parallel.gtp_api import is_gtp_param
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    rank = dist.get_rank()
    config, provider, checkpoint = make_config(args, gtp_size=gtp_size)
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=args.cp_size,
        expert_model_parallel_size=args.ep_size,
        expert_tensor_parallel_size=1,
        gtp_remat_size=gtp_size,
        expert_gtp_remat_size=1,
    )
    model_parallel_cuda_manual_seed(42)
    init_num_microbatches_calculator(
        rank=rank,
        global_batch_size=16 // args.cp_size,
        micro_batch_size=1,
        data_parallel_size=16 // args.cp_size,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    provider._pg_collection = groups
    configure_gtp_remat(provider)
    config.ddp.use_distributed_optimizer = False
    config.ddp.overlap_grad_reduce = False
    config.ddp.overlap_param_gather = False
    config.ddp.grad_reduce_in_fp32 = True

    def load_weights(chunks):
        _load_model_weights_from_checkpoint(
            checkpoint,
            chunks,
            fully_parallel_load=config.checkpoint.fully_parallel_load,
        )
        return chunks

    if rank == 0:
        print(f"GTP={gtp_size} building full NM4 and loading {checkpoint}", flush=True)
    chunks = get_model(
        provider,
        config.ddp,
        wrap_with_ddp=True,
        use_layer_wise_distributed_optimizer=False,
        pre_wrap_hook=[load_weights],
        pg_collection=groups,
    )
    assert len(chunks) == 1
    model = chunks[0]
    classify_gtp_remat_chains(chunks, provider)
    named = list(model.module.named_parameters())
    native_count = sum(is_mxfp8tensor(param) for _, param in named)
    shard_count = sum(is_gtp_param(param) for _, param in named)
    assert native_count > 0 and (shard_count > 0) == (gtp_size == 2)
    root = (
        args.baseline_dir
        if args.baseline_dir is not None
        else args.output.parent / "baseline"
    )
    owner_dir = root / f"rank-{rank}"
    owner_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = owner_dir / "manifest.json"
    baseline = json.loads(manifest_file.read_text()) if gtp_size == 2 else {}
    weights = {}
    for name, param in named:
        expert = not getattr(param, "allreduce", True)
        shape = (
            tuple(param.shape)
            if gtp_size == 1
            else baseline["parameters"][name]["shape"]
        )
        with torch.no_grad():
            if is_mxfp8tensor(param):
                value = (
                    dequantize_gtp_native_fp8(param)
                    if is_gtp_param(param)
                    else dequantize_fp8_tensor(param)
                )
            else:
                value = param.detach()
            value = full_tensor(param, value, shape).float().cpu().contiguous()
        checksum = digest(value)
        item = {
            "shape": list(shape),
            "physical_shape": list(param.shape),
            "gtp_sharded": is_gtp_param(param),
            "expert": expert,
            "weight_sha256": checksum,
        }
        path = tensor_path(root, name=name, expert=expert, rank=rank, kind="weight")
        if gtp_size == 1:
            if not expert and rank == 0:
                torch.save(value, path)
        else:
            item["weight_hash_matches"] = (
                checksum == baseline["parameters"][name]["weight_sha256"]
            )
            if not expert and rank == 0:
                reference = torch.load(path, weights_only=True).float()
                difference = value.float() - reference
                item["weight_relative_l2"] = float(
                    difference.norm() / reference.norm().clamp_min(1e-12)
                )
                item["weight_max_absolute_error"] = float(difference.abs().max())
        weights[name] = item
        del value
    batch, input_hash, token_count, lane = mock_batch(args, provider)
    if gtp_size == 2:
        assert input_hash == baseline["input_sha256"] and lane == baseline["lane"]
        assert int(token_count) == baseline["valid_tokens"]
    model.zero_grad_buffer()
    if rank == 0:
        print(
            f"GTP={gtp_size} forward: full 72-layer NM4, MTP2, tokens={int(token_count)}",
            flush=True,
        )
    with fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
        result = model(**batch)
        assert isinstance(result, tuple) and len(result) == 2, (
            "Provider must return local CP loss mask"
        )
        token_losses, local_mask = result
        local_sum = (
            token_losses.float().reshape(-1) * local_mask.float().reshape(-1)
        ).sum()
    loss = local_sum.detach().clone()
    dist.all_reduce(loss)
    loss = float(loss / token_count)
    if rank == 0:
        print(f"GTP={gtp_size} loss={loss:.9f}; backward and finalize", flush=True)
    # Match NeMo-RL: accumulate sums (including MTP), reduce, then divide all gradients
    # by the single global valid-token denominator. Never average per-rank means.
    local_sum.backward()
    finalize_model_grads(chunks, num_tokens=None)
    failures = []
    for name, param in named:
        expert = weights[name]["expert"]
        if not param.requires_grad:
            weights[name]["gradient_checked"] = False
            continue
        gradient = full_tensor(
            param, param.main_grad.detach().float(), weights[name]["shape"]
        )
        gradient = (gradient / token_count).cpu()
        norm = float(gradient.norm())
        weights[name]["gradient_norm"] = norm
        if expert or rank == 0:
            path = tensor_path(
                root, name=name, expert=expert, rank=rank, kind="gradient"
            )
            if gtp_size == 1:
                if norm > 0:
                    torch.save(gradient, path)
            else:
                base_norm = baseline["parameters"][name]["gradient_norm"]
                reference = (
                    torch.load(path, weights_only=True)
                    if base_norm > 0
                    else torch.zeros_like(gradient)
                )
                difference = gradient - reference
                relative = float(difference.norm()) / max(base_norm, 1e-12)
                finite = bool(
                    torch.isfinite(gradient).all() and torch.isfinite(reference).all()
                )
                weights[name].update(
                    gradient_checked=True,
                    gradient_relative_l2=relative,
                    gradient_norm_ratio=norm / max(base_norm, 1e-12),
                    gradient_max_absolute_error=float(difference.abs().max()),
                )
                if not finite or relative > args.grad_rtol:
                    failures.append(name)
        del gradient
    report = {
        "gtp_size": gtp_size,
        "architecture": architecture(provider),
        "checkpoint": checkpoint,
        "input_sha256": input_hash,
        "lane": lane,
        "valid_tokens": int(token_count),
        "loss": loss,
        "parameters": weights,
        "native_mxfp8_parameters": native_count,
        "gtp_parameters": shard_count,
        "gradient_failures": failures,
        "optimizer_updates": 0,
    }
    if gtp_size == 1:
        manifest_file.write_text(json.dumps(report, indent=2) + "\n")
    else:
        report["baseline_loss"] = baseline["loss"]
        report["loss_relative"] = abs(loss - baseline["loss"]) / abs(baseline["loss"])
        report["weight_hash_mismatches"] = [
            name for name, item in weights.items() if not item["weight_hash_matches"]
        ]
        report["passed"] = (
            not failures
            and not report["weight_hash_mismatches"]
            and report["loss_relative"] <= args.loss_rtol
        )
        (args.output.parent / f"rank-{rank}.parity.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    del model, chunks, named, batch, result, token_losses, local_mask, local_sum
    dist.barrier()
    ps.destroy_model_parallel()
    destroy_num_microbatches_calculator()
    reset_gtp_state()
    FP8GlobalStateManager.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        default="examples/configs/sft_v2_tests/nm4_generalist_32n_mxfp8_cp64ep64_gtp2_gbs2_hybridep.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--loss-rtol", type=float, default=1e-3)
    parser.add_argument("--grad-rtol", type=float, default=2e-2)
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--ranks-per-nvlink-domain", type=int, default=8)
    parser.add_argument("--cp-size", type=int, default=8)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--reuse-baseline", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.describe_only:
        _, provider, checkpoint = make_config(args, gtp_size=1)
        args.output.write_text(
            json.dumps(
                {"architecture": architecture(provider), "checkpoint": checkpoint},
                indent=2,
            )
            + "\n"
        )
        print(args.output.read_text(), flush=True)
        return
    assert 16 % (args.cp_size * 2) == 0 and 16 % args.ep_size == 0
    assert (
        args.sequence_length >= 512
        and args.sequence_length % (2 * args.cp_size * 32) == 0
    )
    trace_file = (args.output.parent / f"rank-{os.environ['RANK']}.stack.log").open("w")
    faulthandler.dump_traceback_later(180, repeat=True, file=trace_file)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    assert dist.get_world_size() == 16
    if not args.reuse_baseline:
        run_phase(args, gtp_size=1)
    report = run_phase(args, gtp_size=2)
    passed = torch.tensor(int(report["passed"]), device="cuda")
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if dist.get_rank() == 0:
        summary = {
            "passed": bool(passed),
            "baseline_loss": report["baseline_loss"],
            "gtp2_loss": report["loss"],
            "loss_relative": report["loss_relative"],
            "architecture": report["architecture"],
            "optimizer_updates": 0,
            "world_size": 16,
            "cp_size": args.cp_size,
            "ep_size": args.ep_size,
            "expert_dp": 16 // args.ep_size,
            "baseline_dp": 16 // args.cp_size,
            "gtp2_dp": 16 // args.cp_size // 2,
            "logical_data_lanes": 16 // args.cp_size,
            "gradient_rtol": args.grad_rtol,
            "loss_rtol": args.loss_rtol,
            "scope": "Full checkpoint architecture, text-only mock input; no optimizer update",
        }
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"FULL NM4 PARITY {'PASS' if bool(passed) else 'FAIL'}: {args.output}",
            flush=True,
        )
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    trace_file.close()
    if not bool(passed):
        raise AssertionError(
            "Full NM4 GTP parity failed; inspect per-rank JSON reports"
        )


if __name__ == "__main__":
    main()
