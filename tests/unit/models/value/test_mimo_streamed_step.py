# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise real MIMO worker methods and scalar loss with a CPU schedule stand-in.

Native heterogeneous collectives still require the GPU integration gate. These
checks isolate accumulation, normalization, commit and resident-inference state.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import pytest
import torch

from nemo_rl.algorithms.loss.interfaces import LossType
from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.value import ValuePostProcessor


def _worker_methods(path, class_name, names, namespace, base=None):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node.bases = [] if base is None else [ast.Name(id="Base", ctx=ast.Load())]
    if base is not None:
        namespace["Base"] = base
    node.decorator_list = []
    node.body = [
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for method in node.body:
        method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


class _Model:
    def __init__(self, per_token):
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.syncs = 0
        self.depth = 0
        self.training = True
        self.ddp_config = SimpleNamespace(
            use_distributed_optimizer=True,
            reuse_grad_buf_for_mxfp8_param_ag=False,
        )
        self.config = SimpleNamespace(
            grad_sync_func=lambda: None,
            no_sync_func=nullcontext,
            finalize_model_grads_func=self.finalize,
            calculate_per_token_loss=per_token,
        )

    def finalize(self, models, tokens, **kwargs):
        assert models == [self]
        assert (
            tokens is not None
            if self.config.calculate_per_token_loss
            else tokens is None
        )
        if tokens is not None:
            assert tokens.item() == 0
        self.syncs += 1

    @contextmanager
    def no_sync(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def scale_gradients(self, factor):
        self.weight.grad.mul_(factor)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


@pytest.fixture
def make_worker(monkeypatch):
    leaf = ModuleType("megatron.bridge.training.megatron_mimo_parallel_utils")
    leaf.is_current_rank_in_grid = lambda grid: True
    leaf.zero_grad_buffer_for_multimodule = lambda modules: None
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def aggregate(*, all_mb_metrics, losses, data_parallel_group):
        out = defaultdict(list)
        for metrics in all_mb_metrics:
            for key, value in metrics.items():
                out[key].append(value)
        return dict(out), torch.tensor(sum(losses))

    def schedule(**kwargs):
        model = kwargs["model"]
        assert model.depth == 1
        assert model.config.finalize_model_grads_func is None
        batch = next(kwargs["data_iterator"])
        collect = kwargs["post_processing_fn"](
            data_dict=batch,
            cu_seqlens_padded=None,
            original_seq_length=batch["features"].shape[1],
            cp_layout="contiguous",
            cp_group=None,
            global_valid_seqs=kwargs["global_valid_seqs"],
            global_valid_toks=kwargs["global_valid_toks"],
        )
        loss, metrics = collect((model.weight * batch["features"]).unsqueeze(-1))
        loss.backward()
        return [metrics]

    namespace = {
        "torch": torch,
        "nullcontext": nullcontext,
        "Any": Any,
        "Optional": Optional,
        "BatchedDataDict": BatchedDataDict,
        "MIMO_LANGUAGE_MODULE_KEY": "language",
        "DistributedDataParallel": _Model,
        "gc": __import__("gc"),
        "aggregate_training_statistics": aggregate,
        "megatron_mimo_forward_backward": schedule,
        "time": __import__("time"),
        "get_microbatch_iterator": lambda data, *args, **kwargs: (
            iter([data]),
            1,
            1,
            data["features"].shape[1],
            data["features"].shape[1],
        ),
    }
    root = Path(__file__).parents[4]
    worker_type = _worker_methods(
        root / "nemo_rl/models/policy/workers/megatron_policy_worker.py",
        "MegatronPolicyWorkerImpl",
        {
            "_suspend_mimo_train_hooks",
            "_mimo_active_modules",
            "_restore_saved_mcore_hooks",
            "_train_mimo_microbatch_body",
            "_finish_train_step_body",
            "abort_train_step",
            "pause_train_step_with_offloading",
        },
        namespace,
    )

    def build(*, cliprange=None, per_token=False):
        worker = worker_type()
        worker.model = model = _Model(per_token)
        worker.mimo_runtime = SimpleNamespace(
            active_module_name="language",
            module_to_grid_tuple=[(model, object())],
            multimodule_pg_collection=object(),
            local_pg_collection=SimpleNamespace(dp_cp=object()),
            schedulers={},
        )
        worker.cfg = {
            "megatron_cfg": {
                "distributed_data_parallel_config": {"overlap_grad_reduce": False}
            }
        }
        worker.mimo_encoder_media_split = None
        worker.delegate_pack_to_model = False
        worker.delegate_mtp_loss_mask_to_model = False
        worker.mtp_enabled = False
        worker.mcore_state = SimpleNamespace(straggler_timer=None)
        worker.sampling_params = None
        worker.scheduler = SimpleNamespace(get_lr=lambda group: 0.1, get_wd=lambda: 0.0)
        worker._collect_mtp_metrics = lambda *args, **kwargs: None
        worker.finalize_async_save = lambda: None
        worker._release_opd_full_teacher_lm_head = lambda: None
        worker._add_step_phase = lambda *args: None
        worker._assemble_step_metrics = lambda **kwargs: kwargs
        worker._mimo_loss_post_processor = lambda fn, n: ValuePostProcessor(fn, n)
        optimizer = torch.optim.SGD([model.weight], lr=0.1)
        commits = []

        def step():
            commits.append(float(model.weight.grad))
            optimizer.step()
            return True, abs(commits[-1]), None

        worker.optimizer = SimpleNamespace(
            step=step,
            param_groups=optimizer.param_groups,
            zero_grad=optimizer.zero_grad,
        )
        loss_fn = MseValueLossFn(MseValueLossConfig(cliprange=cliprange))
        metric_normalizations = getattr(loss_fn, "metric_normalizations", {})
        worker._train_step_state = {
            "num_chunks": 0,
            "total_num_microbatches": 0,
            "mbs": 1,
            "gbs": 4,
            "loss_fn": loss_fn,
            "loss_type": LossType.TOKEN_LEVEL,
            "metric_normalizations": metric_normalizations
            if isinstance(metric_normalizations, dict)
            else {},
            "mimo_losses_reduced": [],
        }
        worker._suspend_mimo_train_hooks(worker._train_step_state)
        worker._assert_step_open = lambda: worker._train_step_state

        def counts(data):
            mask = data["sample_mask"]
            return mask.sum(), (data["token_mask"][:, 1:] * mask[:, None]).sum()

        worker._reduce_mimo_valid_counts = counts
        return worker, commits, loss_fn

    return build


def _batch():
    return {
        "features": torch.arange(16, dtype=torch.float32).reshape(4, 4) / 8,
        "returns": torch.full((4, 4), 0.4),
        "values": torch.zeros(4, 4),
        "sample_mask": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "token_mask": torch.tensor(
            [[0, 1, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]]
        ),
    }


@pytest.mark.parametrize("cliprange", [None, 0.2])
@pytest.mark.parametrize("per_token", [False, True])
def test_uneven_mimo_chunks_match_whole_batch_gradient_and_commit(
    make_worker, cliprange, per_token
):
    outputs = []
    for partitions in ((4,), (3, 1)):
        worker, commits, loss_fn = make_worker(cliprange=cliprange, per_token=per_token)
        data = _batch()
        start = 0
        state = worker._train_step_state
        for size in partitions:
            chunk = {key: value[start : start + size] for key, value in data.items()}
            worker._train_mimo_microbatch_body(
                state, chunk, num_microbatches=1, sequence_length=4
            )
            start += size
            assert worker.model.syncs == 0
            assert commits == []
        result = worker._finish_train_step_body(state)
        assert worker.model.syncs == 1
        assert len(commits) == 1
        assert worker._train_step_state is None
        assert state["global_valid_seqs"].item() == 3
        assert state["global_valid_toks"].item() == 6
        assert state["num_chunks"] == len(partitions)
        outputs.append(
            (worker.model.weight.detach().clone(), commits[0], result["global_loss"])
        )
    for whole, chunked in zip(*outputs):
        torch.testing.assert_close(
            torch.as_tensor(whole), torch.as_tensor(chunked), rtol=1e-6, atol=1e-7
        )


def test_abort_discards_accumulation_and_restores_schedule(make_worker):
    worker, commits, _ = make_worker()
    state = worker._train_step_state
    worker._train_mimo_microbatch_body(
        state, _batch(), num_microbatches=1, sequence_length=4
    )
    worker.abort_train_step()
    assert worker._train_step_state is None
    assert worker.model.weight.grad is None
    assert worker.model.config.finalize_model_grads_func == worker.model.finalize
    assert worker.model.syncs == 0
    assert commits == []


def test_mimo_policy_pause_keeps_pending_gradients_on_device(make_worker):
    worker, commits, _ = make_worker()
    state = worker._train_step_state
    worker._train_mimo_microbatch_body(
        state, _batch(), num_microbatches=1, sequence_length=4
    )
    grad = worker.model.weight.grad
    grad_copy = grad.clone()
    moves = []

    def move_model(model, device, *, move_params, move_grads):
        assert model is worker.model
        assert device == "cpu" and move_params and not move_grads
        moves.append("parameters")
        return model

    worker.move_model = move_model
    worker.move_optimizer = lambda device: moves.append("optimizer_" + device)
    worker.optimizer_cpu_offload = False
    worker.pause_train_step_with_offloading()
    assert state["offloaded"] is True
    assert worker.model.weight.grad is grad
    torch.testing.assert_close(grad, grad_copy)
    assert moves == ["parameters", "optimizer_cpu"]
    assert worker.model.config.finalize_model_grads_func is None
    assert commits == []


def test_resident_inference_pause_preserves_gradients_and_commits_once(monkeypatch):
    class Base:
        def _assert_step_open(self):
            assert self._train_step_state is not None
            return self._train_step_state

        def finish_train_step(self):
            self._train_step_state = None
            return {"global_loss": torch.tensor(8.0)}

    source = (
        Path(__file__).parents[4]
        / "nemo_rl/models/value/workers/megatron_mimo_value_worker.py"
    )
    worker_type = _worker_methods(
        source,
        "MegatronMIMOValueWorkerImpl",
        {
            "prepare_for_inference",
            "finish_inference",
            "train_microbatch",
            "finish_train_step",
            "finish_training",
        },
        {"torch": torch},
        base=Base,
    )
    worker = worker_type()
    worker.model = _Model(False)
    worker.model.weight.grad = torch.tensor(0.25)
    grad = worker.model.weight.grad
    worker._train_step_state = {"num_chunks": 1}
    worker.mimo_runtime = SimpleNamespace(
        local_pg_collection=SimpleNamespace(cp=object())
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    worker.prepare_for_inference(keep_resident=True, preserve_train_step=True)
    assert not worker.model.training
    assert worker.model.weight.grad is grad
    with pytest.raises(RuntimeError, match="already paused"):
        worker.prepare_for_inference(keep_resident=True, preserve_train_step=True)
    with pytest.raises(RuntimeError, match="Resume value training"):
        worker.train_microbatch(None)
    with pytest.raises(RuntimeError, match="Resume value training"):
        worker.finish_train_step()
    with pytest.raises(RuntimeError, match="Finish or abort"):
        worker.finish_training(keep_resident=True)
    worker.finish_inference(keep_resident=True, preserve_train_step=True)
    assert worker.model.training
    assert worker.model.weight.grad is grad
    assert grad.item() == 0.25
    result = worker.finish_train_step()
    assert result["global_loss"].item() == 4.0
    worker.finish_training(keep_resident=True)
    assert not worker.model.training
    assert worker.model.weight.grad is grad
