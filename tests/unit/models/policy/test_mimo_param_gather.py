# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression coverage for staging MIMO parameters after clearing shared buffers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.mcore


@pytest.mark.parametrize("overlap_param_gather", [False, True])
@pytest.mark.parametrize("reuse_grad_buffer", [False, True])
def test_mimo_begin_stages_shared_params_before_step_initialization(
    monkeypatch: pytest.MonkeyPatch,
    overlap_param_gather: bool,
    reuse_grad_buffer: bool,
) -> None:
    # These imports require the optional Megatron/TE worker environment.
    from megatron.bridge.training import megatron_mimo_parallel_utils

    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = (
        torch.nn.Module()
    )  # MIMO wraps its children, not this module, in DDP.
    worker.model.eval()
    worker.mimo_runtime = SimpleNamespace(module_to_grid_tuple=[(worker.model, None)])
    worker.megatron_cfg = SimpleNamespace(
        optimizer=SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buffer),
        ddp=SimpleNamespace(overlap_param_gather=overlap_param_gather),
    )
    worker._train_step_state = None
    worker._maybe_start_manual_gc = Mock()
    masters = torch.tensor([1.0, 2.0, 3.0, 4.0])
    shared_buffer = masters.clone()
    events: list[str] = []
    should_stage = overlap_param_gather and reuse_grad_buffer

    def clear_buffers(module_to_grid_tuple: list[tuple[object, object]]) -> None:
        assert module_to_grid_tuple is worker.mimo_runtime.module_to_grid_tuple
        shared_buffer.zero_()
        events.append("clear_buffers")

    def clear_optimizer_grads() -> None:
        events.append("clear_optimizer_grads")

    def stage_params() -> None:
        assert events == ["clear_buffers", "clear_optimizer_grads"]
        assert torch.count_nonzero(shared_buffer).item() == 0
        shared_buffer.copy_(masters)
        events.append("stage_params")

    def initialize_step(**kwargs: object) -> dict[str, object]:
        assert kwargs == {"loss_fn": None, "gbs": 4, "mbs": 1}
        expected = masters if should_stage else torch.zeros_like(masters)
        torch.testing.assert_close(shared_buffer, expected, rtol=0, atol=0)
        events.append("initialize_step")
        return {}

    worker.optimizer = SimpleNamespace(
        zero_grad=Mock(side_effect=clear_optimizer_grads),
        prepare_model_params_for_param_sync=Mock(side_effect=stage_params),
    )
    worker._split_step_state_init = Mock(side_effect=initialize_step)
    monkeypatch.setattr(
        megatron_mimo_parallel_utils, "zero_grad_buffer_for_multimodule", clear_buffers
    )

    worker.begin_train_step(None, gbs=4, mbs=1)

    expected_events = ["clear_buffers", "clear_optimizer_grads"]
    if should_stage:
        expected_events.append("stage_params")
        worker.optimizer.prepare_model_params_for_param_sync.assert_called_once_with()
    else:
        worker.optimizer.prepare_model_params_for_param_sync.assert_not_called()
    assert events == [*expected_events, "initialize_step"]
    assert worker.model.training
    assert worker._train_step_state == {"mimo": True, "mimo_losses_reduced": []}
