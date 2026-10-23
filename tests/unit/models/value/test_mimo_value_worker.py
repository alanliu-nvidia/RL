# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Backend wiring checks; require the optional Megatron-Bridge environment."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import nemo_rl  # noqa: F401 -- initialize the vendored Megatron namespace first

pytest.importorskip("megatron.bridge")

from nemo_rl.models.value.workers.megatron_mimo_value_worker import (  # noqa: E402
    MegatronMIMOValueWorkerImpl,
    install_mimo_value_head,
)

pytestmark = pytest.mark.mcore


def test_head_is_installed_on_language_and_base_load_skip_reaches_outer_model():
    language = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4, sequence_parallel=False),
        output_layer=torch.nn.Linear(4, 8, bias=False),
    )
    language.sharded_state_dict = lambda prefix="", **kwargs: {
        prefix + "output_layer.weight": language.output_layer.weight,
        prefix + "embedding.weight": torch.ones(8, 4),
    }
    outer = SimpleNamespace(language_model=language)
    encoder = SimpleNamespace(language_model=None)
    with (
        patch(
            "megatron.core.parallel_state.get_pipeline_model_parallel_world_size",
            side_effect=AssertionError("MIMO globals are not installed at pre-wrap"),
        ),
        patch(
            "megatron.core.parallel_state.is_pipeline_last_stage",
            side_effect=AssertionError("MIMO globals unavailable"),
        ),
    ):
        models = install_mimo_value_head([outer, encoder])
    assert models == [outer, encoder]
    assert language.output_layer.weight.shape == (1, 4)
    assert language.share_embeddings_and_output_weights is False
    assert "lm.output_layer.weight" in language.sharded_state_dict(prefix="lm.")
    with outer.hide_loss_modules():
        assert "lm.output_layer.weight" not in language.sharded_state_dict(prefix="lm.")
        assert "lm.embedding.weight" in language.sharded_state_dict(prefix="lm.")
    assert "lm.output_layer.weight" in language.sharded_state_dict(prefix="lm.")


def test_value_forward_reuses_policy_inference_with_scalar_processor():
    worker = object.__new__(MegatronMIMOValueWorkerImpl)
    observed = {}

    def forward(**kwargs):
        observed.update(kwargs)
        return {"values": torch.ones(1, 3)}

    worker._get_mimo_predictions = forward
    data = {"pixel_values": torch.ones(1, 4, 8), "imgs_sizes": torch.tensor([[2, 2]])}
    with patch("torch.no_grad", return_value=nullcontext()):
        out = worker.get_values(
            data, num_microbatches=1, sequence_length=3, split_encoder_media=True
        )
    assert observed["data"] is data
    assert observed["result_key"] == "values"
    assert observed["split_encoder_media"] is True
    assert observed["post_processing_fn"].loss_fn is None
    assert out["values"].shape == (1, 3)


def test_finish_training_offloads_model_and_optimizer() -> None:
    worker = object.__new__(MegatronMIMOValueWorkerImpl)
    worker.model = MagicMock()
    worker.optimizer = object()
    worker._train_step_state = None
    worker.optimizer_cpu_offload = False
    worker.move_model = MagicMock(return_value=worker.model)
    worker.move_optimizer = MagicMock()
    with patch("torch.cuda.empty_cache"), patch("gc.collect"):
        worker.finish_training()
    worker.move_model.assert_called_once_with(
        worker.model, "cpu", move_params=True, move_grads=True
    )
    worker.move_optimizer.assert_called_once_with("cpu")
