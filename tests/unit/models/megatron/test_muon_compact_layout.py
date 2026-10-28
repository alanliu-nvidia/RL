# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Check compact-layout wiring before the paired NM4 production experiments."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("layout", [True, False])
def test_bridge_forwards_layout_choice(monkeypatch, layout):
    import megatron.bridge.models.model_provider as module

    model = [torch.nn.Linear(4, 4)]
    config = SimpleNamespace(
        use_cpu_initialization=True,
        init_model_with_meta_device=False,
        fp16=False,
        bf16=False,
    )
    monkeypatch.setattr(module, "_create_model", Mock(return_value=model))
    monkeypatch.setattr(module, "_print_num_params", Mock())
    monkeypatch.setattr(module, "get_model_config", Mock(return_value=config))
    monkeypatch.setattr(module, "correct_amax_history_if_needed", None)
    wrap = Mock(return_value=model)
    monkeypatch.setattr(module, "_ddp_wrap", wrap)
    result = module.get_model(
        SimpleNamespace(),
        SimpleNamespace(),
        pg_collection=SimpleNamespace(),
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=layout,
    )
    assert result is model
    assert wrap.call_args.kwargs["use_layer_wise_param_layout"] is layout
    assert wrap.call_args.kwargs["use_layer_wise_distributed_optimizer"] is True


def test_bridge_default_preserves_padded_layout(monkeypatch):
    import inspect

    from megatron.bridge.models.model_provider import get_model

    assert (
        inspect.signature(get_model).parameters["use_layer_wise_param_layout"].default
        is True
    )


@pytest.mark.parametrize(
    "distributed,layerwise", [(False, True), (False, False), (True, True)]
)
def test_nm4_compact_configuration_validation(distributed, layerwise):
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.sft_v2 import MasterConfig
    from nemo_rl.models.megatron.setup import setup_model_config
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    data = OmegaConf.to_container(
        load_config(
            "examples/configs/sft_v2_tests/nm4_generalist_128n_mxfp8_cp64ep64_gbs32_compactmuon_nooffload.yaml"
        ),
        resolve=True,
    )
    options = data["policy"]["megatron_cfg"]
    options["use_layer_wise_param_layout"] = False
    options["optimizer"]["use_distributed_optimizer"] = distributed
    options["optimizer"]["use_layer_wise_distributed_optimizer"] = layerwise
    master = MasterConfig.model_validate(data)
    policy = master.policy
    policy["pretrained_checkpoint"] = master.checkpointing["pretrained_checkpoint"]
    assert policy["megatron_cfg"]["use_layer_wise_param_layout"] is False
    assert (
        policy["megatron_cfg"]["optimizer"]["use_layer_wise_distributed_optimizer"]
        is layerwise
    )

    def build():
        return setup_model_config(
            policy,
            0,
            torch.bfloat16,
            policy["model_name"],
            policy["pretrained_checkpoint"]["path"],
        )

    if distributed or not layerwise:
        with pytest.raises(
            ValueError, match="use_layer_wise_param_layout=False requires"
        ):
            build()
    else:
        config, provider = build()
        assert config.optimizer.use_layer_wise_distributed_optimizer is True
        assert config.optimizer.use_distributed_optimizer is False
        assert config.ddp.use_distributed_optimizer is False
        assert provider.num_layers == 72 and provider.hidden_size == 3072
        assert provider.fp8_recipe == "mxfp8" and provider.fp8_param
