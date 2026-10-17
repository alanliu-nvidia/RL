# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import builtins
import pickle

import pytest
from pydantic import BaseModel, ValidationError

from nemo_rl.models.megatron.mimo import (
    MIMOSFTConfig,
    mimo_worker_sharding,
    resolve_mimo_sft_config,
    validate_mimo_parallelism,
)


@pytest.fixture
def bridge_available() -> None:
    """Skip Bridge-backed cases in unit lanes without the mcore extra."""
    pytest.importorskip("megatron.bridge")


def _megatron_config() -> dict[str, object]:
    return {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 2,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
    }


def _mimo_config() -> dict[str, object]:
    return {
        "parallelism": {
            "module_parallelisms": {
                "images": {
                    "data_parallel_size": 4,
                    "rank_offset": 0,
                },
                "language": {
                    "tensor_model_parallel_size": 2,
                    "context_parallel_size": 2,
                    "data_parallel_size": 1,
                    "rank_offset": 4,
                },
            }
        },
    }


@pytest.mark.mcore
def test_mimo_config_uses_bridge_parallelism_and_explicit_topology(
    bridge_available: None,
) -> None:
    resolved = resolve_mimo_sft_config(
        {"megatron_cfg": {"enabled": True, "mimo": _mimo_config()}}
    )
    assert resolved is not None

    parallelism = validate_mimo_parallelism(
        resolved,
        world_size=8,
        megatron_config=_megatron_config(),
    )

    assert type(parallelism).__name__ == "MegatronMIMOParallelismConfig"
    assert type(parallelism.get_parallelism("language")).__name__ == (
        "ModuleParallelismConfig"
    )
    assert resolved.parallelism == _mimo_config()["parallelism"]
    assert parallelism.total_world_size == 8
    assert mimo_worker_sharding(8).names == ["worker"]


@pytest.mark.mcore
def test_mimo_language_parallelism_must_match_first_class_megatron_config(
    bridge_available: None,
) -> None:
    megatron_config = _megatron_config()
    megatron_config["context_parallel_size"] = 1
    resolved = resolve_mimo_sft_config(
        {"megatron_cfg": {"enabled": True, "mimo": _mimo_config()}}
    )
    assert resolved is not None

    with pytest.raises(ValueError, match="language parallelism must match"):
        validate_mimo_parallelism(
            resolved,
            world_size=8,
            megatron_config=megatron_config,
        )


def test_mimo_config_is_absent_without_mimo_block() -> None:
    assert resolve_mimo_sft_config({"megatron_cfg": {"enabled": True}}) is None


def test_mimo_config_validation_and_transport_do_not_import_backend(monkeypatch):
    original_import = builtins.__import__

    def without_megatron(name, *args, **kwargs):
        if name == "megatron" or name.startswith("megatron."):
            raise AssertionError(f"Driver imported backend: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_megatron)

    class Config(BaseModel):
        mimo: MIMOSFTConfig

    config = Config.model_validate({"mimo": _mimo_config()})
    resolved = resolve_mimo_sft_config({"megatron_cfg": config.model_dump()})
    assert resolved is not None
    assert (
        pickle.loads(pickle.dumps(resolved)).parallelism
        == _mimo_config()["parallelism"]
    )


def test_mimo_block_requires_explicit_bridge_parallelism() -> None:
    with pytest.raises(ValidationError):
        resolve_mimo_sft_config({"megatron_cfg": {"enabled": True, "mimo": {}}})


def test_mimo_loader_mode_defaults_to_canonical_and_accepts_independent() -> None:
    assert MIMOSFTConfig.model_validate(_mimo_config()).single_owner_loader is True
    config = {**_mimo_config(), "single_owner_loader": False}
    assert MIMOSFTConfig.model_validate(config).single_owner_loader is False
