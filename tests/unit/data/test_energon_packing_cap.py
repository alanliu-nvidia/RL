# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonPackingOptions


@pytest.mark.parametrize("cap", [0, -1])
def test_packing_cap_rejects_nonpositive_values(cap: int) -> None:
    with pytest.raises(ValueError):
        EnergonPackingOptions(
            max_sequence_length=128,
            sequence_length_pad_multiple=8,
            max_sequences_per_bin=cap,
        )


@pytest.mark.mcore
@pytest.mark.parametrize("cap", [None, 1, 2])
def test_task_encoder_packing_cap_limits_actual_bin_membership(
    monkeypatch: pytest.MonkeyPatch, cap: int | None
) -> None:
    from nemo_rl.data.energon import sft_dataloader as module

    config = EnergonLoaderConfig(
        model_family="qwen",
        task_encoder={
            "packing": {
                "name": "greedy_knapsack",
                "buffer_size": 10,
                "options": {
                    "max_sequence_length": 128,
                    "sequence_length_pad_multiple": 8,
                    "max_sequences_per_bin": cap,
                },
            }
        },
    )
    encoder_type = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        module.TASK_ENCODER_REGISTRY,
        "resolve_configured",
        Mock(return_value=encoder_type),
    )
    encoder = module._task_encoder(
        loader_config=config,
        adapter=object(),
        include_source_ids=True,
        max_sequence_length=128,
        tokenizer=object(),
        only_unmask_final=False,
    )
    bins = encoder.packer.pack([8, 8, 8])
    assert sorted(index for packed in bins for index in packed) == [0, 1, 2]
    assert max(map(len, bins)) == (3 if cap is None else cap)


@pytest.mark.mcore
@pytest.mark.parametrize(
    "cap,packed", [(None, False), (None, True), (1, True), (2, True)]
)
def test_loader_identity_records_task_encoder_packing_cap(
    monkeypatch: pytest.MonkeyPatch, cap: int | None, packed: bool
) -> None:
    from nemo_rl.data.energon import sft_dataloader as module

    packing = {
        "name": "greedy_knapsack",
        "buffer_size": 10,
        "options": {
            "max_sequence_length": 128,
            "sequence_length_pad_multiple": 8,
            "max_sequences_per_bin": cap,
        },
    }
    config = EnergonLoaderConfig(
        model_family="qwen", task_encoder={"packing": packing} if packed else {}
    )
    monkeypatch.setattr(module, "_loader_config", lambda _: config)
    monkeypatch.setattr(
        module,
        "build_processor_adapter",
        Mock(return_value=SimpleNamespace(fingerprint="adapter")),
    )
    monkeypatch.setattr(
        module, "_task_encoder", Mock(return_value=SimpleNamespace(cookers=[]))
    )
    monkeypatch.setattr(module, "_worker_config", Mock())
    monkeypatch.setattr(module, "get_val_dataset", Mock())
    monkeypatch.setattr(module, "get_savable_loader", Mock())
    monkeypatch.setattr(module, "_v2_topology", Mock(return_value={}))
    monkeypatch.setattr(module, "_loader_identity", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        module,
        "EnergonSFTDataLoader",
        lambda loader, *, identity: SimpleNamespace(identity=identity),
    )
    loader = module.build_energon_sft_loader(
        data_config={"energon": config},
        source={"path": "/data/prepared", "split": "validation"},
        processor=SimpleNamespace(tokenizer=object()),
        batch_size=1,
        max_sequence_length=128,
        split_role="validation",
        logical_rank=0,
        logical_world_size=1,
        placement_fingerprint="placement",
        only_unmask_final=False,
    )
    assert loader.identity["max_sequences_per_bin"] == cap
    assert loader.identity["packing_algorithm"] == (
        "greedy_knapsack" if packed else None
    )


@pytest.mark.mcore
@pytest.mark.parametrize("algorithm", ["balanced_greedy_knapsack", "greedy_knapsack"])
def test_task_encoder_balancing_delta(
    monkeypatch: pytest.MonkeyPatch,
    algorithm: str,
) -> None:
    from nemo_rl.data.energon import sft_dataloader as module

    config = EnergonLoaderConfig(
        model_family="qwen",
        task_encoder={
            "packing": {
                "name": algorithm,
                "buffer_size": 10,
                "options": {
                    "max_sequence_length": 128,
                    "sequence_length_pad_multiple": 8,
                    "balanced_knapsack_delta": 5,
                },
            }
        },
    )
    monkeypatch.setattr(module.TASK_ENCODER_REGISTRY, "resolve_configured", Mock())

    def build_encoder():
        return module._task_encoder(
            loader_config=config,
            adapter=object(),
            include_source_ids=True,
            max_sequence_length=128,
            tokenizer=object(),
            only_unmask_final=False,
        )

    if algorithm == "balanced_greedy_knapsack":
        build_encoder()
        packer = module.TASK_ENCODER_REGISTRY.resolve_configured.return_value.call_args.kwargs[
            "packer"
        ]
        assert packer.balanced_knapsack_delta == 5
    else:
        with pytest.raises(
            ValueError, match="balanced_knapsack_delta is only supported"
        ):
            build_encoder()
