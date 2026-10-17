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

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

# sft_dataloader imports megatron.energon and sft_worker imports megatron.core,
# both of which ship only in the `mcore` extra. importorskip must run before the
# imports below: the mcore mark is applied in pytest_collection_modifyitems,
# too late to prevent a collection error.
pytest.importorskip("megatron.energon")
pytest.importorskip("megatron.core")

pytestmark = pytest.mark.mcore

from nemo_rl.data.energon.config import (  # noqa: E402
    EnergonLoaderConfig,
    EnergonSourceConfig,
)
from nemo_rl.data.energon.mimo_image_split import EncoderShard  # noqa: E402
from nemo_rl.data.energon.sft_dataloader import (  # noqa: E402
    _identity_fingerprint,
    _loader_identity,
    _v2_topology,
    _worker_config,
    build_energon_sft_loader,
)
from nemo_rl.data_plane import KVBatchMeta  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.distributed.ray_actor_environment_registry import (  # noqa: E402
    get_actor_python_env,
)


def _v2_fingerprint(
    *,
    source: EnergonSourceConfig,
    loader_config: EnergonLoaderConfig,
    adapter_fingerprint: str,
    split_role: str,
    logical_rank: int,
    logical_world_size: int,
    placement_fingerprint: str,
    batch_size: int = 1,
) -> str:
    """Hash a V2 loader identity the way ``build_energon_sft_loader`` does."""
    return _identity_fingerprint(
        _loader_identity(
            source=source,
            loader_config=loader_config,
            adapter_fingerprint=adapter_fingerprint,
            split_role=split_role,
            batch_size=batch_size,
            shuffle=split_role == "train",
            topology=_v2_topology(
                loader_config=loader_config,
                placement_fingerprint=placement_fingerprint,
                logical_rank=logical_rank,
                logical_world_size=logical_world_size,
            ),
            packing_algorithm=None,
            max_sequences_per_bin=None,
            sequence_length_pad_multiple=1,
            only_unmask_final=False,
        )
    )


def test_worker_config_uses_logical_data_rank() -> None:
    worker = _worker_config(
        EnergonLoaderConfig(model_family="qwen", num_workers=3),
        logical_rank=2,
        logical_world_size=4,
    )

    assert worker.rank == 2
    assert worker.world_size == 4
    assert worker.num_workers == 3


def test_v2_fingerprint_identifies_each_logical_shard() -> None:
    source = EnergonSourceConfig(path="/dataset", split="train", virtual_epoch_length=8)
    loader = EnergonLoaderConfig(model_family="qwen")
    common = {
        "source": source,
        "loader_config": loader,
        "adapter_fingerprint": "processor",
        "split_role": "train",
        "logical_world_size": 2,
        "placement_fingerprint": "placement",
    }

    rank_zero = _v2_fingerprint(logical_rank=0, **common)
    rank_one = _v2_fingerprint(logical_rank=1, **common)

    assert rank_zero != rank_one
    assert rank_zero == _v2_fingerprint(logical_rank=0, **common)

    nemotron = loader.model_copy(update={"model_family": "nemotron"})
    assert rank_zero != _v2_fingerprint(
        logical_rank=0,
        **{**common, "loader_config": nemotron},
    )
    assert rank_zero != _v2_fingerprint(
        logical_rank=0, **{**common, "placement_fingerprint": "other-placement"}
    )


def test_v2_loader_applies_cache_pool_and_gc_controls() -> None:
    adapter = MagicMock(fingerprint="processor")
    task_encoder = MagicMock()
    task_encoder.cookers = [MagicMock(need_cache=True)]
    dataset = object()
    cache_pool = object()
    raw_loader = MagicMock()

    with (
        patch(
            "nemo_rl.data.energon.sft_dataloader.build_processor_adapter",
            return_value=adapter,
        ),
        patch(
            "nemo_rl.data.energon.sft_dataloader._task_encoder",
            return_value=task_encoder,
        ),
        patch(
            "nemo_rl.data.energon.sft_dataloader.get_train_dataset",
            return_value=dataset,
        ) as get_train_dataset,
        patch(
            "nemo_rl.data.energon.sft_dataloader.FileStoreCachePool",
            return_value=cache_pool,
        ) as cache_pool_type,
        patch(
            "nemo_rl.data.energon.sft_dataloader.get_savable_loader",
            return_value=raw_loader,
        ) as get_savable_loader,
    ):
        build_energon_sft_loader(
            data_config={
                "shuffle": True,
                "energon": {
                    "model_family": "qwen",
                    "cache_pool_max_gbytes": 8,
                    "cache_pool_num_workers": 3,
                    "gc_collect_every_n_steps": 1234,
                    "task_encoder": {
                        "packing": {
                            "name": "balanced_greedy_knapsack",
                            "buffer_size": 5000,
                            "options": {
                                "max_sequence_length": 128,
                                "sequence_length_pad_multiple": 8,
                                "balanced_knapsack_delta": 5,
                            },
                        }
                    },
                },
            },
            source=EnergonSourceConfig(
                path="/dataset", split="train", virtual_epoch_length=8
            ),
            processor=MagicMock(tokenizer=MagicMock()),
            batch_size=2,
            max_sequence_length=128,
            split_role="train",
            logical_rank=0,
            logical_world_size=1,
            placement_fingerprint="placement",
            only_unmask_final=False,
        )

    cache_pool_type.assert_called_once_with(
        method="raw", num_workers=3, max_cache_size_gbytes=8.0
    )
    assert get_train_dataset.call_args.kwargs["packing_buffer_size"] == 5000
    assert get_savable_loader.call_args.kwargs["cache_pool"] is cache_pool
    assert get_savable_loader.call_args.kwargs["gc_collect_every_n_steps"] == 1234


def test_sft_v2_worker_uses_megatron_worker_environment() -> None:
    assert get_actor_python_env(
        "nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker"
    ) == get_actor_python_env(
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )


def test_sft_v2_worker_defers_processor_construction() -> None:
    from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker_cls = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    with (
        patch.object(MegatronPolicyWorkerImpl, "__init__", return_value=None),
        patch("nemo_rl.algorithms.utils.get_tokenizer") as get_tokenizer,
    ):
        worker = worker_cls(
            {"tokenizer": {"use_processor": True}}, tokenizer=MagicMock()
        )

    get_tokenizer.assert_not_called()
    assert worker._sft_processor is None

    worker.cfg = {"tokenizer": {"use_processor": True}}
    with patch("nemo_rl.algorithms.utils.get_tokenizer") as get_tokenizer:
        assert worker._ensure_sft_processor() is get_tokenizer.return_value
        assert worker._ensure_sft_processor() is get_tokenizer.return_value
    get_tokenizer.assert_called_once_with(worker.cfg["tokenizer"], get_processor=True)


def test_sft_v2_worker_publishes_sequence_alignment() -> None:
    from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker

    worker_cls = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker._sft_loader = MagicMock()
    worker._sft_loader_iterator = iter([{"message_log": []}])
    worker._sft_active_envelope = None
    worker._sft_logical_rank = 0
    worker._sft_logical_world_size = 2
    worker._sft_mimo_role = None
    worker._sft_single_owner = False
    worker._sft_encoder_dp_client = None
    worker._sft_next_batch_index = 0
    worker._ld_on = False
    worker._ld_phase = None
    worker._ld_t0 = 0.0
    worker._ld_durations = {}
    worker._ld_watchdog = None
    worker.tokenizer = MagicMock(pad_token_id=0)
    worker._dp_client = MagicMock()

    prepared = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "input_lengths": torch.tensor([3, 2], dtype=torch.int32),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "sample_mask": torch.ones(2),
            "source_ids": ["source-a", "source-b"],
        }
    )
    policy_fields = [key for key in prepared if key != "source_ids"]
    worker._dp_client.put_samples.return_value = KVBatchMeta(
        partition_id="sft_v2_dp0_batch0",
        task_name=None,
        sample_ids=["sft_v2_dp0_batch0_row0", "sft_v2_dp0_batch0_row1"],
        fields=policy_fields,
        sequence_lengths=[3, 2],
        extra_info={"generation": 7},
    )

    with patch(
        "nemo_rl.data.energon.sft_worker.prepare_sft_batch",
        return_value=prepared,
    ):
        envelope = worker.load_next_sft_batch(
            only_unmask_final=False,
            make_sequence_length_divisible_by=4,
        )

    assert envelope.meta.extra_info == {
        "generation": 7,
        "pad_to_multiple": 4,
    }
    assert envelope.source_ids == ("source-a", "source-b")
    assert envelope.field_names == tuple(policy_fields)
    assert "source_ids" not in worker._dp_client.put_samples.call_args.kwargs["fields"]
    assert worker._dp_client.put_samples.call_args.kwargs["tags"] == [
        {"source_id": "source-a"},
        {"source_id": "source-b"},
    ]
    assert set(envelope.load_phase_seconds) == {
        "iter",
        "prepare",
        "post-prepare",
        "tensordict",
        "publish",
        "publish_setup",
        "publish_register_partition",
        "publish_source_tags",
        "publish_put_samples",
        "publish_batch_metadata",
    }
    assert all(value >= 0.0 for value in envelope.load_phase_seconds.values())
    assert envelope.load_seconds >= sum(
        envelope.load_phase_seconds[phase]
        for phase in ("iter", "prepare", "post-prepare", "tensordict", "publish")
    )


@pytest.mark.parametrize("mimo", [False, True])
def test_sft_v2_non_owner_loader_lifecycle_is_a_noop(mimo: bool) -> None:
    from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker

    worker_cls = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker._sft_loader = None
    worker._sft_loader_iterator = None
    worker._sft_active_envelope = None
    worker._sft_mimo_role = None
    worker._sft_single_owner = False
    worker.mimo_runtime = object() if mimo else None
    worker._sft_active_validation_envelope = None
    worker._is_replica_leader = MagicMock(return_value=False)
    worker._mimo_loader_role = MagicMock(return_value=None)
    worker._configure_sft_dataloader = MagicMock()

    assert (
        worker.setup_sft_dataloader(
            single_owner=False,
            data_config={},
            batch_size=1,
            max_sequence_length=8,
            placement_fingerprint="placement",
            only_unmask_final=False,
        )
        is None
    )
    worker._configure_sft_dataloader.assert_not_called()

    assert (
        worker.load_next_sft_batch(
            only_unmask_final=False,
            make_sequence_length_divisible_by=1,
        )
        is None
    )
    assert worker.sft_dataloader_state_dict() is None
    assert worker.commit_sft_batch() is None
    assert worker.abort_sft_batch() is None
    assert worker.close_sft_dataloader() is None


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("single_owner", [False, True])
def test_mimo_dp4_dp2_loader_owners_and_lanes(rank: int, single_owner: bool) -> None:
    from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker

    worker_cls = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker._sft_loader = None
    worker._sft_encoder_dp_client = None
    worker._dp_client = MagicMock()
    encoder = rank < 4
    dp_group = MagicMock()
    dp_group.rank.return_value = rank if encoder else (rank - 4) // 2
    dp_group.size.return_value = 4 if encoder else 2
    worker.mimo_runtime = MagicMock()
    worker.mimo_runtime.active_module_name = "images" if encoder else "language"
    worker.mimo_runtime.local_pg_collection.dp = dp_group
    language_grid = worker.mimo_runtime.megatron_mimo_infra.module_to_grid_map[
        "language"
    ]
    language_grid.shape = [1, 2, 2, 1]
    language_grid.dim_names = ["tp", "cp", "dp", "pp"]
    worker.mimo_parallelism_config = MagicMock()
    worker.mimo_parallelism_config.get_parallelism.return_value.data_parallel_size = 2
    worker._is_replica_leader = MagicMock(return_value=encoder or rank % 2 == 0)
    worker._configure_sft_dataloader = MagicMock()

    def module(dp: int, offset: int, cp: int) -> dict[str, int]:
        return {
            "tensor_model_parallel_size": 1,
            "context_parallel_size": cp,
            "pipeline_model_parallel_size": 1,
            "data_parallel_size": dp,
            "rank_offset": offset,
        }

    # The encoder loaders check their shard against the controller's placement.
    worker.cfg = {
        "megatron_cfg": {
            "mimo": {
                "parallelism": {
                    "module_parallelisms": {
                        "images": module(4, 0, 1),
                        "language": module(2, 4, 2),
                    }
                }
            }
        }
    }
    with (
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.get_world_size", return_value=8),
    ):
        result = worker.setup_sft_dataloader(
            single_owner=single_owner,
            data_config={},
            batch_size=2,
            max_sequence_length=2048,
            placement_fingerprint="test",
            only_unmask_final=False,
        )
    # Every encoder DP replica loads its lane; its shard selects its media range.
    if rank in ((0, 1, 2, 3) if single_owner else (0, 1, 2, 3, 4, 6)):
        assert result is True
        args = worker._configure_sft_dataloader.call_args.kwargs
        assert args["logical_rank"] == (rank // 2 if encoder else (rank - 4) // 2)
        assert args["logical_world_size"] == 2
        assert args["encoder_shard"] == (
            EncoderShard(index=rank % 2, count=2) if encoder else None
        )
        from nemo_rl.data_plane.adapters.local import LocalDataPlaneClient

        if single_owner:
            assert isinstance(worker._sft_encoder_dp_client, LocalDataPlaneClient)
            assert worker._sft_encoder_dp_client is not worker._dp_client
        else:
            assert worker._sft_encoder_dp_client is None
        assert args["mimo_role"] == ("encoder" if encoder else "language")
    else:
        assert result is None
        worker._configure_sft_dataloader.assert_not_called()


@pytest.mark.parametrize("release", ["commit_sft_batch", "abort_sft_batch"])
@pytest.mark.parametrize(
    "role", ["canonical", "canonical-shard1", "encoder", "language"]
)
def test_single_owner_keeps_pixels_local_and_releases_both_views(release, role) -> None:
    from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker
    from nemo_rl.data.multimodal_utils import PackedTensor
    from nemo_rl.data_plane.adapters.local import (
        LocalDataPlaneClient,
        is_local_batch_meta,
    )
    from nemo_rl.data_plane.interfaces import LocalDataPlaneConfig

    worker_cls = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker._sft_loader = MagicMock()
    worker._sft_loader_iterator = iter([None])
    worker._sft_active_envelope = None
    worker._sft_next_batch_index = 0
    worker._sft_logical_rank = 0
    worker._sft_logical_world_size = 1
    worker._sft_mimo_role = "encoder" if role.startswith("canonical") else role
    worker._sft_single_owner = role.startswith("canonical")
    # Encoder shard 1 of a lane keeps its pixels and does not publish the language batch.
    worker._sft_encoder_shard = (
        EncoderShard(index=1, count=2) if role == "canonical-shard1" else None
    )
    worker._sft_encoder_dp_client = (
        LocalDataPlaneClient(LocalDataPlaneConfig())
        if role.startswith("canonical")
        else None
    )
    remote = worker._dp_client = MagicMock()
    if not role.startswith("canonical"):
        worker._dp_client = LocalDataPlaneClient(LocalDataPlaneConfig())
    remote.put_samples.side_effect = lambda **kw: KVBatchMeta(
        sample_ids=kw["sample_ids"],
        partition_id=kw["partition_id"],
        task_name="train",
        fields=list(kw["fields"].keys()),
        tags=kw["tags"],
    )
    worker._ld_mark = MagicMock()
    worker.tokenizer = None
    pixels = PackedTensor([torch.ones(2, 3, dtype=torch.bfloat16)], dim_to_pack=0)
    prepared = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "input_lengths": torch.tensor([4]),
            "sample_mask": torch.ones(1),
            "token_mask": torch.ones(1, 4),
            "pixel_values": pixels,
            "cu_seqlens": PackedTensor(
                [torch.tensor([0, 4], dtype=torch.int32)], dim_to_pack=0
            ),
        }
    )
    with patch(
        "nemo_rl.data.energon.sft_worker.prepare_sft_batch", return_value=prepared
    ):
        envelope = worker._load_next_sft_batch(
            validation=False,
            only_unmask_final=False,
            make_sequence_length_divisible_by=2,
        )
    assert is_local_batch_meta(envelope.meta)
    if role == "canonical-shard1":
        assert "_encoder1_" in envelope.meta.partition_id
        assert envelope.role_metas == {}
        remote.put_samples.assert_not_called()
        local = worker._require_dp_client().get_data(envelope.meta)
        assert "pixel_values" in local.keys()
        getattr(worker, release)()
        assert worker._sft_active_envelope is None
        remote.clear_samples.assert_not_called()
        assert not worker._sft_encoder_dp_client._partitions
        return
    if role != "canonical":
        assert envelope.role_metas == {}
        remote.put_samples.assert_not_called()
        local = worker._require_dp_client().get_data(envelope.meta)
        assert ("pixel_values" in local.keys()) == (role == "encoder")
        assert "cu_seqlens" in local.keys()
        getattr(worker, release)()
        assert worker._sft_active_envelope is None
        assert not worker._dp_client._partitions
        return
    language_meta = envelope.role_metas["language"]
    assert not is_local_batch_meta(language_meta)
    assert language_meta.partition_id != envelope.meta.partition_id
    assert language_meta.sequence_lengths == envelope.meta.sequence_lengths
    assert language_meta.extra_info["pad_to_multiple"] == 2
    published = remote.put_samples.call_args.kwargs["fields"]
    assert "pixel_values" not in published.keys()
    assert "cu_seqlens" in published.keys()
    local = worker._require_dp_client().get_data(envelope.meta)
    assert "pixel_values" in local.keys()
    getattr(worker, release)()
    remote.clear_samples.assert_called_once_with(
        sample_ids=language_meta.sample_ids,
        partition_id=language_meta.partition_id,
    )
    assert worker._sft_active_envelope is None
    assert not worker._sft_encoder_dp_client._partitions
