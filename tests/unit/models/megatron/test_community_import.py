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

"""Unit tests for community_import."""

import importlib
import os
import pickle
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest


def _ensure_package(monkeypatch, name: str) -> ModuleType:
    """Create a minimal package module in sys.modules for import stubbing."""
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent_module = _ensure_package(monkeypatch, parent_name)
        setattr(parent_module, child_name, module)

    return module


def _load_community_import_module(monkeypatch):
    """Import community_import with lightweight dependency stubs."""
    # Stub torch symbols used at import time/type annotations.
    fake_torch = ModuleType("torch")
    fake_torch.dtype = type("dtype", (), {})
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()
    fake_torch.float16 = object()
    # import_model_from_hf_name guards its collectives on this; single-process
    # tests take the non-distributed path.
    fake_torch.distributed = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Stub megatron imports required by the module.
    _ensure_package(monkeypatch, "megatron")
    _ensure_package(monkeypatch, "megatron.core")
    megatron_bridge = ModuleType("megatron.bridge")
    megatron_bridge.AutoBridge = type("AutoBridge", (), {})
    monkeypatch.setitem(sys.modules, "megatron.bridge", megatron_bridge)
    megatron_transformer = ModuleType("megatron.core.transformer")
    megatron_transformer.ModuleSpec = type("ModuleSpec", (), {})
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", megatron_transformer)

    # Stub MegatronConfig type import.
    nemo_policy = ModuleType("nemo_rl.models.policy")
    nemo_policy.MegatronConfig = dict
    monkeypatch.setitem(sys.modules, "nemo_rl.models.policy", nemo_policy)

    module_name = "nemo_rl.models.megatron.community_import"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _install_runtime_stubs_for_hf_import(monkeypatch):
    """Install minimal megatron-core stubs needed by import_model_from_hf_name."""
    core_module = _ensure_package(monkeypatch, "megatron.core")

    parallel_state = ModuleType("megatron.core.parallel_state")
    parallel_state.model_parallel_is_initialized = lambda: False
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    core_module.parallel_state = parallel_state

    rerun_state_machine = ModuleType("megatron.core.rerun_state_machine")
    rerun_state_machine.destroy_rerun_state_machine = lambda: None
    monkeypatch.setitem(
        sys.modules, "megatron.core.rerun_state_machine", rerun_state_machine
    )
    core_module.rerun_state_machine = rerun_state_machine

    tensor_parallel = ModuleType("megatron.core.tensor_parallel")
    tensor_parallel.model_parallel_cuda_manual_seed = lambda seed: None
    tensor_parallel_random = ModuleType("megatron.core.tensor_parallel.random")
    tensor_parallel_random._CUDA_RNG_STATE_TRACKER = "stale"
    tensor_parallel_random._CUDA_RNG_STATE_TRACKER_INITIALIZED = True
    tensor_parallel.random = tensor_parallel_random
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel)
    monkeypatch.setitem(
        sys.modules, "megatron.core.tensor_parallel.random", tensor_parallel_random
    )
    core_module.tensor_parallel = tensor_parallel


def test_iter_vlm_config_overrides_yields_super35_runtime_values(monkeypatch):
    module = _load_community_import_module(monkeypatch)

    overrides = dict(
        module.iter_vlm_config_overrides(
            {
                "radio_force_eval_mode": False,
                "recompute_vision": True,
                "vision_recompute_granularity": "full",
                "vision_recompute_method": "block",
                "vision_recompute_num_layers": 30,
            }
        )
    )

    assert overrides == {
        "radio_force_eval_mode": False,
        "recompute_vision": True,
        "vision_recompute_granularity": "full",
        "vision_recompute_method": "block",
        "vision_recompute_num_layers": 30,
    }


def _stage_conversion(path) -> None:
    """Materialize conversion config, metadata, and a nonempty tensor shard."""
    os.makedirs(os.path.join(str(path), "iter_0000000"), exist_ok=True)
    with open(os.path.join(str(path), "iter_0000000", "run_config.yaml"), "w") as f:
        f.write("{}\n")
    iteration = os.path.join(str(path), "iter_0000000")
    with open(os.path.join(iteration, "__0_0.distcp"), "wb") as f:
        f.write(b"checkpoint tensors")
    metadata = SimpleNamespace(
        storage_data={"weight": SimpleNamespace(relative_path="__0_0.distcp")}
    )
    with open(os.path.join(iteration, ".metadata"), "wb") as f:
        pickle.dump(metadata, f)


def test_import_model_from_hf_name_calls_bridge_save(monkeypatch, tmp_path):
    module = _load_community_import_module(monkeypatch)
    _install_runtime_stubs_for_hf_import(monkeypatch)
    # Force this import path to stay unavailable even if real megatron modules
    # were preloaded by earlier tests in the same process.
    monkeypatch.setitem(
        sys.modules, "megatron.core.dist_checkpointing.strategies.torch", None
    )

    class FakeProvider:
        def __init__(self):
            self.tensor_model_parallel_size = 1
            self.pipeline_model_parallel_size = 1
            self.context_parallel_size = 1
            self.expert_model_parallel_size = 1
            self.expert_tensor_parallel_size = 1
            self.num_layers_in_first_pipeline_stage = None
            self.num_layers_in_last_pipeline_stage = None
            self.pipeline_dtype = "fp32"

        def finalize(self):
            pass

        def initialize_model_parallel(self, seed):
            self.seed = seed

        def provide_distributed_model(self, wrap_with_ddp, post_wrap_hook):
            config = SimpleNamespace()
            return [SimpleNamespace(config=config)]

    class FakeBridge:
        def __init__(self):
            self.provider = FakeProvider()
            self.saved_model = None
            self.saved_path = None

        def to_megatron_provider(self, load_weights):
            assert load_weights is True
            return self.provider

        def save_megatron_model(self, megatron_model, output_path):
            self.saved_model = megatron_model
            self.saved_path = output_path
            # The real save materializes the checkpoint; publish needs the
            # staged directory (and its completion marker) to exist.
            _stage_conversion(output_path)

    fake_bridge = FakeBridge()

    class FakeAutoBridge:
        @staticmethod
        def from_hf_pretrained(hf_model_name, *args, **kwargs):
            # Keep this test focused on bridge-save flow, not HF API defaults.
            assert hf_model_name == "fake/hf-model"
            return fake_bridge

    monkeypatch.setattr(module, "AutoBridge", FakeAutoBridge)

    output_path = tmp_path / "out"
    module.import_model_from_hf_name("fake/hf-model", str(output_path))

    assert fake_bridge.saved_model is not None
    # The save lands in a hidden staging sibling, then is renamed into place.
    assert os.path.basename(fake_bridge.saved_path).startswith(".out.staging-")
    assert module.megatron_conversion_is_complete(str(output_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


@pytest.mark.parametrize(
    ("occupant", "overwrite", "staged_wins"),
    [
        # The empty-target happy path is covered end to end by
        # test_import_model_from_hf_name_calls_bridge_save.
        # A concurrent producer's complete artifact wins; the staged copy is discarded.
        ("complete", False, False),
        # A stale partial occupant (bare iter_0000000/, interrupted run) is displaced.
        ("partial", False, True),
        # force_reconvert_from_hf replaces even a complete artifact.
        ("complete", True, True),
    ],
)
def test_publish_conversion(monkeypatch, tmp_path, occupant, overwrite, staged_wins):
    """Publish atomically renames the staging dir, resolving occupants of the final path."""
    module = _load_community_import_module(monkeypatch)
    staging, final = tmp_path / ".ckpt.staging-abc", tmp_path / "ckpt"
    _stage_conversion(staging)
    (staging / "staged_marker").touch()
    if occupant == "complete":
        _stage_conversion(final)
    elif occupant == "partial":
        (final / "iter_0000000").mkdir(parents=True)

    module.publish_megatron_conversion(str(staging), str(final), overwrite=overwrite)

    assert module.megatron_conversion_is_complete(str(final))
    assert (final / "staged_marker").exists() is staged_wins
    # Doomed copies are deleted on a background thread; wait for it.
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and sorted(
        p.name for p in tmp_path.iterdir()
    ) != ["ckpt"]:
        time.sleep(0.01)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_metadata",
        "invalid_metadata",
        "empty_metadata",
        "missing_shard",
        "empty_shard",
    ],
)
def test_conversion_requires_all_metadata_referenced_shards(
    monkeypatch, tmp_path, failure
):
    module = _load_community_import_module(monkeypatch)
    _stage_conversion(tmp_path)
    iteration = tmp_path / "iter_0000000"
    if failure == "missing_metadata":
        (iteration / ".metadata").unlink()
    elif failure == "invalid_metadata":
        (iteration / ".metadata").write_bytes(b"invalid pickle")
    elif failure == "empty_metadata":
        (iteration / ".metadata").write_bytes(
            pickle.dumps(SimpleNamespace(storage_data={}))
        )
    elif failure == "missing_shard":
        (iteration / "__0_0.distcp").unlink()
    else:
        (iteration / "__0_0.distcp").write_bytes(b"")
    assert not module.megatron_conversion_is_complete(str(tmp_path))


def _gtp_provider(**fields):
    provider = SimpleNamespace(
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        tensor_parallel_num_weight_shards=1,
        expert_tensor_parallel_num_weight_shards=1,
        gtp_weight_remat_size=1,
        expert_gtp_weight_remat_size=1,
    )
    for key, value in fields.items():
        setattr(provider, key, value)
    return provider


def test_apply_gtp_weight_sharding_sets_shards_from_parallel_size(monkeypatch):
    module = _load_community_import_module(monkeypatch)
    provider = _gtp_provider(expert_tensor_parallel_size=2)

    module._apply_gtp_weight_sharding(
        provider,
        {
            "model_overrides": {
                "gtp_weight_remat_size": 1,
                "expert_gtp_weight_remat_size": 4,
            }
        },
    )

    # shards = parallel size x GTP on each axis; the GTP fields follow.
    assert provider.tensor_parallel_num_weight_shards == 1
    assert provider.gtp_weight_remat_size == 1
    assert provider.expert_tensor_parallel_num_weight_shards == 8
    assert provider.expert_gtp_weight_remat_size == 4


def test_apply_gtp_weight_sharding_ignores_absent_overrides(monkeypatch):
    module = _load_community_import_module(monkeypatch)
    provider = _gtp_provider()

    module._apply_gtp_weight_sharding(provider, {})
    module._apply_gtp_weight_sharding(provider, {"model_overrides": None})

    assert provider.expert_tensor_parallel_num_weight_shards == 1
    assert provider.expert_gtp_weight_remat_size == 1


@pytest.mark.parametrize("bad", [0, -1, "4", 2.0])
def test_apply_gtp_weight_sharding_rejects_invalid_sizes(monkeypatch, bad):
    module = _load_community_import_module(monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        module._apply_gtp_weight_sharding(
            _gtp_provider(), {"model_overrides": {"expert_gtp_weight_remat_size": bad}}
        )


def test_apply_gtp_weight_sharding_requires_provider_support_above_one(monkeypatch):
    module = _load_community_import_module(monkeypatch)
    plain = SimpleNamespace(tensor_model_parallel_size=1, expert_tensor_parallel_size=1)

    # GTP 1 is the default layout and needs no provider field.
    module._apply_gtp_weight_sharding(
        plain, {"model_overrides": {"expert_gtp_weight_remat_size": 1}}
    )
    with pytest.raises(ValueError, match="does not support GTP weight sharding"):
        module._apply_gtp_weight_sharding(
            plain, {"model_overrides": {"expert_gtp_weight_remat_size": 4}}
        )


def test_import_model_from_hf_name_forwards_gtp_and_cp_from_megatron_config(
    monkeypatch, tmp_path
):
    module = _load_community_import_module(monkeypatch)
    _install_runtime_stubs_for_hf_import(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "megatron.core.dist_checkpointing.strategies.torch", None
    )

    class FakeProvider:
        def __init__(self):
            self.tensor_model_parallel_size = 1
            self.pipeline_model_parallel_size = 1
            self.context_parallel_size = 1
            self.expert_model_parallel_size = 1
            self.expert_tensor_parallel_size = 1
            self.tensor_parallel_num_weight_shards = 1
            self.expert_tensor_parallel_num_weight_shards = 1
            self.gtp_weight_remat_size = 1
            self.expert_gtp_weight_remat_size = 1
            self.num_layers_in_first_pipeline_stage = None
            self.num_layers_in_last_pipeline_stage = None
            self.pipeline_dtype = "fp32"
            self.linear_cp_mode = "headwise"
            self.linear_cp_layout = "zigzag"
            self.attention_cp_layout = "zigzag"
            self.fp8 = None
            self.fp8_param = False
            self.finalized_with = None

        def finalize(self):
            # The trainer-facing layout must be set before finalize derives GTP.
            self.finalized_with = (
                self.expert_tensor_parallel_num_weight_shards,
                self.expert_gtp_weight_remat_size,
                self.context_parallel_size,
                self.linear_cp_mode,
                self.linear_cp_layout,
                self.attention_cp_layout,
            )

        def initialize_model_parallel(self, seed):
            pass

        def provide_distributed_model(self, wrap_with_ddp, post_wrap_hook):
            assert self.linear_cp_mode == "chunkwise"
            assert self.linear_cp_layout == "contiguous"
            assert self.attention_cp_layout == "zigzag"
            assert self.context_parallel_size == 64
            assert self.fp8 is None
            assert self.fp8_param is False
            chunks = [SimpleNamespace(config=self)]
            for hook in self._pre_wrap_hooks:
                hook(chunks)
            return chunks

    class FakeBridge:
        def __init__(self):
            self.provider = FakeProvider()

        def to_megatron_provider(self, load_weights):
            assert load_weights is True
            # Bridge registers its HF loader on the original provider instance.
            self.provider._pre_wrap_hooks = [
                lambda chunks: setattr(chunks[0], "weights_loaded", True)
            ]
            return self.provider

        def save_megatron_model(self, megatron_model, output_path):
            assert megatron_model[0].weights_loaded
            _stage_conversion(output_path)

    fake_bridge = FakeBridge()
    monkeypatch.setattr(
        module,
        "AutoBridge",
        SimpleNamespace(from_hf_pretrained=lambda *a, **k: fake_bridge),
    )
    megatron_config = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 64,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "pipeline_dtype": "bf16",
        "sequence_parallel": False,
        "gradient_accumulation_fusion": False,
        "model_overrides": {
            "gtp_weight_remat_size": 1,
            "expert_gtp_weight_remat_size": 4,
            "linear_cp_mode": "chunkwise",
            "linear_cp_layout": "contiguous",
            "attention_cp_layout": "zigzag",
            "fp8": "e4m3",
            "fp8_param": True,
        },
    }

    module.import_model_from_hf_name(
        "fake/hf-model", str(tmp_path / "out"), megatron_config
    )

    provider = fake_bridge.provider
    assert provider.finalized_with == (4, 4, 64, "chunkwise", "contiguous", "zigzag")
    # The restore step resets parallel sizes but must keep the weight-shard layout.
    assert provider.context_parallel_size == 1
    assert provider.expert_tensor_parallel_size == 1
    assert provider.expert_tensor_parallel_num_weight_shards == 4
    assert provider.expert_gtp_weight_remat_size == 4


@pytest.mark.parametrize(
    "config", [{}, {"model_overrides": None}, {"model_overrides": {}}]
)
def test_context_parallel_overrides_preserve_provider_defaults(monkeypatch, config):
    module = _load_community_import_module(monkeypatch)
    provider = SimpleNamespace(
        linear_cp_mode="headwise",
        linear_cp_layout="zigzag",
        attention_cp_layout="zigzag",
    )
    expected = vars(provider).copy()
    module._apply_context_parallel_overrides(provider, config)
    assert vars(provider) == expected


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("linear_cp_mode", "chunkwise"),
        ("linear_cp_layout", "contiguous"),
        ("attention_cp_layout", "zigzag"),
    ],
)
def test_context_parallel_overrides_reject_unsupported_provider(
    monkeypatch, key, value
):
    module = _load_community_import_module(monkeypatch)
    with pytest.raises(ValueError, match=key):
        module._apply_context_parallel_overrides(
            SimpleNamespace(), {"model_overrides": {key: value}}
        )
