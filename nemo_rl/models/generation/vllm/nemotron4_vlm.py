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
"""Register NT4 (Nemotron-4) as a multimodal model in vLLM.

vLLM ships NT4 as text only: the registry maps
``Nemotron4ForConditionalGeneration`` into ``_TEXT_GENERATION_MODELS`` as
``('nemotron4', 'Nemotron4ForCausalLM')``, and ``nemotron4.py`` contains no
vision, projector or image symbols at all. ``ModelConfig`` therefore reports
``is_multimodal_model=False`` and the Pixtral tower in the checkpoint is never
built, so an image-bearing request is served as if the images were not there.

NT4's HF export is Mistral3-shaped, which is what makes this cheap. Its weights
are named ``model.vision_tower.*``, ``model.multi_modal_projector.{norm,
patch_merger.merging_layer,linear_1,linear_2}`` and ``model.language_model.*``,
which is exactly what ``Mistral3ForConditionalGeneration.hf_to_vllm_mapper``
rewrites, and its vision config is a ``PixtralVisionConfig`` with
``spatial_merge_size: 2`` and ``multimodal_projector_bias: False``, matching
Mistral3's projector including the hardcoded ``RMSNorm(eps=1e-5)``
(NT4 declares ``projector_norm_eps: 1e-05``).

Two config-level differences are handled in the model directory rather than
here, because they are facts about the checkpoint and not behaviour: NT4 names
the image token ``image_token_id`` where vLLM reads ``image_token_index``, and
its ``text_config.architectures`` is null, which is what
``init_vllm_registered_model`` resolves the inner language model from. See
``.scratch-assemble-nt4-vllm-dir.sh``. With those present, vLLM builds the
Nemotron4 language model on its own -- ``Nemotron4ForCausalLM`` reads
``model_config.hf_text_config``, which is that block -- so no model ``__init__``
override is needed.

Two things do need overriding. First, two ``isinstance(..., Mistral3Config)``
assertions on the processing path, since NT4's config is a ``Nemotron4Config``;
neither guards anything NT4 lacks, the fields actually read being
``spatial_merge_size``, ``vision_config.patch_size`` and
``vision_config.num_hidden_layers``. Second, NT4's language model is a hybrid
Mamba/attention stack while Mistral3 is pure attention, so the class has to
carry vLLM's ``IsHybrid`` interface or the Mamba state caches are never sized.
"""

import math
from collections.abc import Mapping
from typing import Any

NEMOTRON4_VLM_ARCH = "Nemotron4ForConditionalGeneration"
NEMOTRON4_TEXT_ARCH = "Nemotron4ForCausalLM"
_MODEL_CLS_NAME = "Nemotron4VLMForConditionalGeneration"

_model_cls: type | None = None


def _build_nemotron4_vlm_cls() -> type:
    """Build the multimodal NT4 class on top of vLLM's Mistral3 implementation."""
    from vllm.model_executor.models.interfaces import IsHybrid
    from vllm.model_executor.models.mistral3 import (
        Mistral3DummyInputsBuilder,
        Mistral3ForConditionalGeneration,
        Mistral3HFEncoderInfo,
        Mistral3MultiModalProcessor,
        Mistral3ProcessingInfo,
    )
    from vllm.model_executor.models.nemotron4 import Nemotron4ForCausalLM
    from vllm.multimodal import MULTIMODAL_REGISTRY

    class _Nemotron4HFEncoderInfo(Mistral3HFEncoderInfo):
        """Mistral3's encoder info without its config type assertion."""

        def get_patch_grid_size(
            self, *, image_width: int, image_height: int
        ) -> tuple[int, int]:
            max_size = self.get_image_size()
            ratio = max(image_width / max_size, image_height / max_size)
            if ratio > 1:
                image_width = math.floor(image_width / ratio)
                image_height = math.floor(image_height / ratio)

            merged_patch_size = (
                self.vision_config.patch_size * self.hf_config.spatial_merge_size
            )
            ncols = (image_width - 1) // merged_patch_size + 1
            nrows = (image_height - 1) // merged_patch_size + 1
            return ncols, nrows

    class _Nemotron4ProcessingInfo(Mistral3ProcessingInfo):
        def get_hf_config(self) -> Any:
            # Mistral3 calls ctx.get_hf_config(Mistral3Config), which asserts the
            # type. Read the config directly instead; the fields the processing
            # path uses (image_token_index, spatial_merge_size, vision_config)
            # are all present on Nemotron4Config.
            return self.ctx.model_config.hf_config

        def get_vision_encoder_info(
            self, mm_processor_kwargs: Mapping[str, object] | None = None
        ) -> Any:
            processor = self.get_hf_processor()
            size = processor.image_processor.size
            merged_kwargs = self.ctx.get_merged_mm_kwargs(mm_processor_kwargs or {})
            if override_size := merged_kwargs.get("size"):
                size = size | override_size
            return _Nemotron4HFEncoderInfo(self.get_hf_config(), size["longest_edge"])

    @MULTIMODAL_REGISTRY.register_processor(
        Mistral3MultiModalProcessor,
        info=_Nemotron4ProcessingInfo,
        dummy_inputs=Mistral3DummyInputsBuilder,
    )
    class Nemotron4VLMForConditionalGeneration(
        Mistral3ForConditionalGeneration, IsHybrid
    ):
        """NT4 Nano as a VLM.

        The vision tower, the 2x2 patch merger, the projector and the
        HF-to-vLLM weight mapping are all Mistral3's, and the inner language
        model is resolved from ``text_config.architectures``. What Mistral3
        cannot supply is the hybrid Mamba plumbing below.
        """

        # NT4's language model interleaves gated-delta-product layers with
        # attention, and vLLM decides whether to size the Mamba state caches
        # from `is_hybrid` on the *top-level* class (ModelRegistry records it
        # into _ModelInfo at inspection time). Mistral3 is pure attention, so
        # without inheriting IsHybrid the engine leaves
        # `cache_config.mamba_block_size` unset and the GDP layers inside the
        # language model hit `assert mamba_block_size is not None` while
        # building the KV cache spec -- which is how job 7770949 died, after
        # loading all 181GB.
        #
        # The three calculators below are vLLM's own, delegated rather than
        # reimplemented: they read `model_config.hf_text_config`, which is the
        # same nested nemotron4_text block our language model is built from, so
        # they are already correct for this model.
        @classmethod
        def get_mamba_state_dtype_from_config(cls, vllm_config: Any) -> Any:
            return Nemotron4ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

        @classmethod
        def get_mamba_state_shape_from_config(cls, vllm_config: Any) -> Any:
            return Nemotron4ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

        @classmethod
        def get_mamba_state_copy_func(cls) -> Any:
            return Nemotron4ForCausalLM.get_mamba_state_copy_func()

    return Nemotron4VLMForConditionalGeneration


def _get_model_cls() -> type:
    """Build the class once per process.

    Caching is required, not just an optimization: the class carries a
    ``MULTIMODAL_REGISTRY.register_processor`` registration keyed by the class
    object, so handing out a fresh class per call would register a new processor
    each time and leave the engine holding a class the registry does not know.
    """
    global _model_cls
    if _model_cls is None:
        _model_cls = _build_nemotron4_vlm_cls()
    return _model_cls


def __getattr__(name: str) -> Any:
    """Expose the model class as a module attribute, built on first access.

    vLLM resolves string-registered models as ``<module>:<class>`` by importing
    the module and getattr-ing the class, so the class has to be reachable this
    way. Doing it through PEP 562 rather than at module scope keeps ``import
    nemo_rl...nemotron4_vlm`` from importing vLLM, which matters because
    ``patches.py`` is imported in processes that never build an engine.
    """
    if name != _MODEL_CLS_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _get_model_cls()


def register_nemotron4_vlm(logger) -> bool:
    """Re-register NT4 as multimodal, replacing vLLM's text-only entry.

    Must run before the engine builds its ``ModelConfig``: whether a model is
    multimodal is decided there, from the resolved class.

    Returns True when the architecture is registered as multimodal.
    """
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except ImportError:
        logger.warning(
            "Could not import vLLM's ModelRegistry; leaving %s as vLLM ships it. "
            "NT4 image inputs will be ignored.",
            NEMOTRON4_VLM_ARCH,
        )
        return False

    try:
        _get_model_cls()
    except Exception as exc:
        # The Mistral3 symbols this builds on are vLLM-version-dependent, so a
        # rename upstream should name itself rather than surface later as a
        # model that silently drops its images.
        logger.error(
            "Could not build the multimodal %s class from vLLM's Mistral3 "
            "implementation (%s: %s). NT4 would be served text-only, with image "
            "inputs ignored.",
            NEMOTRON4_VLM_ARCH,
            type(exc).__name__,
            exc,
        )
        return False

    # Register by `<module>:<class>` rather than passing the class object. The
    # engine's TP workers are started with `spawn`, so they do not inherit this
    # registration; the string form makes vLLM re-import per process. Passing the
    # class also cannot work here, since it is defined inside a function and so
    # has no importable qualified name for the workers to resolve.
    ModelRegistry.register_model(NEMOTRON4_VLM_ARCH, f"{__name__}:{_MODEL_CLS_NAME}")

    # vLLM registers exactly one Nemotron4 entry, keyed by the architecture
    # checkpoints declare:
    #
    #   "Nemotron4ForConditionalGeneration": ("nemotron4", "Nemotron4ForCausalLM")
    #
    # There is no key for the text-only class itself, because until now nothing
    # asked for it by name. We do: the assembled config sets
    # `text_config.architectures = ["Nemotron4ForCausalLM"]`, and Mistral3 builds
    # its language model with `init_vllm_registered_model(hf_config=text_config)`,
    # which resolves that name. Unregistered, the lookup fell back to the
    # conditional-generation entry -- now ours -- so the multimodal class was
    # instantiated a second time with the *text* config and died on
    # `projector_hidden_act` (job 7771738).
    if NEMOTRON4_TEXT_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            NEMOTRON4_TEXT_ARCH,
            "vllm.model_executor.models.nemotron4:Nemotron4ForCausalLM",
        )
    logger.info(
        "Registered %s as multimodal (Pixtral tower + 2x2 patch merger + "
        "projector from vLLM's Mistral3 implementation, Nemotron4 language "
        "model from text_config.architectures), replacing vLLM's text-only "
        "entry.",
        NEMOTRON4_VLM_ARCH,
    )
    return True
