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

"""Inference wrappers for Megatron-core pins that predate the modality flags.

`MegatronPolicyWorker` decides whether to drive the multimodal generation path
by reading `supports_image` / `supports_video` / `supports_audio` off the
configured wrapper class. Megatron-core gained those attributes after some
pinned branches were cut; on those pins every wrapper reads as text-only, so the
worker hands the full multimodal module to code that expects a bare language
model. Point `megatron_inference_wrapper` at a subclass here to restore the
flags without patching the vendored Megatron-core source.
"""

from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)


class VLMInferenceWrapperWithModalities(VLMInferenceWrapper):
    """`VLMInferenceWrapper` with the modality flags newer Megatron-core declares."""

    supports_image = True
    supports_video = True
    supports_audio = False
