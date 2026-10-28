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

"""Image-conditioned PPO critic using the policy's native MIMO runtime."""

import gc
from contextlib import contextmanager
from typing import Any, Optional

import ray
import torch
from megatron.bridge.training.utils.train_utils import LinearForLastLayer

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.mimo import resolve_mimo_sft_config
from nemo_rl.models.megatron.setup import make_policy_like_config
from nemo_rl.models.megatron.value import ValuePostProcessor
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.workers.megatron_value_worker import (
    _install_value_head_load_skip,
)


def install_mimo_value_head(models: list[Any]) -> list[Any]:
    """Replace only the language head before DDP and optimizer construction."""
    for model in models:
        language = model.language_model
        if language is None:
            continue
        language.share_embeddings_and_output_weights = False
        cfg = getattr(language, "config")
        cfg.share_embeddings_and_output_weights = False
        # MIMO installs legacy parallel_state globals only after provider hooks
        # finish. PP=1 is enforced by validate_mimo_parallelism, so create the
        # language head directly with its explicit TP group here.
        language.output_layer = LinearForLastLayer(
            input_size=cfg.hidden_size,
            output_size=1,
            sequence_parallel=cfg.sequence_parallel,
            bias=False,
            tp_group=getattr(getattr(language, "pg_collection", None), "tp", None),
        )
        _install_value_head_load_skip(language)

        # Bridge checks the outer MIMO model when skipping a fresh scalar head
        # during a base-checkpoint load. Its state dict delegates to language.
        @contextmanager
        def hide_loss_modules(language_model: Any = language):
            with language_model.hide_loss_modules():
                yield

        model.hide_loss_modules = hide_loss_modules
    return models


class MegatronMIMOValueWorkerImpl(MegatronPolicyWorkerImpl):
    """Reuse policy MIMO loading, role placement, scheduling, and memory lifecycle."""

    def __init__(self, config: ValueConfig, tokenizer: Any, **kwargs: Any) -> None:
        if resolve_mimo_sft_config(config) is None or not config.get("is_vlm"):
            raise ValueError("MIMO critic requires is_vlm=true and megatron_cfg.mimo.")
        if not kwargs.get("init_optimizer", True):
            raise NotImplementedError("MIMO critic construction requires an optimizer.")
        policy_config = make_policy_like_config(config)
        super().__init__(
            config=policy_config,
            tokenizer=tokenizer,
            init_reference_model=False,
            **kwargs,
        )
        if self.mtp_enabled:
            raise ValueError("MIMO critic requires MTP disabled.")

    def _configure_mimo_model_provider(self) -> None:
        provider = self.megatron_cfg.model
        # Rebuild the derived spec after untying so construction also sees it.
        provider.standard_provider.share_embeddings_and_output_weights = False
        provider.language_model_spec = (
            provider.standard_provider.build_language_model_spec(pp_rank=0)
        )
        provider.register_pre_wrap_hook(install_mimo_value_head)

    def _mimo_loss_post_processor(
        self, loss_fn: LossFunction, num_microbatches: int
    ) -> ValuePostProcessor:
        return ValuePostProcessor(loss_fn=loss_fn, num_microbatches=num_microbatches)

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        sequence_length: Optional[int] = None,
        split_encoder_media: bool = False,
    ) -> dict[str, Any]:
        """Run the complete critic batch as one synchronized MIMO optimizer step."""
        if eval_mode:
            raise NotImplementedError(
                "MIMO critic train(eval_mode=True) is not supported; use get_values."
            )
        self.begin_train_step(loss_fn=loss_fn, gbs=gbs, mbs=mbs)
        try:
            self.train_microbatch(
                data,
                num_microbatches=num_microbatches,
                sequence_length=sequence_length,
                split_encoder_media=split_encoder_media,
            )
            result = self.finish_train_step()
            # The full scalar loss is replicated on CP ranks, while the policy
            # finisher sums losses over DP x CP. Keep critic reporting unique.
            result["global_loss"] /= torch.distributed.get_world_size(
                self.mimo_runtime.local_pg_collection.cp
            )
            return result
        except Exception:
            self.abort_train_step()
            raise

    def get_values(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        sequence_length: Optional[int] = None,
        split_encoder_media: bool = False,
    ) -> BatchedDataDict[Any]:
        """Run the identical image/text forward used by policy, with a scalar head."""
        with torch.no_grad():
            return self._get_mimo_predictions(
                data=data,
                micro_batch_size=micro_batch_size,
                num_microbatches=num_microbatches,
                sequence_length=sequence_length,
                split_encoder_media=split_encoder_media,
                result_key="values",
                post_processing_fn=ValuePostProcessor(),
            )

    def prepare_for_inference(self) -> None:
        """Reload parameters and release training buffers using the policy lifecycle."""
        self.prepare_for_lp_inference()

    def finish_training(self) -> None:
        """Release both language and visual model buffers and optimizer state."""
        self.model.eval()
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=True
        )
        if self.optimizer is not None and not self.optimizer_cpu_offload:
            self.move_optimizer("cpu")
        gc.collect()
        torch.cuda.empty_cache()


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronMIMOValueWorker(MegatronMIMOValueWorkerImpl):
    pass
