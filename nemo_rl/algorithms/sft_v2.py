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

"""Single-controller SFT with one colocated Energon loader per DP replica."""

from __future__ import annotations

import os
import statistics
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import ray
import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
from nemo_rl.algorithms.sft import SFTConfig
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.energon.config import EnergonSourceConfig
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data.energon.topology import (
    DataLoaderPlacementPlan,
    LoaderCopy,
    build_mimo_data_loader_placement,
    resolve_topology_mapper,
)
from nemo_rl.data.packing import PackingAlgorithm
from nemo_rl.data_plane.interfaces import DataPlaneRuntimeConfig, LocalDataPlaneConfig
from nemo_rl.distributed.named_sharding import REPLICATED_AXES
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    get_node_block_counts,
    node_block_constraints,
    prepare_segment_topology,
)
from nemo_rl.models.megatron.alignment import (
    get_fp8_token_alignment,
    get_parallel_token_alignment,
)
from nemo_rl.models.megatron.mimo import resolve_mimo_sft_config
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.telemetry.config import TelemetryConfig
from nemo_rl.telemetry.instrumentation import (
    accepts_trace_context,
    managed_span,
    trace_context_kwargs,
    umbrella_span,
    umbrella_trace_fn,
)
from nemo_rl.telemetry.setup import init_telemetry_worker, shutdown_telemetry
from nemo_rl.telemetry.span_groups import RLSpanGroup
from nemo_rl.telemetry.vocabulary import TeedMetric, register_teed_metrics
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.timer import TimeoutChecker

LOADER_LATENCY_MAX_KEY = "loader_latency_max"
POLICY_TIME_KEY = "policy_time"
TOTAL_STEP_TIME_KEY = "total_step_time"
VALID_TOKENS_PER_SECOND_KEY = "valid_tokens_per_second"

SFT_V2_TEED_METRICS = (
    TeedMetric(
        LOADER_LATENCY_MAX_KEY,
        "rl.sft_v2.loader.latency.max",
        unit="s",
        description="Slowest data-loader rank per SFT v2 step.",
    ),
    TeedMetric(
        POLICY_TIME_KEY,
        "rl.sft_v2.policy.duration",
        unit="s",
        description="Controller policy update duration per SFT v2 step.",
    ),
    TeedMetric(
        TOTAL_STEP_TIME_KEY,
        "rl.sft_v2.step.duration",
        unit="s",
        description="Wall-clock duration of an SFT v2 step.",
    ),
    TeedMetric(
        VALID_TOKENS_PER_SECOND_KEY,
        "rl.sft_v2.valid_tokens_per_second",
        unit="{token}/s",
        description="Supervised token throughput per SFT v2 step.",
    ),
)
register_teed_metrics(SFT_V2_TEED_METRICS)


class MasterConfig(BaseModel, extra="allow"):
    """Standalone SFTv2 configuration."""

    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    data_plane: DataPlaneRuntimeConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    telemetry: Optional[TelemetryConfig] = None


@dataclass
class SFTV2SaveState:
    """Controller state committed at an optimizer-step boundary."""

    total_steps: int
    consumed_samples: int
    total_valid_tokens: int
    placement_hash: str


@dataclass
class SFTV2ActorArgs:
    """Driver-built objects used by the SFTv2 controller actor."""

    trainer: TQPolicy
    loss_fn: NLLLossFn
    train_cluster: RayVirtualCluster
    placement_plan: DataLoaderPlacementPlan
    save_state: SFTV2SaveState
    loader_states: list[dict[str, Any]] | None


def _initial_save_state(placement_hash: str) -> SFTV2SaveState:
    return SFTV2SaveState(
        total_steps=0,
        consumed_samples=0,
        total_valid_tokens=0,
        placement_hash=placement_hash,
    )


def _restore_save_state(
    loaded: Optional[dict[str, Any]], *, placement_hash: str
) -> SFTV2SaveState:
    if loaded is None:
        return _initial_save_state(placement_hash)
    saved_hash = loaded.get("placement_hash")
    if saved_hash != placement_hash:
        raise ValueError(
            "SFTv2 loader placement changed since the checkpoint was written: "
            f"saved={saved_hash!r}, current={placement_hash!r}."
        )
    defaults = vars(_initial_save_state(placement_hash))
    known = {item.name for item in fields(SFTV2SaveState)}
    defaults.update({key: value for key, value in loaded.items() if key in known})
    return SFTV2SaveState(**defaults)


def _max_train_steps(master_config: MasterConfig) -> int:
    source = EnergonSourceConfig.model_validate(master_config.data["train"])
    # The loader rejects a non-positive value too, but only once the actor builds
    # it -- by then the cluster is up and megatron_cfg.train_iters has already
    # been set from this result. Fail on the driver, before any allocation.
    if source.virtual_epoch_length <= 0:
        raise ValueError(
            "Energon training requires data.train.virtual_epoch_length in batches "
            "(optimizer steps per virtual epoch); it drives both the epoch budget "
            "and megatron_cfg.train_iters."
        )
    return min(
        master_config.sft.max_num_steps,
        master_config.sft.max_num_epochs * source.virtual_epoch_length,
    )


def _validate_mimo_topology_mapper(mapper_name: str) -> None:
    """Reject mapper selectors that the fixed initial MIMO placement cannot honor."""
    if mapper_name != "default":
        raise ValueError(
            "MIMO SFT v2 currently requires "
            "data.energon.topology_mapper='default'; "
            f"got {mapper_name!r}."
        )


def check_mimo_lanes_agree(
    copies: Sequence[LoaderCopy], envelopes: Sequence[StepEnvelope]
) -> None:
    """Require all loaders of a lane (encoder shards, language) to make one batch."""
    first_by_lane: dict[int, StepEnvelope] = {}
    for copy, envelope in zip(copies, envelopes, strict=True):
        if envelope.logical_rank != copy.logical_rank:
            raise RuntimeError(
                f"A MiMo loader of lane {copy.logical_rank} returned lane "
                f"{envelope.logical_rank}."
            )
        first = first_by_lane.setdefault(copy.logical_rank, envelope)
        if (
            envelope.source_ids != first.source_ids
            or envelope.meta.tags != first.meta.tags
            or envelope.sequence_lengths != first.sequence_lengths
            or envelope.valid_tokens != first.valid_tokens
        ):
            raise RuntimeError(
                "Independent MiMo loaders diverged on logical lane "
                f"{copy.logical_rank}: sources, packs, or token counts differ."
            )


def mimo_accounting_envelopes(
    copies: Sequence[LoaderCopy],
    envelopes: Sequence[StepEnvelope],
    *,
    single_owner: bool,
) -> list[StepEnvelope]:
    """Return one envelope per logical lane for sample and token counts."""
    if not single_owner:
        return [envelope for envelope in envelopes if envelope.mimo_role == "language"]
    # Encoder shards of a lane load the same batch; shard 0 counts for the lane.
    return [
        envelope
        for copy, envelope in zip(copies, envelopes, strict=True)
        if copy.encoder_shard in (None, 0)
    ]


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SFTSingleControllerActor:
    """Drive colocated loaders and the existing TQPolicy from one actor."""

    def __init__(self, master_config: MasterConfig, actor_args: SFTV2ActorArgs) -> None:
        telemetry = init_telemetry_worker(
            rank=0, world_size=1, worker_group="sft_v2_controller"
        )
        self._tracer = telemetry.tracer if telemetry is not None else None
        self._master_config = master_config
        self._trainer = actor_args.trainer
        self._loss_fn = actor_args.loss_fn
        self._train_cluster = actor_args.train_cluster
        self._placement_plan = actor_args.placement_plan
        self._save_state = actor_args.save_state
        self._max_steps = _max_train_steps(master_config)
        self._loader_states = actor_args.loader_states
        self._logger = Logger(master_config.logger)  # type: ignore[arg-type]
        self._logger.log_hyperparams(master_config.model_dump())
        self._checkpointer = CheckpointManager(master_config.checkpointing)
        # Also built here, not on the driver: TimeoutChecker starts its
        # wall clock in __init__, so a driver-built one would count the
        # cluster and policy setup against the training budget.
        self._timeout = TimeoutChecker(
            timeout=master_config.checkpointing.get("checkpoint_must_save_by"),
            fit_last_save_time=True,
        )
        self._timeout.start_iterations()
        self._setup_loaders()

    @accepts_trace_context
    def run_with_trace_context(self) -> dict[str, Any]:
        """Run under the driver's trace context when called through Ray."""
        return self.run()

    def run(self) -> dict[str, Any]:
        """Run SFT training."""
        try:
            with umbrella_span(RLSpanGroup.U_JOB, "rl.sft_v2.job", tracer=self._tracer):
                self._trainer.prepare_for_training()
                if (
                    self._master_config.sft.val_at_start
                    and self._save_state.total_steps == 0
                ):
                    self._log_validation()
                while self._save_state.total_steps < self._max_steps:
                    metrics = self._run_train_step()
                    self._logger.log_metrics(metrics, self._save_state.total_steps)
                    if self._should_validate():
                        self._log_validation()
                    metric = self._checkpoint_metric(metrics)
                    self._timeout.mark_iteration()
                    save_by_timeout = self._timeout.check_save()
                    if self._should_save(save_by_timeout=save_by_timeout):
                        self._save_checkpoint(metric)
                    if save_by_timeout:
                        # check_save fires once and then latches, so continuing
                        # would train unsaved until the walltime kill.
                        print("Timeout has been reached, stopping training early")
                        break
                return vars(self._save_state).copy()
        finally:
            for cleanup, name in (
                (self._close_loaders, "loader close"),
                (self._logger.finish, "logger close"),
                (self._checkpointer.shutdown, "checkpoint close"),
                (shutdown_telemetry, "telemetry flush"),
            ):
                try:
                    cleanup()
                except Exception as error:  # teardown must preserve the run failure
                    warnings.warn(f"SFTv2 {name} failed: {error}", stacklevel=2)

    def _setup_loaders(self) -> None:
        config = self._master_config
        common_kwargs = {
            "data_config": config.data,
            "batch_size": config.policy["train_global_batch_size"]
            // self._placement_plan.logical_world_size,
            "max_sequence_length": config.data["max_input_seq_length"],
            "placement_fingerprint": self._placement_plan.placement_hash,
            "only_unmask_final": config.sft.only_unmask_final,
            "single_owner": bool(
                self._trainer.mimo_config is not None
                and self._trainer.mimo_config.single_owner_loader
            ),
        }
        if self._loader_states is None:
            futures = self._trainer.worker_group.run_all_workers_single_data(
                "setup_sft_dataloader",
                run_rank_0_only_axes=(
                    list(REPLICATED_AXES) if self._trainer.mimo_config is None else None
                ),
                **common_kwargs,
            )
        else:
            if len(self._loader_states) != len(self._placement_plan.copies):
                raise ValueError(
                    "The checkpoint must contain one state per loader owner."
                )
            restored_by_rank: list[dict[str, Any] | None] = [None] * (
                self._train_cluster.world_size()
            )
            for copy, state in zip(
                self._placement_plan.copies, self._loader_states, strict=True
            ):
                restored_by_rank[copy.owner_rank] = state
            futures = self._trainer.worker_group.run_all_workers_multiple_data(
                "setup_sft_dataloader",
                run_rank_0_only_axes=(
                    list(REPLICATED_AXES) if self._trainer.mimo_config is None else None
                ),
                restored_state=(
                    self._loader_states
                    if self._trainer.mimo_config is None
                    else restored_by_rank
                ),
                common_kwargs=common_kwargs,
            )
        results = self._ordered_owner_results(
            ray.get(futures), operation="loader setup"
        )
        if results != [True] * len(self._placement_plan.copies):
            raise RuntimeError(f"Unexpected SFT loader setup results: {results!r}.")
        if config.data.get("validation") is not None:
            val_global_batch_size = config.sft.val_global_batch_size
            if val_global_batch_size % self._placement_plan.logical_world_size:
                raise ValueError(
                    "sft.val_global_batch_size must be divisible by the logical "
                    f"DP size: {val_global_batch_size} % "
                    f"{self._placement_plan.logical_world_size} != 0."
                )
            futures = self._trainer.worker_group.run_all_workers_single_data(
                "setup_sft_validation_dataloader",
                run_rank_0_only_axes=list(REPLICATED_AXES),
                data_config=config.data,
                batch_size=(
                    val_global_batch_size // self._placement_plan.logical_world_size
                ),
                max_sequence_length=config.data["max_input_seq_length"],
                placement_fingerprint=self._placement_plan.placement_hash,
                only_unmask_final=config.sft.only_unmask_final,
            )
            results = ray.get(futures)
            if results != [True] * self._placement_plan.logical_world_size:
                raise RuntimeError(
                    f"Unexpected SFT validation loader setup results: {results!r}."
                )

    def _load_envelopes(self) -> list[StepEnvelope]:
        futures = self._trainer.worker_group.run_all_workers_single_data(
            "load_next_sft_batch",
            run_rank_0_only_axes=(
                list(REPLICATED_AXES) if self._trainer.mimo_config is None else None
            ),
            only_unmask_final=self._master_config.sft.only_unmask_final,
            make_sequence_length_divisible_by=self._master_config.policy[
                "make_sequence_length_divisible_by"
            ],
            **trace_context_kwargs(),
        )
        envelopes = cast(
            list[StepEnvelope],
            self._ordered_owner_results(ray.get(futures), operation="batch loading"),
        )
        if self._trainer.mimo_config is not None:
            check_mimo_lanes_agree(self._placement_plan.copies, envelopes)
        return envelopes

    @umbrella_trace_fn(RLSpanGroup.U_STEP, "rl.sft_v2.step")
    def _run_train_step(self) -> dict[str, Any]:
        started = time.monotonic()
        envelopes = self._load_envelopes()
        loader_wait = time.monotonic() - started
        train_started = time.monotonic()
        accounting_envelopes = (
            envelopes
            if self._trainer.mimo_config is None
            else mimo_accounting_envelopes(
                self._placement_plan.copies,
                envelopes,
                single_owner=self._trainer.mimo_config.single_owner_loader,
            )
        )
        step_open = False
        try:
            train_metas = [envelope.meta for envelope in envelopes]
            delivery_ranks = None
            if self._trainer.mimo_config is not None:
                delivery_ranks = [
                    copy.delivery_ranks for copy in self._placement_plan.copies
                ]
                for copy, envelope in zip(
                    self._placement_plan.copies, envelopes, strict=True
                ):
                    for role, ranks in copy.role_delivery_ranks.items():
                        train_metas.append(envelope.role_metas[role])
                        delivery_ranks.append(ranks)
            with managed_span(
                RLSpanGroup.POLICY_UPDATE,
                "rl.sft_v2.policy_training",
                tracer=self._tracer,
            ):
                phase_started = time.monotonic()
                self._trainer.begin_train_step(self._loss_fn)
                begin_train_step = time.monotonic() - phase_started
                step_open = True
                phase_started = time.monotonic()
                placed_phases = self._trainer.train_placed_microbatches(
                    train_metas,
                    delivery_ranks=delivery_ranks,
                )
                train_placed_microbatches = time.monotonic() - phase_started
                phase_started = time.monotonic()
                train_results = self._trainer.finish_train_step()
                finish_train_step = time.monotonic() - phase_started
                step_open = False
                phase_started = time.monotonic()
            policy_seconds = time.monotonic() - train_started
            self._loader_call("commit_sft_batch")
            commit_sft_batch = time.monotonic() - phase_started
        except Exception:
            if step_open:
                try:
                    self._trainer.abort_train_step()
                except Exception as error:  # preserve the policy-step failure
                    warnings.warn(f"SFTv2 policy abort failed: {error}", stacklevel=2)
            try:
                self._loader_call("abort_sft_batch")
            except Exception as error:  # preserve the policy-step failure
                warnings.warn(f"SFTv2 loader abort failed: {error}", stacklevel=2)
            raise

        valid_tokens = sum(envelope.valid_tokens for envelope in accounting_envelopes)
        self._save_state.total_steps += 1
        self._save_state.consumed_samples += sum(
            len(envelope.source_ids) for envelope in accounting_envelopes
        )
        self._save_state.total_valid_tokens += valid_tokens
        loader_seconds = [envelope.load_seconds for envelope in envelopes]
        loader_latency_max = max(loader_seconds)
        metrics: dict[str, Any] = {
            LOADER_LATENCY_MAX_KEY: loader_latency_max,
            "loader_latency_mean": statistics.fmean(loader_seconds),
            "loader_copy_imbalance": loader_latency_max - min(loader_seconds),
            POLICY_TIME_KEY: policy_seconds,
            "begin_train_step": begin_train_step,
            "train_placed_microbatches": train_placed_microbatches,
            "finish_train_step": finish_train_step,
            "commit_sft_batch": commit_sft_batch,
            "loader_wait": loader_wait,
            "queue_depth": 1,
            TOTAL_STEP_TIME_KEY: time.monotonic() - started,
            "valid_tokens": valid_tokens,
            "source_samples": sum(
                len(envelope.source_ids) for envelope in accounting_envelopes
            ),
            "physical_packs": sum(
                len(envelope.meta.sample_ids) for envelope in accounting_envelopes
            ),
            VALID_TOKENS_PER_SECOND_KEY: valid_tokens
            / max(time.monotonic() - started, 1e-12),
        }
        if isinstance(placed_phases, dict):
            for key, value in placed_phases.items():
                metrics[f"placed_{key}"] = float(value)
        phase_names = sorted(
            {phase for envelope in envelopes for phase in envelope.load_phase_seconds}
        )
        for phase in phase_names:
            values = [
                envelope.load_phase_seconds.get(phase, 0.0) for envelope in envelopes
            ]
            metric_phase = phase.replace("-", "_")
            metrics[f"loader_{metric_phase}_max"] = max(values)
            metrics[f"loader_{metric_phase}_mean"] = statistics.fmean(values)
        metrics.update(self._policy_metrics(train_results))
        return metrics

    def _should_validate(self) -> bool:
        step = self._save_state.total_steps
        config = self._master_config.sft
        return bool(
            (config.val_period > 0 and step % config.val_period == 0)
            or (config.val_at_end and step == self._max_steps)
        )

    def _log_validation(self) -> None:
        step = self._save_state.total_steps
        print(f"Running SFT validation at step {step}", flush=True)
        started = time.monotonic()
        metrics = self._run_validation()
        self._logger.log_metrics(metrics, step, prefix="validation")
        self._logger.log_metrics(
            {"total_time": time.monotonic() - started},
            step,
            prefix="timing/validation",
        )
        print(
            "Validation step=%d loss=%.6f answer_token_ce=%.6f "
            "answer_exact_match_accuracy=%.6f samples=%d"
            % (
                step,
                metrics["loss"],
                metrics.get("answer_token_ce", float("nan")),
                metrics.get("answer_exact_match_accuracy", float("nan")),
                metrics["num_valid_samples"],
            ),
            flush=True,
        )
        self._trainer.prepare_for_training()

    def _run_validation(self) -> dict[str, Any]:
        self._owner_call("reset_sft_validation_dataloader")
        totals = {
            "loss_nll_sum": 0.0,
            "num_unmasked_tokens": 0.0,
            "answer_nll_sum": 0.0,
            "answer_exact_match_count": 0.0,
            "num_answer_tokens": 0.0,
            "num_answer_sequences": 0.0,
            "num_valid_samples": 0.0,
        }
        batch_count = 0
        while True:
            if (
                self._master_config.sft.val_batches > 0
                and batch_count >= self._master_config.sft.val_batches
            ):
                break
            futures = self._trainer.worker_group.run_all_workers_single_data(
                "load_next_sft_validation_batch",
                run_rank_0_only_axes=list(REPLICATED_AXES),
                only_unmask_final=self._master_config.sft.only_unmask_final,
                make_sequence_length_divisible_by=self._master_config.policy[
                    "make_sequence_length_divisible_by"
                ],
            )
            envelopes = ray.get(futures)
            if all(envelope is None for envelope in envelopes):
                break
            if any(envelope is None for envelope in envelopes):
                self._owner_call("abort_sft_validation_batch")
                raise RuntimeError(
                    "SFT validation loaders exhausted on different DP batches."
                )
            placed = cast(list[StepEnvelope], envelopes)
            logical_ranks = [envelope.logical_rank for envelope in placed]
            expected = list(range(self._placement_plan.logical_world_size))
            if logical_ranks != expected:
                self._owner_call("abort_sft_validation_batch")
                raise RuntimeError(
                    "SFT validation envelopes arrived for ranks "
                    f"{logical_ranks}; expected {expected}."
                )
            try:
                result = self._trainer.evaluate_placed_microbatches(
                    [envelope.meta for envelope in placed],
                    self._loss_fn,
                    gbs=sum(len(envelope.meta.sample_ids) for envelope in placed),
                    mbs=self._master_config.sft.val_micro_batch_size,
                )
                batch_metrics = self._policy_metrics(result)
                tokens = float(batch_metrics["num_unmasked_tokens"])
                totals["loss_nll_sum"] += float(result["loss"]) * tokens
                totals["num_unmasked_tokens"] += tokens
                for key in (
                    "answer_nll_sum",
                    "answer_exact_match_count",
                    "num_answer_tokens",
                    "num_answer_sequences",
                ):
                    totals[key] += float(batch_metrics.get(key, 0.0))
                # ``num_valid_samples`` from the loss counts physical rows. With
                # sequence packing, one row can contain several source samples,
                # so it under-reports validation coverage. The loader envelope
                # retains the source ids before packing and is authoritative.
                totals["num_valid_samples"] += sum(
                    len(envelope.source_ids) for envelope in placed
                )
                self._owner_call("commit_sft_validation_batch")
            except Exception:
                self._owner_call("abort_sft_validation_batch")
                raise
            batch_count += 1

        if totals["num_unmasked_tokens"] == 0:
            raise RuntimeError("SFT validation produced no supervised tokens.")
        metrics = {
            "loss": totals["loss_nll_sum"] / totals["num_unmasked_tokens"],
            "num_batches": batch_count,
            **{key: value for key, value in totals.items() if key != "loss_nll_sum"},
        }
        if totals["num_answer_tokens"]:
            metrics["answer_token_ce"] = (
                totals["answer_nll_sum"] / totals["num_answer_tokens"]
            )
        if totals["num_answer_sequences"]:
            metrics["answer_exact_match_accuracy"] = (
                totals["answer_exact_match_count"] / totals["num_answer_sequences"]
            )
        return metrics

    @staticmethod
    def _policy_metrics(train_results: dict[str, Any]) -> dict[str, Any]:
        """Convert policy output into flat scalar logger metrics."""
        metrics: dict[str, Any] = {
            "loss": float(train_results["loss"]),
            "grad_norm": float(train_results["grad_norm"]),
        }
        for key, values in train_results.get("all_mb_metrics", {}).items():
            if key in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                metrics[key] = np.mean(values).item()
            else:
                metrics[key] = np.sum(values).item()
        answer_tokens = metrics.get("num_answer_tokens", 0)
        answer_sequences = metrics.get("num_answer_sequences", 0)
        if answer_tokens:
            metrics["answer_token_ce"] = metrics["answer_nll_sum"] / answer_tokens
        if answer_sequences:
            metrics["answer_exact_match_accuracy"] = (
                metrics["answer_exact_match_count"] / answer_sequences
            )
        for key, value in train_results.get("moe_metrics", {}).items():
            metrics[f"moe/{key}"] = value
        for key, value in train_results.get("mtp_metrics", {}).items():
            metrics[f"mtp/{key}"] = value
        for key in ("total_flops", "num_ranks", "theoretical_tflops"):
            if key in train_results:
                metrics[key] = train_results[key]
        for key, value in train_results.get("step_phases", {}).items():
            metrics[f"worker_{key}"] = float(value)
        return metrics

    def _loader_call(self, method_name: str) -> list[Any]:
        """Invoke a loader lifecycle method; non-owner workers return ``None``."""
        futures = self._trainer.worker_group.run_all_workers_single_data(
            method_name,
            run_rank_0_only_axes=(
                list(REPLICATED_AXES) if self._trainer.mimo_config is None else None
            ),
        )
        return ray.get(futures)

    def _owner_call(self, method_name: str) -> list[Any]:
        """Call ordinary-SFT validation owners using the base's replica layout."""
        futures = self._trainer.worker_group.run_all_workers_single_data(
            method_name, run_rank_0_only_axes=list(REPLICATED_AXES)
        )
        return ray.get(futures)

    def _ordered_owner_results(
        self, results: list[Any], *, operation: str
    ) -> list[Any]:
        """Validate owner ranks and return their results in placement-copy order."""
        if self._trainer.mimo_config is None:
            if len(results) != len(self._placement_plan.copies):
                raise RuntimeError(
                    f"Unexpected {operation} result count: {len(results)}."
                )
            return results
        world_size = self._train_cluster.world_size()
        if len(results) != world_size:
            raise RuntimeError(
                f"{operation} returned {len(results)} worker results; "
                f"expected {world_size}."
            )
        by_rank = {
            rank: result for rank, result in enumerate(results) if result is not None
        }
        expected_ranks = {copy.owner_rank for copy in self._placement_plan.copies}
        if set(by_rank) != expected_ranks:
            raise RuntimeError(
                f"Unexpected {operation} owners: actual={sorted(by_rank)}, "
                f"expected={sorted(expected_ranks)}."
            )
        return [by_rank[copy.owner_rank] for copy in self._placement_plan.copies]

    def _loader_state_dicts(self) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._ordered_owner_results(
                self._loader_call("sft_dataloader_state_dict"),
                operation="loader checkpoint",
            ),
        )

    def _should_save(self, *, save_by_timeout: bool) -> bool:
        config = self._master_config.checkpointing
        ft_save_period = config.get("ft_save_period")
        steps = self._save_state.total_steps
        return bool(config["enabled"]) and (
            save_by_timeout
            or steps == self._max_steps
            or steps % config["save_period"] == 0
            or (ft_save_period is not None and steps % ft_save_period == 0)
        )

    def _checkpoint_metric(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Read checkpointing.metric_name out of one step's training metrics.

        Called every step rather than only on a save so a name that no step
        produces fails on step 1 instead of at the first checkpoint.
        """
        metric_name = self._master_config.checkpointing["metric_name"]
        if metric_name is None:
            return {}
        # setup_sft_v2 already rejected anything but a train: name.
        key = metric_name.split(":", 1)[1]
        if key not in metrics:
            raise ValueError(
                f"checkpointing.metric_name={metric_name!r} names a training "
                f"metric this step did not produce. Available: {sorted(metrics)}."
            )
        return {metric_name: float(metrics[key])}

    def _save_checkpoint(self, metric: dict[str, float]) -> None:
        step = self._save_state.total_steps
        loader_states = self._loader_state_dicts()
        expected_loader_states = len(self._placement_plan.copies)
        if len(loader_states) != expected_loader_states:
            raise RuntimeError("Refusing to save without every logical loader state.")
        # CheckpointManager ranks keep_top_k on metric_name read back out of
        # training_info, so the metric travels with the save state.
        training_info = vars(self._save_state) | metric
        checkpoint_path = self._checkpointer.init_tmp_checkpoint(
            step, training_info, self._master_config
        )
        self._trainer.save_checkpoint(
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            optimizer_path=(
                os.path.join(checkpoint_path, "policy", "optimizer")
                if self._checkpointer.save_optimizer
                else None
            ),
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            is_final_checkpoint=step == self._max_steps,
        )
        torch.save(loader_states, os.path.join(checkpoint_path, "sft_v2_loaders.pt"))
        self._checkpointer.begin_finalization(
            checkpoint_path, wait_fn=self._trainer.finalize_async_save
        )

    def _close_loaders(self) -> None:
        self._loader_call("close_sft_dataloader")


def setup_sft_v2(
    master_config: MasterConfig, tokenizer_or_processor: Any
) -> SFTV2ActorArgs:
    """Build the V2 cluster, TQPolicy, placement, and resume state."""
    set_seed(master_config.sft.seed)
    mimo_config = resolve_mimo_sft_config(master_config.policy)
    local_plane = isinstance(master_config.data_plane, LocalDataPlaneConfig)
    single_owner = mimo_config is not None and mimo_config.single_owner_loader
    if single_owner and local_plane:
        raise ValueError(
            "MIMO single_owner_loader requires data_plane.impl=transfer_queue."
        )
    if not single_owner and not local_plane:
        raise ValueError(
            "Ordinary SFT and independent MiMo loaders require data_plane.impl=local."
        )
    if mimo_config is not None:
        if master_config.data.get("validation") is not None:
            raise NotImplementedError("MIMO SFT v2 validation is not supported yet.")
        _validate_mimo_topology_mapper(master_config.data["energon"].topology_mapper)
        if master_config.data["energon"].task_encoder.packing is None:
            raise ValueError("MIMO SFT v2 requires Energon-owned sequence packing.")
        if master_config.policy["train_micro_batch_size"] != 1:
            raise ValueError("MIMO SFT v2 currently requires train_micro_batch_size=1.")
        model_overrides = (
            master_config.policy["megatron_cfg"].get("model_overrides") or {}
        )
        for name in ("gtp_weight_remat_size", "expert_gtp_weight_remat_size"):
            size = model_overrides.get(name)
            if size is not None and size != 1:
                raise NotImplementedError(
                    "The pinned Bridge MIMO provider constructs singleton GTP groups; "
                    f"{name}={size} requires Bridge MIMO GTP support."
                )
    # Workers read policy.pretrained_checkpoint. YAML stores the block under
    # checkpointing, so copy it here the same way SFT v1 does.
    checkpointing_pretrained = master_config.checkpointing.get("pretrained_checkpoint")
    if checkpointing_pretrained is not None:
        master_config.policy["pretrained_checkpoint"] = checkpointing_pretrained
    if master_config.data.get("backend") != "energon":
        raise ValueError("SFTv2 requires data.backend=energon.")
    if not master_config.policy["megatron_cfg"]["enabled"]:
        raise ValueError("SFTv2 supports only the Megatron policy backend.")
    sequence_packing = master_config.policy["sequence_packing"]
    dynamic_batching = master_config.policy["dynamic_batching"]
    energon_packing = master_config.data["energon"].task_encoder.packing
    if energon_packing is None:
        if sequence_packing["enabled"] or dynamic_batching["enabled"]:
            raise ValueError("SFTv2 without Energon packing requires fixed batching.")
    else:
        if not sequence_packing["enabled"] or not sequence_packing.get(
            "fuse_loss", False
        ):
            raise ValueError(
                "Energon packing requires sequence_packing enabled with fuse_loss."
            )
        if energon_packing.name not in {
            algorithm.value for algorithm in PackingAlgorithm
        }:
            raise ValueError("Energon SFT requires a supported packing algorithm.")
        if dynamic_batching["enabled"]:
            raise ValueError("Energon packing does not support dynamic batching.")
    validation_requested = (
        master_config.sft.val_period != 0
        or master_config.sft.val_at_start
        or master_config.sft.val_at_end
    )
    if validation_requested and master_config.data.get("validation") is None:
        raise ValueError("SFTv2 validation was requested but data.validation is null.")
    if master_config.sft.val_period < 0:
        raise ValueError("sft.val_period must be non-negative.")
    # Checkpoint ranking still consumes the current training-step metrics.
    metric_name = master_config.checkpointing["metric_name"]
    if metric_name is not None and not metric_name.startswith("train:"):
        raise ValueError(
            "SFTv2 can rank checkpoints on training metrics only. Set "
            "checkpointing.metric_name to null or 'train:<metric>'; got "
            f"{metric_name!r}."
        )
    max_sequence_length = master_config.data["max_input_seq_length"]
    if max_sequence_length is None:
        raise ValueError("SFTv2 requires data.max_input_seq_length.")
    if energon_packing is not None:
        megatron_cfg = master_config.policy["megatron_cfg"]
        if (
            megatron_cfg.get("moe_token_dispatcher_type") == "flex"
            and megatron_cfg.get("moe_flex_dispatcher_backend") == "hybridep"
        ):
            # Packed batches can have different physical lengths across EP ranks.
            model_overrides = megatron_cfg.get("model_overrides") or {}
            if not model_overrides.get("moe_hybridep_pad_uneven_dispatch_inputs"):
                raise ValueError(
                    "Energon packing with HybridEP requires "
                    "policy.megatron_cfg.model_overrides."
                    "moe_hybridep_pad_uneven_dispatch_inputs=true."
                )

        pack_options = energon_packing.options
        if pack_options.max_sequence_length != max_sequence_length:
            raise ValueError(
                "Energon pack capacity must match data.max_input_seq_length."
            )
        pad_multiple = pack_options.sequence_length_pad_multiple
        parallel_multiple = get_parallel_token_alignment(megatron_cfg)
        if pad_multiple % parallel_multiple != 0:
            raise ValueError(
                "Energon packing requires make_sequence_length_divisible_by to "
                f"be a multiple of {parallel_multiple}."
            )
        if max_sequence_length % pad_multiple != 0:
            raise ValueError(
                "Energon packing requires max_input_seq_length to be divisible by "
                "make_sequence_length_divisible_by."
            )

        fp8_multiple = get_fp8_token_alignment(megatron_cfg) * parallel_multiple
        if fp8_multiple > parallel_multiple:
            if max_sequence_length % fp8_multiple != 0:
                raise ValueError(
                    "Energon packing requires max_input_seq_length to be divisible "
                    f"by the FP8 packed-token alignment ({fp8_multiple})."
                )

    processor = None
    tokenizer = tokenizer_or_processor
    if not isinstance(tokenizer_or_processor, PreTrainedTokenizerBase):
        processor = tokenizer_or_processor
        tokenizer = tokenizer_or_processor.tokenizer
    if processor is None:
        raise ValueError("SFTv2 requires a multimodal processor.")
    # Workers rebuild the processor in-process rather than receiving it as a
    # pickled constructor argument. A trust_remote_code processor's class lives
    # in ``transformers_modules``, which Ray's worker interpreters cannot import
    # while deserializing their arguments, so shipping the object fails there.
    master_config.policy["tokenizer"]["use_processor"] = True

    checkpoint_probe = CheckpointManager(master_config.checkpointing)
    latest = checkpoint_probe.get_latest_checkpoint_path()
    loaded_training_info = checkpoint_probe.load_training_info(latest)
    weights_path, optimizer_path = checkpoint_probe.get_resume_paths(latest)
    checkpoint_probe.shutdown()

    # Resume diagnostics are intentionally controlled by environment variables so
    # they can be enabled by an isolated launcher without extending the public SFT
    # configuration schema.  A checkpoint directory that merely looks plausible
    # must never silently fall back to the pretrained model during a resume test.
    require_resume = os.environ.get("NEMO_RL_REQUIRE_RESUME", "0") == "1"
    expected_checkpoint = os.environ.get("NEMO_RL_EXPECTED_RESUME_CHECKPOINT")
    if require_resume:
        print(
            "SFTv2 checkpoint preflight: "
            f"latest={latest!r} weights_path={str(weights_path) if weights_path else None!r} "
            f"optimizer_path={str(optimizer_path) if optimizer_path else None!r} "
            f"require_resume={require_resume} expected_checkpoint={expected_checkpoint!r}",
            flush=True,
        )
        if latest is None or weights_path is None or optimizer_path is None:
            raise RuntimeError(
                "Resume gate failed: expected a checkpoint with embedded optimizer "
                f"state, got latest={latest!r}, weights_path={weights_path!r}, "
                f"optimizer_path={optimizer_path!r}."
            )
        if expected_checkpoint and os.path.realpath(latest) != os.path.realpath(
            expected_checkpoint
        ):
            raise RuntimeError(
                "Resume gate failed: selected checkpoint differs from the launcher "
                f"expectation: selected={latest!r}, expected={expected_checkpoint!r}."
            )

    cluster_config = master_config.cluster
    num_nodes = cluster_config.num_nodes
    gpu_bundles_per_node = [cluster_config.gpus_per_node] * num_nodes
    if mimo_config is not None:
        # The explicit module rank spans define the worker world. MiMo may
        # use only part of the last allocated node (e.g. 64 language + 1 vision).
        worker_count = max(
            module["rank_offset"]
            + module["tensor_model_parallel_size"]
            * module["context_parallel_size"]
            * module["pipeline_model_parallel_size"]
            * module["data_parallel_size"]
            for module in mimo_config.parallelism["module_parallelisms"].values()
        )
        last_node_workers = worker_count - sum(gpu_bundles_per_node[:-1])
        if not 0 < last_node_workers <= cluster_config.gpus_per_node:
            raise ValueError("cluster.num_nodes must fit the explicit MIMO rank spans.")
        gpu_bundles_per_node[-1] = last_node_workers
    segment_size = cluster_config.segment_size
    node_block_counts = get_node_block_counts()
    if node_block_counts:
        # The launcher placed each node block (one NVLink segment per block);
        # keep each block's range of global ranks on that block's nodes.
        if segment_size is not None:
            raise ValueError(
                "cluster.segment_size must be null when nodes are split into node blocks "
                "(node_block_<n> resources); give each NVLink segment its own block "
                "(on Slurm: one het component with --nodes=N --segment=N)."
            )
        node_constraints = node_block_constraints(node_block_counts, num_nodes)
        print(
            "  ✓ Node blocks: global ranks follow block order, "
            f"nodes per block {node_block_counts}",
            flush=True,
        )
    else:
        node_constraints, _, _ = prepare_segment_topology(segment_size, num_nodes)
    cluster = RayVirtualCluster(
        name="sft_v2_cluster",
        bundle_ct_per_node_list=gpu_bundles_per_node,
        use_gpus=True,
        num_gpus_per_node=cluster_config.gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.master_port_range_low,
        port_range_high=cluster_config.master_port_range_high,
        segment_size=segment_size,
        node_resource_constraints=node_constraints,
    )
    megatron_config = cast(dict[str, Any], master_config.policy["megatron_cfg"])
    megatron_config["train_iters"] = _max_train_steps(master_config)
    trainer = TQPolicy(
        cluster=cluster,
        config=master_config.policy,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
        worker_extension_cls_fqn=(
            "nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker"
        ),
        dp_cfg=master_config.data_plane,
        allow_mimo_sft=True,
    )
    mapper_name = master_config.data["energon"].topology_mapper
    if trainer.mimo_config is not None:
        placement_plan = build_mimo_data_loader_placement(
            trainer.mimo_config.parallelism,
            world_size=cluster.world_size(),
            single_owner=trainer.mimo_config.single_owner_loader,
        )
    else:
        placement_plan = resolve_topology_mapper(mapper_name).map(
            trainer.sharding_annotations
        )
    global_batch_size = master_config.policy["train_global_batch_size"]
    if global_batch_size % placement_plan.logical_world_size != 0:
        raise ValueError(
            "policy.train_global_batch_size must be divisible by the logical "
            f"DP size: {global_batch_size} % "
            f"{placement_plan.logical_world_size} != 0."
        )
    save_state = _restore_save_state(
        loaded_training_info, placement_hash=placement_plan.placement_hash
    )
    loader_states = None
    if latest is not None:
        loader_path = Path(latest) / "sft_v2_loaders.pt"
        if not loader_path.exists():
            raise ValueError(f"SFTv2 checkpoint is missing {loader_path.name}.")
        loader_states = torch.load(loader_path, weights_only=False)
    loss_fn = NLLLossFn(
        use_fused_linear_logprobs=megatron_config["use_fused_linear_logprobs"]
    )
    return SFTV2ActorArgs(
        trainer=trainer,
        loss_fn=loss_fn,
        train_cluster=cluster,
        placement_plan=placement_plan,
        save_state=save_state,
        loader_states=loader_states,
    )


__all__ = [
    "MasterConfig",
    "SFTSingleControllerActor",
    "SFTV2ActorArgs",
    "SFTV2SaveState",
    "check_mimo_lanes_agree",
    "mimo_accounting_envelopes",
    "setup_sft_v2",
]
