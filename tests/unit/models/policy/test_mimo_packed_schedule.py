# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for the microbatch count every MIMO role steps through."""

from types import SimpleNamespace

import pytest

from nemo_rl.models.policy.tq_policy import TQPolicy


def _lane(rows: int, bins: int | None = None) -> SimpleNamespace:
    """One language lane's metadata, optionally carrying the driver's plan."""
    extra: dict[str, object] = {}
    if bins is not None:
        extra["micro_batch_indices"] = [[[i, i + 1] for i in range(bins)]]
    return SimpleNamespace(sample_ids=list(range(rows)), extra_info=extra)


def _policy(*, packing: bool = False, dynamic: bool = False) -> SimpleNamespace:
    return SimpleNamespace(use_sequence_packing=packing, use_dynamic_batches=dynamic)


def _count(policy, lanes, micro_batch_size=1):
    return TQPolicy._mimo_lane_microbatches(policy, lanes, micro_batch_size)


def test_unpacked_count_comes_from_the_lane_row_count():
    assert _count(_policy(), [_lane(8), _lane(8)], micro_batch_size=2) == 4


def test_packed_count_comes_from_the_driver_plan_not_the_rows():
    # The whole point of packing: three bins hold eight sequences, and the
    # schedule runs three times rather than eight.
    lanes = [_lane(8, bins=3), _lane(8, bins=3)]
    assert _count(_policy(packing=True), lanes, micro_batch_size=1) == 3


def test_packing_without_a_driver_plan_is_refused():
    with pytest.raises(RuntimeError, match="did not stamp one"):
        _count(_policy(packing=True), [_lane(8), _lane(8)])


def test_lanes_that_disagree_are_refused():
    # Letting these through would park one role in a collective the other
    # never enters.
    lanes = [_lane(8, bins=3), _lane(8, bins=4)]
    with pytest.raises(ValueError, match="agree on their microbatch count"):
        _count(_policy(packing=True), lanes)


def test_dynamic_batching_is_still_refused():
    with pytest.raises(NotImplementedError, match="dynamic batching"):
        _count(_policy(dynamic=True), [_lane(8), _lane(8)])


def test_uneven_unpacked_lane_is_refused():
    with pytest.raises(ValueError, match="whole number of"):
        _count(_policy(), [_lane(7)], micro_batch_size=2)
