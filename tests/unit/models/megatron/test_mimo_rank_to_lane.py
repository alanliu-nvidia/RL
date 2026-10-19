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

import pytest

from nemo_rl.models.megatron.mimo import mimo_rank_to_lane


def _parallelism(
    *,
    language_dp,
    encoder_dp,
    language_tp=1,
    language_cp=1,
    language_offset=0,
    encoder_offset=None,
):
    language_replica = language_tp * language_cp
    if encoder_offset is None:
        encoder_offset = language_offset + language_dp * language_replica
    return {
        "module_parallelisms": {
            "language": {
                "tensor_model_parallel_size": language_tp,
                "context_parallel_size": language_cp,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": language_dp,
                "rank_offset": language_offset,
            },
            "images": {
                "tensor_model_parallel_size": 1,
                "context_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": encoder_dp,
                "rank_offset": encoder_offset,
            },
        }
    }


def test_language_ranks_come_first_then_strided_encoder_lanes():
    lanes = mimo_rank_to_lane(
        _parallelism(language_dp=2, encoder_dp=4), world_size=6
    )
    # Ranks 0-1 are the two language lanes; 2-5 are two encoder ranks per lane.
    assert lanes == [0, 1, 0, 0, 1, 1]


def test_a_language_replica_spans_its_whole_tp_cp_group():
    lanes = mimo_rank_to_lane(
        _parallelism(language_dp=2, encoder_dp=2, language_tp=2, language_cp=2),
        world_size=10,
    )
    assert lanes[:4] == [0, 0, 0, 0]
    assert lanes[4:8] == [1, 1, 1, 1]
    assert lanes[8:] == [0, 1]


def test_equal_dp_gives_each_encoder_rank_its_own_lane():
    lanes = mimo_rank_to_lane(
        _parallelism(language_dp=4, encoder_dp=4), world_size=8
    )
    assert lanes == [0, 1, 2, 3, 0, 1, 2, 3]


def test_rejects_encoder_dp_that_is_not_a_multiple():
    with pytest.raises(ValueError, match="multiple of"):
        mimo_rank_to_lane(
            _parallelism(language_dp=4, encoder_dp=6), world_size=10
        )


def test_rejects_a_world_the_modules_do_not_cover():
    with pytest.raises(ValueError, match="without a module"):
        mimo_rank_to_lane(_parallelism(language_dp=2, encoder_dp=2), world_size=5)


def test_rejects_overlapping_module_rank_spans():
    with pytest.raises(ValueError, match="claimed by two modules"):
        mimo_rank_to_lane(
            _parallelism(language_dp=2, encoder_dp=2, encoder_offset=1),
            world_size=4,
        )
