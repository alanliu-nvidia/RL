#!/usr/bin/env python3
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

"""Check NM4 answer masks with the actual processor, entirely on CPU."""

import argparse
import json
from pathlib import Path

from inspect_processor import load_processor

from nemo_rl.data.energon.multimodal.task_encoders.nemotron4_tokenization import (
    tokenize_nemotron_conversation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--processor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, processor = load_processor(args.config, args.processor)
    messages = [
        {"role": "user", "content": "UNTRAINED_QUESTION_A"},
        {"role": "assistant", "content": "TRAINED_ANSWER_A"},
        {"role": "user", "content": "UNTRAINED_QUESTION_B"},
        {"role": "assistant", "content": "TRAINED_ANSWER_B"},
    ]
    cases = [
        (
            "all_answers",
            messages,
            False,
            None,
            False,
            ["TRAINED_ANSWER_A", "TRAINED_ANSWER_B"],
            [],
        ),
        # Preserve the existing Nemotron rule: last-turn filtering activates
        # for a nonempty thinking trace or native tool-response boundaries.
        (
            "last_flag_without_thinking",
            messages,
            True,
            None,
            False,
            ["TRAINED_ANSWER_A", "TRAINED_ANSWER_B"],
            [],
        ),
        (
            "last_answer_with_thinking",
            [
                {**m, "content": "<think>reason</think>" + m["content"]}
                if m["role"] == "assistant"
                else m
                for m in messages
            ],
            True,
            None,
            False,
            ["TRAINED_ANSWER_B"],
            ["TRAINED_ANSWER_A"],
        ),
        (
            "explicit_first",
            messages,
            False,
            [True, False],
            False,
            ["TRAINED_ANSWER_A"],
            ["TRAINED_ANSWER_B"],
        ),
        (
            "explicit_second",
            messages,
            False,
            [False, True],
            False,
            ["TRAINED_ANSWER_B"],
            ["TRAINED_ANSWER_A"],
        ),
        (
            "native_tool_boundary",
            [
                *messages[:2],
                {"role": "tool", "content": "UNTRAINED_TOOL_RESULT"},
                messages[-1],
            ],
            True,
            None,
            True,
            ["TRAINED_ANSWER_B"],
            ["TRAINED_ANSWER_A", "UNTRAINED_TOOL_RESULT"],
        ),
    ]
    results = []
    for (
        name,
        conversation,
        last_only,
        explicit,
        native_tool,
        included,
        excluded,
    ) in cases:
        encoded = tokenize_nemotron_conversation(
            conversation,
            processor=processor,
            prompt_format="nemotron6-moe",
            skip_chat_template=False,
            train_only_on_last_assistant_turn=last_only,
            tool_response_as_turn_boundary=native_tool,
            assistant_turn_loss=explicit,
        )[0]
        trained = processor.tokenizer.decode(
            encoded["token_ids"][encoded["token_loss_mask"].bool()].tolist()
        )
        for text in included:
            assert text in trained, (name, "missing answer", text, trained)
        for text in [
            *excluded,
            "UNTRAINED_QUESTION_A",
            "UNTRAINED_QUESTION_B",
            "<|im_start|>assistant",
        ]:
            assert text not in trained, (
                name,
                "unexpected training target",
                text,
                trained,
            )
        assert trained.count("<|im_end|>") == len(included), (
            name,
            "missing end target",
            trained,
        )
        results.append({"case": name, "passed": True, "trained_text": trained})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
