# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""vLLM general plugin that registers NT4 as a multimodal model.

Calling ``register_nemotron4_vlm`` from NeMo-RL's own patch path only reaches
the process that applies the patches. vLLM runs the model in separate worker
processes -- Ray actors when ``tensor_parallel_size > 1``, spawned children
otherwise -- and each builds a fresh ``ModelRegistry`` from vLLM's built-in
tables. Those workers resolved NT4 back to the text-only
``Nemotron4ForCausalLM`` and failed with "has no registered multimodal
processor" (job 7771127), which is the benign version of this bug; the
dangerous version is a worker that loads text-only weights and silently
ignores every image.

vLLM's plugin loader is the supported way across processes. Its own docstring
says "plugins can be loaded for multiple times in different processes", which
is exactly the property needed here, and ``register_nemotron4_vlm`` is
idempotent.

Discovery is by entry point, so this directory carries a ``.dist-info`` and is
placed on ``PYTHONPATH``; nothing in the container image is modified.
"""


def register() -> None:
    """Entry point for the ``vllm.general_plugins`` group."""
    from vllm.logger import init_logger

    from nemo_rl.models.generation.vllm.nemotron4_vlm import register_nemotron4_vlm

    register_nemotron4_vlm(init_logger("nemo_rl.nt4_vlm_plugin"))
