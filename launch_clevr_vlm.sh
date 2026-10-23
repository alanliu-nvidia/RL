#!/bin/bash
# NT4 Nano CLEVR VLM GRPO launcher -- 8-node smoke and 64-node full scale.
#
# Purpose: confirm the VLM path now starts training. The last ten attempts (Oct
# 3-4, jobs 7654555 .. 7658550) all died inside 10 minutes, the final one on
#
#   TypeError: prepare_swap_model_weights() got an unexpected keyword argument
#   'execution_batch_bytes'
#   in MegatronPolicyWorker.init_collective_mcore_generation()
#
# which f5359a588 and de91a2f23 fixed by probing the signature instead of passing
# the kwarg blindly. The 86-node text run uses the vLLM backend and never calls
# init_collective_mcore_generation, so that fix has never been exercised.
#
# The recipe asks for 5 steps, which is what matters: step 2 is where the text
# runs died, and Adam's state only exists from then on.
set -euo pipefail

# `pwd -P`, not `pwd`. On AGA /lustre/fsw/portfolios/nemotron/users/$USER is a
# symlink into ../projects/nemotron_sw_pre/users/$USER, and /lustre/fsw and
# /scratch/fsw are two names for one mount. ray.sub passes
# --container-workdir=$SLURM_SUBMIT_DIR and pyxis resolves it to the physical
# path, so a logical PROJECT_ROOT gets mounted under one name and chdir'd under
# another: job 1157658 lost all 8 nodes in 2m10s to
#   pyxis: couldn't chdir to /scratch/fsw/.../nemotron_sw_pre/users/sauramishra/
#   nt4-nano/RL: No such file or directory
# Canonicalizing here keeps the mount, the workdir and the recipe's paths in
# agreement.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STAMP="$(date +%Y%m%d-%H%M%S)"

# OCI-AGA. The HSG defaults pointed at nemotron_sw_post and a /lustre/fs1
# coreai cache; fs1 is not mounted here, so every path below is AGA-local and
# under this user's own tree rather than another user's.
NT4_AGA_ROOT="${NT4_AGA_ROOT:-/lustre/fsw/portfolios/nemotron/users/${USER}/nt4-nano}"
RUN_ROOT="${RUN_ROOT:-${NT4_AGA_ROOT}/runs}"
# Names the run in squeue, in wandb and on disk. Defaults to the scale so the
# two recipes do not collide; override it to tell two runs of the same scale
# apart (RUN_TAG=rerun, RUN_TAG=nightly, ...).
RUN_TAG="${RUN_TAG:-${SCALE:-8}n}"
RUN_DIR="${RUN_ROOT}/nt4nano-clevr-${RUN_TAG}-${STAMP}"
CACHE_DIR="${CACHE_DIR:-${NT4_AGA_ROOT}/cache}"
# Pinned to an image rather than the rl-nt4-nightly.sqsh symlink, which the CI
# job repoints and which therefore moves under you between runs. Every run
# quoted in RUN-ON-AGA.md used this one: the 64-node 4h baseline 1331328 and the
# 8-node pair 1334625/1334626.
CONTAINER="${CONTAINER:-/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_ci/nemo_rl_ci/sqsh_files/high_stripe/rl-gym.74660702.sqsh}"
# /scratch, not /lustre, and resolved rather than taken on trust.
#
# The shared filesystem is mounted at /scratch/fsw; `/lustre` is a symlink to
# /scratch that lives on each node's *own* root filesystem, not on the shared
# mount. A node provisioned without it resolves every /lustre path to nothing
# and pyxis reports "No such file or directory" for a file that 63 of its 64
# peers are reading happily. That is the error that killed 1163092, 1164499,
# 1167559 and 1167825, on four different nodes in one day. Naming the real
# path removes the dependency on a per-node symlink entirely.
#
# realpath so that a caller passing the /lustre spelling gets the same
# protection; it is a no-op on a path that is already canonical.
if [[ -e "${CONTAINER}" ]]; then
  CONTAINER="$(realpath "${CONTAINER}")"
fi

# The two supported scales, named explicitly rather than assembled from a
# filename template. The template only worked while a recipe existed for every
# scale we had swept, and it failed open: an unsupported SCALE produced a
# plausible path, missed the file, and blamed the scale.
#
#   SCALE=8  ./launch_clevr_vlm.sh    # 4 train + 4 gen   -- smoke, 5 steps
#   SCALE=64 ./launch_clevr_vlm.sh    # 32 train + 32 gen -- full scale
#
# These are MIMO recipes: the vision encoder and the language model occupy
# disjoint rank grids, so a node count is not freely choosable. The rank
# budget is language_DP * (stride + CP) = trainer_ranks, which admits 8 and 64
# at CP 2 and would need a role split derived from scratch at anything else.
#
# There is no megatron-generation variant. Megatron cannot serve this model:
# job 7769415 reached generation and died because our Megatron-core pin
# predates media_tokens_preexpanded.
#
# CONFIG may still be set directly to run a recipe that is not one of these.
if [[ -z "${CONFIG:-}" ]]; then
  case "${SCALE:-8}" in
    8)  CONFIG="examples/configs/recipes/vlm/vlm_grpo-nt4nano-clevr-mimo-8n4g-smoke.yaml" ;;
    64) CONFIG="examples/configs/recipes/vlm/vlm_grpo-nt4nano-clevr-mimo-64n4g.yaml" ;;
    *)  echo "ERROR: SCALE must be 8 or 64, got '${SCALE}'. Set CONFIG= for another recipe." >&2; exit 1 ;;
  esac
fi
if [[ -n "${GEN:-}" && "${GEN}" != vllm ]]; then
  echo "ERROR: GEN must be vllm (or unset), got '${GEN}'" >&2; exit 1
fi
[[ -f "${PROJECT_ROOT}/${CONFIG}" ]] || { echo "ERROR: recipe not found: ${CONFIG}" >&2; exit 1; }

# --nodes must equal the recipe's cluster.num_nodes. Asking SLURM for a
# different number than the config expects either hangs waiting for workers
# that will never arrive or strands whole nodes, so read it out of the config
# rather than trusting the two to be edited together.
NODES="$(cd "${PROJECT_ROOT}" && python3 -c "
import yaml,sys
cfg=yaml.safe_load(open('${CONFIG}')) or {}
n=(cfg.get('cluster') or {}).get('num_nodes')
while n is None and 'defaults' in cfg:
    import os
    cfg=yaml.safe_load(open(os.path.join(os.path.dirname('${CONFIG}'), cfg['defaults']))) or {}
    n=(cfg.get('cluster') or {}).get('num_nodes')
print(n or 8)
")"
echo "Nodes:     ${NODES} (from ${CONFIG})"

# --segment must agree with cluster.segment_size for the same reason --nodes
# must agree with cluster.num_nodes, so it is read out of the config too.
# SLURM's --segment decides the shape of the allocation; the recipe's
# segment_size decides how NeMo-RL carves placement groups out of it. If SLURM
# hands back domains holding a non-multiple of segment_size,
# select_segment_nodes keeps only the whole segments and raises
# ResourceInsufficientError on what is left.
#
# Unlike the NODES probe this stops at the first config that mentions the key,
# so an explicit null in a child beats a number in its parent -- which is what
# the recipe loader's deep merge does. Empty means the recipe does not ask for
# topology-aware placement, and ray.sub's own #SBATCH --segment default stands.
SEGMENT="$(cd "${PROJECT_ROOT}" && python3 -c "
import os, yaml
path = '${CONFIG}'
while True:
    cfg = yaml.safe_load(open(path)) or {}
    cluster = cfg.get('cluster') or {}
    if 'segment_size' in cluster:
        print('' if cluster['segment_size'] is None else cluster['segment_size'])
        break
    if 'defaults' not in cfg:
        print('')
        break
    path = os.path.join(os.path.dirname(path), cfg['defaults'])
")"
echo "Segment:   ${SEGMENT:-<ray.sub default>} (from cluster.segment_size)"

# Which generation backend the plugin is wired for has to come from the recipe,
# the same way NODES and SEGMENT do. It used to come from a GEN env default of
# "megatron", which disagreed with every shipped recipe and failed silently:
# the workers still ran vLLM, just without the NT4 multimodal registration, so
# they fell back to a stock processor and died in the profiling run (section 9
# of RUN-ON-AGA.md). Both recipes declare policy.generation.backend: vllm.
# Stops at the first config that mentions the key, like the SEGMENT probe.
GEN_BACKEND="$(cd "${PROJECT_ROOT}" && python3 -c "
import os, yaml
path = '${CONFIG}'
while True:
    cfg = yaml.safe_load(open(path)) or {}
    gen = (cfg.get('policy') or {}).get('generation') or {}
    if 'backend' in gen:
        print(gen['backend'] or '')
        break
    if 'defaults' not in cfg:
        print('')
        break
    path = os.path.join(os.path.dirname(path), cfg['defaults'])
")"
if [[ -z "${GEN_BACKEND}" ]]; then
  echo "ERROR: ${CONFIG} declares no policy.generation.backend" >&2
  exit 1
fi
# An explicit GEN still wins, so a one-off override stays possible.
GEN="${GEN:-${GEN_BACKEND}}"
echo "Generation: ${GEN} (from policy.generation.backend)"

# The three paths the recipe names via oc.env. Naming them here is what gets them
# mounted; job 7753865 lost 86 nodes to a path the recipe knew about and the
# launcher did not.
# The one model directory, used by the trainer and by vLLM. It is the BF16 SFT
# HF export plus the processor files and the few config facts vLLM needs, built
# by assemble_nt4_vllm_dir.sh; NeMo-RL converts it to Megatron format once into
# $HF_HOME/nemo_rl. It replaced the sanitized_mbridge checkpoint, which
# deserializes to a MegatronMIMOProvider that NeMo-RL would silently
# misconfigure -- see the comment on policy.model_name in the recipe.
#
# One directory and not two because setup.py:600 copies policy.model_name over
# generation.model_name regardless of what the config asks for.
NT4_NANO_MODEL_DIR="${NT4_NANO_MODEL_DIR:-${NT4_AGA_ROOT}/models/nt4-nano-sft-vllm}"
NT4_NANO_PROCESSOR="${NT4_NANO_PROCESSOR:-${NT4_AGA_ROOT}/models/processor}"
# Both recipes point checkpointing.pretrained_checkpoint here. A MIMO run
# cannot start from HF weights -- the HF->Megatron cache is written before the
# MIMO provider replaces the standard one, so it holds the flat vision_model.*
# layout rather than modality_submodules.images.* -- so this has to exist
# before the first run. Build it with convert_hf_to_mimo_bridge.py.
# Exported at declaration, not with the others below: the block that reads
# pretrained_checkpoint out of the recipe runs in a subshell and only sees
# exported variables.
export NT4_NANO_MIMO_CKPT="${NT4_NANO_MIMO_CKPT:-${NT4_AGA_ROOT}/models/nt4-nano-sft-mimo-bridge}"
# Repairs two packaging defects in rl-gym.72488118 from outside the image; see
# the module docstring for the full diagnosis. In short, the Megatron worker's
# venv has nvidia_cutlass_dsl_internal layered over nvidia_cutlass_dsl and the
# merged cutlass tree raises AttributeError on import, which kills the worker
# before it starts and takes FA4 with it. Each repair checks its own premise,
# so this goes quiet once the container is fixed.
NT4_SHIM_DIR="${NT4_SHIM_DIR:-${NT4_AGA_ROOT}/shims}"

# The NM4 bridge provider only exists in this checkout's Megatron-Bridge, and the
# image's nested Megatron-LM imports mistral_adapter, which ends at a bare
# `import cv2` that NeMo-RL excludes. Overlaying the whole repo at /opt/nemo-rl
# covers both, and is what the Oct 4 attempts did.
NT4_BRIDGE_ROOT="${NT4_BRIDGE_ROOT:-${PROJECT_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}"
_provider="${NT4_BRIDGE_ROOT}/src/megatron/bridge/models/nemotron4/nemotron4_provider.py"
if [[ ! -f "${_provider}" ]]; then
  echo "ERROR: NM4 bridge provider missing: ${_provider}" >&2
  exit 1
fi

# The bind-mount below carries Megatron-Bridge's whole subtree, so an empty
# nested 3rdparty/Megatron-LM shadows the image's populated copy and megatron.core
# stops existing inside the container. That surfaces as a ModuleNotFoundError in
# every Megatron worker about 19 minutes in, after the vLLM engines are already
# up, so check it here instead: see section 1 of RUN-ON-AGA.md for the fix.
_megatron_core="${NT4_BRIDGE_ROOT}/3rdparty/Megatron-LM/megatron/core/models/gpt/gpt_model.py"
if [[ ! -f "${_megatron_core}" ]]; then
  echo "ERROR: megatron.core missing from the nested Megatron-LM submodule." >&2
  echo "       Expected: ${_megatron_core}" >&2
  echo "       Fix: see 'Check out the code' in RUN-ON-AGA.md -- clone the pin" >&2
  echo "       5c1a9e5bf34c54d72e9166fb821d99e94cdbc43c into that path." >&2
  exit 1
fi

_required=(
  "${NT4_NANO_MODEL_DIR}"
  # vLLM resolves `nemotron4` on its own, but the trainer reaches this same
  # directory with trust_remote_code=True and needs the auto_map modules there.
  "${NT4_NANO_MODEL_DIR}/configuration_nemotron4.py"
  "${NT4_NANO_MODEL_DIR}/preprocessor_config.json"
  "${NT4_NANO_PROCESSOR}"
  "${NT4_NANO_MIMO_CKPT}"
  "${NT4_SHIM_DIR}/sitecustomize.py"
)
for p in "${_required[@]}"; do
  [[ -e "${p}" ]] || { echo "ERROR: missing on host: ${p}" >&2; exit 1; }
done

# Same reason as PROJECT_ROOT above: bind-mount the physical paths so the
# source and target of every mount survive pyxis resolving symlinks.
NT4_NANO_MODEL_DIR="$(realpath "${NT4_NANO_MODEL_DIR}")"
NT4_NANO_PROCESSOR="$(realpath "${NT4_NANO_PROCESSOR}")"
NT4_NANO_MIMO_CKPT="$(realpath "${NT4_NANO_MIMO_CKPT}")"
NT4_SHIM_DIR="$(realpath "${NT4_SHIM_DIR}")"
RUN_DIR="$(realpath -m "${RUN_DIR}")"
CACHE_DIR="$(realpath -m "${CACHE_DIR}")"

mkdir -p "${RUN_DIR}/logs" "${CACHE_DIR}"

# ray.sub runs every srun with --container-workdir=$SLURM_SUBMIT_DIR, so the
# submit dir has to exist inside the container under its host path as well as at
# the /opt/nemo-rl overlay. Job 7769230 lost 8 nodes in 3m40s to pyxis failing on
# "couldn't chdir to ..." because only the overlay was mounted.
MOUNTS="${PROJECT_ROOT}:${PROJECT_ROOT}"
MOUNTS+=",${PROJECT_ROOT}:/opt/nemo-rl"
MOUNTS+=",${NT4_BRIDGE_ROOT}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
MOUNTS+=",${NT4_NANO_MODEL_DIR}:${NT4_NANO_MODEL_DIR}:ro"
MOUNTS+=",${NT4_NANO_PROCESSOR}:${NT4_NANO_PROCESSOR}:ro"
MOUNTS+=",${NT4_NANO_MIMO_CKPT}:${NT4_NANO_MIMO_CKPT}:ro"
MOUNTS+=",${NT4_SHIM_DIR}:${NT4_SHIM_DIR}:ro"
MOUNTS+=",${RUN_DIR}:${RUN_DIR}"
MOUNTS+=",${CACHE_DIR}:${CACHE_DIR}"
# Checkpoints have to outlive a single job, so this cannot live under RUN_DIR,
# which carries a per-launch timestamp. A chained job resumes by globbing
# step_* in this directory, so every job in a chain must be given the same one.
# Mounted read-write, unlike the asset mounts above.
if [[ -n "${NT4_CKPT_DIR:-}" ]]; then
  NT4_CKPT_DIR="$(realpath -m "${NT4_CKPT_DIR}")"
  mkdir -p "${NT4_CKPT_DIR}"
  MOUNTS+=",${NT4_CKPT_DIR}:${NT4_CKPT_DIR}"
  export NT4_CKPT_DIR
fi
# On HSG the model dir's weights were symlinks into tbarnatan's tree, which had
# to be mounted too or they dangled inside pyxis. assemble_nt4_vllm_dir.sh now
# hardlinks instead, but fall back to mounting the target if any remain.
for _link in "${NT4_NANO_MODEL_DIR}"/*.safetensors; do
  [[ -L "${_link}" ]] || continue
  _tgt="$(dirname "$(readlink -f "${_link}")")"
  [[ ",${MOUNTS}," == *",${_tgt}:${_tgt}:ro,"* ]] || MOUNTS+=",${_tgt}:${_tgt}:ro"
done

# EXTRA_MOUNTS is a comma-separated list of host paths to bind read-only at
# their own path inside the container. A recipe that names a path outside this
# user's tree -- a pretrained_checkpoint in another project's directory, say --
# resolves to nothing inside pyxis unless it is mounted, and the failure looks
# like a missing checkpoint rather than a missing mount.
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  IFS=',' read -ra _extra_mounts <<< "${EXTRA_MOUNTS}"
  for _extra in "${_extra_mounts[@]}"; do
    [[ -n "${_extra}" ]] || continue
    if [[ ! -e "${_extra}" ]]; then
      echo "ERROR: EXTRA_MOUNTS path does not exist: ${_extra}" >&2
      exit 1
    fi
    _extra="$(realpath "${_extra}")"
    [[ ",${MOUNTS}," == *",${_extra}:${_extra}:ro,"* ]] || MOUNTS+=",${_extra}:${_extra}:ro"
  done
fi

# EXTRA_MOUNT_SPECS is a comma-separated list of raw "src:dst" pairs, for the
# case EXTRA_MOUNTS cannot express: shadowing a path that already exists inside
# the container with a different one from the host. Used to overlay a working
# nvidia_cutlass_dsl over the broken one in the megatron venv -- the image pairs
# public cutlass-dsl 4.6.2 with an internal build that only matches 4.6.0, so
# cutlass, quack and gdp_attn all fail to import and Megatron silently falls
# back from the CuTeDSL GDN kernel to FLA.
#
# Always bound read-only. Retire this once the image ships a coherent pair.
if [[ -n "${EXTRA_MOUNT_SPECS:-}" ]]; then
  IFS=',' read -ra _mount_specs <<< "${EXTRA_MOUNT_SPECS}"
  for _spec in "${_mount_specs[@]}"; do
    [[ -n "${_spec}" ]] || continue
    _src="${_spec%%:*}"
    _dst="${_spec#*:}"
    if [[ -z "${_src}" || -z "${_dst}" || "${_src}" == "${_spec}" ]]; then
      echo "ERROR: EXTRA_MOUNT_SPECS entry is not src:dst: ${_spec}" >&2
      exit 1
    fi
    if [[ ! -e "${_src}" ]]; then
      echo "ERROR: EXTRA_MOUNT_SPECS source does not exist: ${_src}" >&2
      exit 1
    fi
    _src="$(realpath "${_src}")"
    MOUNTS+=",${_src}:${_dst}:ro"
  done
fi

# The recipe's pretrained_checkpoint lives in another project's tree, so it
# needs the same treatment as EXTRA_MOUNTS above. Leaving it to the caller is
# what cost job 1203471 its 32 nodes: pyxis resolved the unmounted path to
# nothing and every MegatronPolicyWorker raised FileNotFoundError, which reads
# as a deleted checkpoint rather than a forgotten flag. Read it out of the
# recipe instead. The parent is mounted, not the iteration directory itself,
# because the loader accepts either an iter_* directory or one holding
# latest_checkpointed_iteration.txt and will look for the latter.
_CKPT="$(cd "${PROJECT_ROOT}" && python3 -c "
import os,re,yaml
path, cfg = None, '${CONFIG}'
while cfg and path is None:
    doc = yaml.safe_load(open(cfg)) or {}
    # Recipes write this under checkpointing:; the algorithm setup copies it
    # onto the policy before the workers read it, so accept either spelling.
    for section in ('policy', 'checkpointing'):
        path = (((doc.get(section) or {}).get('pretrained_checkpoint')) or {}).get('path')
        if path:
            break
    nxt = doc.get('defaults')
    cfg = os.path.join(os.path.dirname(cfg), nxt) if nxt else None
if path:
    # \${oc.env:NAME} or \${oc.env:NAME,default} -- the launcher cannot ask
    # OmegaConf to resolve this without importing the whole config, and the env
    # var wins if set. The recipes ship the bare form on purpose, so that an
    # unset variable fails here instead of silently loading someone else's
    # checkpoint; keep the two-argument form working for older recipes.
    m = re.fullmatch(r'\\\$\{oc\.env:([^,}]+)(?:,(.*))?\}', path.strip(), re.S)
    if m:
        name = m.group(1).strip()
        fallback = m.group(2).strip() if m.group(2) is not None else ''
        path = os.environ.get(name) or fallback
        if not path:
            raise SystemExit(
                f'ERROR: recipe needs \${{oc.env:{name}}} but it is unset'
            )
    print(path)
")"
if [[ -n "${_CKPT}" ]]; then
  if [[ ! -e "${_CKPT}" ]]; then
    echo "ERROR: recipe pretrained_checkpoint.path does not exist: ${_CKPT}" >&2
    exit 1
  fi
  # realpath collapses the /lustre symlink onto the /scratch mount that is
  # actually provisioned, so both spellings produce one mount entry.
  _ckpt_mnt="$(dirname "$(realpath "${_CKPT}")")"
  if [[ "${_ckpt_mnt}" != "${PROJECT_ROOT}"/* ]]; then
    [[ ",${MOUNTS}," == *",${_ckpt_mnt}:${_ckpt_mnt}:ro,"* ]] || MOUNTS+=",${_ckpt_mnt}:${_ckpt_mnt}:ro"
  fi
fi

# uv must not re-resolve: mounting Megatron-Bridge replaces that subtree's
# pyproject.toml and uv.lock, and the root project depends on it as an editable
# path, so `uv run` without --no-sync reaches a private GitLab index it has no
# credentials for. That killed job 7752254 seven minutes in.
COMMAND="cd /opt/nemo-rl && uv run --no-sync --python /opt/nemo_rl_venv/bin/python python examples/run_grpo_single_controller.py"
COMMAND+=" --config ${CONFIG}"
COMMAND+=" logger.log_dir=${RUN_DIR}/logs"
# NT4_CKPT_DIR when set, because a CLI override beats the recipe's
# ${oc.env:NT4_CKPT_DIR} and this line used to hand every job RUN_DIR
# unconditionally. RUN_DIR carries a per-submit timestamp, so a continuation
# globbed step_* in a directory its predecessor had never written to and
# restarted from the SFT weights while reporting nothing wrong -- which is
# what 1205809 would have done behind 1205808. RUN_DIR stays the fallback for
# unchained runs, whose checkpoints need not outlive the job.
COMMAND+=" checkpointing.checkpoint_dir=${NT4_CKPT_DIR:-${RUN_DIR}/checkpoints}"
COMMAND+=" logger.wandb.name=nt4nano-clevr-vlm-${GEN}-${RUN_TAG}-${STAMP}"
# Appended last so it wins over everything above. Empty for the smoke.
COMMAND+="${EXTRA_OVERRIDES:+ ${EXTRA_OVERRIDES}}"

echo "Run dir:   ${RUN_DIR}"
echo "Config:    ${CONFIG}"
echo "Container: ${CONTAINER}"
# The resolved checkpoint dir is the one setting a chain gets silently wrong,
# and it is invisible until the job starts and writes driver_command.sh, by
# which point the allocation is already spent. Print it at submit instead.
echo "Ckpt dir:  ${NT4_CKPT_DIR:-${RUN_DIR}/checkpoints}"

# Pass everything through the environment with a bare --export=ALL. MOUNTS is a
# comma-separated list and --export is itself comma-separated, so naming the
# variables on the sbatch line would split MOUNTS into bogus assignments.
export CONTAINER MOUNTS COMMAND
export BASE_LOG_DIR="${RUN_DIR}/logs"
export GPUS_PER_NODE=4
export HF_HOME="${CACHE_DIR}"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export UV_NO_SYNC=1
# vLLM runs the model in worker processes (Ray actors at TP>1) that build their
# own ModelRegistry, so the NT4 multimodal registration has to arrive as a vLLM
# plugin rather than an in-process call. Discovery is by entry point from this
# directory's .dist-info, and PYTHONPATH is what every worker inherits.

# Every Ray worker inherits this, which is what we want: `site` imports
# sitecustomize per interpreter, and the shim is a no-op in the venvs that do
# not have the broken cutlass overlay.
export NRL_CONTAINER_SHIM=1
# Tells the shim that EXTRA_MOUNT_SPECS already put a coherent cutlass in the
# Megatron venv, so it must not redirect off it. The shim cannot detect this
# itself: the mount replaces the package tree, not the dist-info it reads.
export NRL_CUTLASS_OVERLAY="${NRL_CUTLASS_OVERLAY:-0}"
export PYTHONPATH="${NT4_SHIM_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Defaults to vllm because both shipped recipes generate with vLLM. It used to
# default to megatron, which silently dropped the plugin from PYTHONPATH while
# the recipe went on using vLLM anyway: the workers then fell back to a stock
# processor, expanded <img> into 1369 image tokens that never reached
# input_ids, and every EngineCore died in the profiling run. Jobs 1336216 and
# 1336344 were lost to exactly that, on two different containers.
if [[ "${GEN}" == vllm ]]; then
  export PYTHONPATH="${NT4_VLM_PLUGIN_DIR:-/opt/nemo-rl/vllm_plugin}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export NRL_IGNORE_VERSION_MISMATCH=1
# Names the first module whose forward output goes nonfinite. Off by default:
# it syncs on the device at every layer boundary, though only for the opening
# forward passes.
export NRL_FWD_NAN="${NRL_FWD_NAN:-0}"
# Checks that the context-parallel ranks loaded identical weights. CP shards
# the sequence, not the parameters, so any disagreement is a checkpoint-load
# bug. One all-gather at startup.
export NRL_CP_CHECK="${NRL_CP_CHECK:-0}"
# Names the step inside a ShortcutMoEBlock that first emits a nonfinite
# value: the block runs two CP layout conversions on the same hidden state
# plus one on the mid activation, and a module hook cannot see between them.
export NRL_SHORTCUT_PROBE="${NRL_SHORTCUT_PROBE:-0}"
export NT4_NANO_MIMO_CKPT
export NT4_NANO_MODEL_DIR NT4_NANO_PROCESSOR
# The agent shell runs with HOME=/tmp/githome, which breaks pyxis inside the job.
export HOME="/home/${USER}"

cd "${PROJECT_ROOT}"
# EXCLUDE_NODES is a comma-separated list passed straight to --exclude. It
# exists because node-local squashfs corruption is a recurring failure on this
# cluster -- pyxis reports "No such file or directory" for the container image
# on one node and the job hangs partway through actor startup rather than
# dying (jobs 1163092 and 1164499, both on a single bad node, ~18 min each).
# Setting it at submit is better than `scontrol update ExcNodeList` afterwards,
# which is what had to be done for 1167134 and only works once the job exists.
SBATCH_OPTIONAL=()
if [[ -n "${EXCLUDE_NODES:-}" ]]; then
  SBATCH_OPTIONAL+=(--exclude="${EXCLUDE_NODES}")
fi
# DEPENDENCY chains a continuation job behind a predecessor, for runs longer
# than one wall-clock allocation. Use afterany, not afterok: the predecessor
# ends in TIMEOUT when it uses its whole allocation, which afterok treats as
# failure and would cancel the continuation exactly when it is needed.
if [[ -n "${DEPENDENCY:-}" ]]; then
  SBATCH_OPTIONAL+=(--dependency="${DEPENDENCY}")
fi
# DRYRUN=1 adds --test-only: Slurm validates the whole request and reports when
# it would start, without submitting. Everything above has already run, so the
# recipe lookup, the node count and the printed checkpoint dir are exercised.
if [[ -n "${DRYRUN:-}" ]]; then
  SBATCH_OPTIONAL+=(--test-only)
fi
# Appended here rather than beside the probe that computes it, because
# SBATCH_OPTIONAL is initialised below that point and would discard it.
# ray.sub carries its own #SBATCH --segment, which this overrides when the
# recipe asks for a specific size; a command-line option beats a directive.
if [[ -n "${SEGMENT:-}" ]]; then
  SBATCH_OPTIONAL+=(--segment="${SEGMENT}")
fi
sbatch \
  "${SBATCH_OPTIONAL[@]+"${SBATCH_OPTIONAL[@]}"}" \
  --nodes="${NODES}" \
  --gres=gpu:4 \
  --job-name=nt4nano-clevr-vlm-${GEN}-${RUN_TAG} \
  --account="${SLURM_ACCOUNT:-nemotron_sw_post}" \
  --partition="${SLURM_PARTITION:-batch}" \
  --time="${WALLTIME:-1:30:00}" \
  --output="${RUN_DIR}/logs/slurm-%j.out" \
  --export=ALL \
  ray.sub
