#!/usr/bin/env bash
# Single-entry launcher for the full SWE-Gym Slime GRPO example.
#
# This script bootstraps the pieces that are safe to automate on a cluster:
# external checkouts, local editable installs, Slime/SGLang patches,
# SWE-Gym train JSONL data, shared agent CLI assets, base runtime images,
# Megatron weight conversion, and finally the Polar + Slime training run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
SLIME_REF="${SLIME_REF:-v0.3.0}"

MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/tmp/Megatron-LM-slime-v0.3.0}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
# Slime v0.3.0 imports megatron.training.tokenizer which 26.04+ removed.
# patch_megatron_tokenizer_shim.sh adds the missing _vocab_size_with_padding.
MEGATRON_REF="${MEGATRON_REF:-26.04-alpha.rc1}"
# SWE-Gym's fork of the SWE-bench harness (grades the SWE-Gym instances). Not on
# PyPI, so installed from git; commit-pinned for reproducibility.
SWEGYM_PACKAGE_SPEC="${SWEGYM_PACKAGE_SPEC:-swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/data/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${TORCH_DIST_DIR:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${REF_LOAD}}"
RUN_ID="${RUN_ID:-${WANDB_RUN_ID:-swegym-slime-grpo-$(date -u +%Y%m%dT%H%M%SZ)}}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/tmp/ckpt/swegym_slime_grpo_qwen35_4b}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${PROJECT_ROOT}/tmp/swegym_apptainer_images}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${PROJECT_ROOT}/tmp/apptainer_cache}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${PROJECT_ROOT}/tmp/apptainer_tmp}"
POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-$(command -v apptainer || echo /usr/bin/apptainer)}"

# Number of SWE-Gym tasks: 16 (local Docker images) or 293 (full dataset).
# When NUM_TASKS=16 the pipeline expects Docker images already loaded (via
# ``docker load``) and uses build_sifs_from_docker.py to convert them into
# Apptainer SIF files instead of pulling from the remote registry.
NUM_TASKS="${NUM_TASKS:-64}"
case "${NUM_TASKS}" in
    16) PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/swegym_train_16.jsonl}" ;;
    64) PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/swegym_train_64.jsonl}" ;;
    *)  PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/swegym_train_293.jsonl}" ;;
esac

INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-0}"  # TE + FLA + flash-attn (SM100 only)
FLASH_LINEAR_ATTENTION_VERSION="${FLASH_LINEAR_ATTENTION_VERSION:-0.5.0}"
MBRIDGE_VERSION="${MBRIDGE_VERSION:-0.15.1}"  # HF<->Megatron weight bridge (slime conversion)
APPLY_SGLANG_PATCH="${APPLY_SGLANG_PATCH:-0}"
PREPARE_IMAGES="${PREPARE_IMAGES:-0}"
APPTAINER_PREPARE_JOBS="${APPTAINER_PREPARE_JOBS:-2}"
CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-6}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
export PROMPT_DATA APPTAINER_CACHEDIR APPTAINER_TMPDIR POLAR_APPTAINER_BIN
export WANDB_API_KEY=local-cbfc7c82164f68aa14bcadfb7efafcee35f99dbf
export ACTOR_NUM_GPUS_PER_NODE ROLLOUT_NUM_GPUS MAX_TOKENS_PER_GPU SGLANG_CONTEXT_LENGTH

# Redirect all output to a timestamped log file.
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
exec &> >(tee -a "${LOG_DIR}/launch_e2e_$(date -u +%Y%m%dT%H%M%SZ).log")

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

clone_if_missing() {
    local name="$1"
    local repo="$2"
    local ref="$3"
    local dest="$4"
    if [ -d "${dest}/.git" ]; then
        echo "${name} checkout exists: ${dest}"
        return
    fi
    if [ -e "${dest}" ]; then
        echo "ERROR: ${name} path exists but is not a git checkout: ${dest}" >&2
        exit 1
    fi
    echo "Cloning ${name} ${ref} -> ${dest}"
    git clone --branch "${ref}" --depth 1 "${repo}" "${dest}"
}

checkpoint_ready() {
    [ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]
}

maybe_login_wandb() {
    if [ -z "${WANDB_API_KEY:-}" ]; then
        return
    fi
    "${PYTHON_BIN}" - <<'PY'
import os

try:
    import wandb
except Exception:
    raise SystemExit(0)

key = os.environ.get("WANDB_API_KEY")
if key and hasattr(wandb, "login"):
    wandb.login(key=key, relogin=True)
PY
}

flash_attn2_ready() {
    "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    installed = version("flash-attn")
except PackageNotFoundError:
    raise SystemExit(1)

if installed != "2.7.4.post1":
    raise SystemExit(1)

try:
    import torch  # noqa: F401
    import flash_attn_2_cuda  # noqa: F401
    import flash_attn.flash_attn_interface  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
}

swegym_harness_ready() {
    "${PYTHON_BIN}" - <<'PY'
try:
    from swegym.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swegym.harness.grading import get_eval_report  # noqa: F401
    from swegym.harness.test_spec import make_test_spec  # noqa: F401
except Exception:
    raise SystemExit(1)

needed = {"dask/dask", "python/mypy", "pandas-dev/pandas"}
if not needed.issubset(MAP_REPO_VERSION_TO_SPECS):
    raise SystemExit(1)
PY
}

ensure_swegym_harness() {
    if swegym_harness_ready; then
        echo "SWE-Gym harness package present; skipping."
    else
        echo "Installing SWE-Gym harness package..."
        uv pip install --python "${PYTHON_BIN}" "${SWEGYM_PACKAGE_SPEC}"
    fi
}

# Install the GPU training-stack extras the editable installs do NOT pull.
# Idempotent — skips whatever is already importable.
#   - Transformer Engine: required by Megatron on ANY GPU (its torch bindings build
#     from source and need cuDNN headers from the pip nvidia-cudnn package).
#   - Flash Linear Attention: required by Qwen3.5 GatedDeltaNet linear-attention layers.
#   - flash-attn 2.x from source: ONLY on SM100/B200, where TE's cuDNN backend has no
#     head_dim=256 kernel. Built for the detected arch; skipped on every other GPU so
#     this script stays safe on H100 etc. (no wasted/failed builds).
ensure_training_stack() {
    local cuda_home cudnn_path cc=""
    cuda_home="${CUDA_HOME:-/usr/local/cuda}"
    if [ ! -d "$cuda_home" ] && command -v nvcc >/dev/null 2>&1; then
        cuda_home="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
    cudnn_path="$("${PYTHON_BIN}" -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])' 2>/dev/null || true)"

    # --- Transformer Engine 2.5.0 (general) ---
    if LD_LIBRARY_PATH="${cudnn_path}/lib:${LD_LIBRARY_PATH:-}" \
        "${PYTHON_BIN}" -c "import transformer_engine.pytorch" >/dev/null 2>&1; then
        echo "Transformer Engine present; skipping."
    else
        if ! command -v nvcc >/dev/null 2>&1; then
            echo "ERROR: nvcc not found — needed to build transformer-engine-torch." >&2
            echo "  Install the CUDA toolkit (or set CUDA_HOME), then re-run." >&2
            exit 1
        fi
        if [ -z "$cudnn_path" ]; then
            echo "ERROR: pip 'nvidia-cudnn' not found in venv — is torch a CUDA build?" >&2
            echo "  TE's source build needs its cuDNN headers; fix torch first (see README)." >&2
            exit 1
        fi
        echo "Installing Transformer Engine 2.5.0 (building transformer-engine-torch)..."
        uv pip install --python "${PYTHON_BIN}" ninja pybind11 setuptools wheel >/dev/null 2>&1 || true
        CUDA_HOME="$cuda_home" \
        CPATH="${cudnn_path}/include:${cuda_home}/include:${CPATH:-}" \
        LIBRARY_PATH="${cudnn_path}/lib:${cuda_home}/lib64:${LIBRARY_PATH:-}" \
            uv pip install --python "${PYTHON_BIN}" --no-build-isolation "transformer-engine[pytorch]==2.5.0"
    fi

    # --- Flash Linear Attention (Qwen3.5 linear-attention layers) ---
    if "${PYTHON_BIN}" -c \
        "from fla.modules import FusedRMSNormGated, ShortConvolution; from fla.ops.gated_delta_rule import chunk_gated_delta_rule" \
        >/dev/null 2>&1; then
        echo "Flash Linear Attention present; skipping."
    else
        echo "Installing Flash Linear Attention ${FLASH_LINEAR_ATTENTION_VERSION}..."
        uv pip install --python "${PYTHON_BIN}" "flash-linear-attention==${FLASH_LINEAR_ATTENTION_VERSION}"
    fi

    # --- flash-attn 2.x from source: SM100 (B200) only ---
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')" || true
    if [ "$cc" = "10.0" ]; then
        if flash_attn2_ready; then
            echo "flash-attn (2.x) present; skipping."
        else
            echo "SM100 detected: building flash-attn 2.7.4.post1 (head_dim=256 fallback)..."
            TORCH_CUDA_ARCH_LIST=10.0 FLASH_ATTN_CUDA_ARCHS=100 FLASH_ATTENTION_FORCE_BUILD=TRUE \
                MAX_JOBS="${FA_MAX_JOBS:-32}" NVCC_THREADS=4 \
                uv pip install --python "${PYTHON_BIN}" --no-build-isolation flash-attn==2.7.4.post1
        fi
    else
        echo "compute_cap=${cc:-unknown} is not SM100; skipping flash-attn 2.x source build."
        echo "  (TE's cuDNN backend serves head_dim=256 off SM100. If training later aborts with"
        echo "   'No dot product attention backend', build flash-attn 2.x for your arch — see README.)"
    fi
}

require_cmd git
require_cmd "${PYTHON_BIN}"
require_cmd uv
require_cmd "${POLAR_APPTAINER_BIN}"
require_cmd envsubst  # run.sh uses it to render YAML templates

clone_if_missing "Slime" "${SLIME_REPO}" "${SLIME_REF}" "${SLIME_DIR}"
clone_if_missing "Megatron-LM" "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_DIR}"

SLIME_DIR="${SLIME_DIR}" bash "${PROJECT_ROOT}/scripts/patch/patch_slime_router_tokens.sh"
# Megatron 26.04+ removed megatron.training.tokenizer; add the one function Slime needs.
MEGATRON_DIR="${MEGATRON_DIR}" bash "${PROJECT_ROOT}/scripts/patch/patch_megatron_tokenizer_shim.sh"

if [ "${INSTALL_EDITABLE}" = "1" ]; then
    # [swebench] is load-bearing even though swegym (installed below) does the actual
    # grading: swegym is a swebench fork that ships NO deps of its own, so it reuses
    # swebench's dependency tree (datasets, docker, ghapi, unidiff, dotenv, requests...).
    # So we need both — [swebench] for the deps, swegym for the SWE-Gym repo specs.
    uv pip install --python "${PYTHON_BIN}" -e ".[swebench]"
    uv pip install --python "${PYTHON_BIN}" -e "${SLIME_DIR}"
    # Slime v0.3.0 references args.enable_gloo_process_groups which 26.04 renamed
    # to disable_gloo_process_groups (negated). Patch all occurrences in slime.
    for f in "${SLIME_DIR}/slime/backends/megatron_utils/initialize.py" \
             "${SLIME_DIR}/slime/backends/megatron_utils/model.py"; do
        sed -i 's/args\.enable_gloo_process_groups/not getattr(args, "disable_gloo_process_groups", False)/' "$f"
    done
    uv pip install --python "${PYTHON_BIN}" -e "${MEGATRON_DIR}"
    # mbridge: HF<->Megatron weight map slime needs to convert Qwen3.5 (slime_plugins.mbridge).
    # --no-deps keeps the pinned torch / TE / flash-attn stack untouched.
    uv pip install --python "${PYTHON_BIN}" --no-deps "mbridge==${MBRIDGE_VERSION}"
    ensure_swegym_harness
    # Ensure sitecustomize.py auto-loads the swegym cache patch for every
    # Python process (Polar gateway, rollout server, Ray workers).
    cat > "$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py" << 'PYEOF'
try:
    from polar.patches.swegym_env_cache import apply
    apply()
except Exception:
    pass
PYEOF
fi

if [ "${INSTALL_TRAINING_STACK}" = "1" ]; then
    ensure_training_stack
fi

if [ "${APPLY_SGLANG_PATCH}" = "1" ]; then
    bash "${PROJECT_ROOT}/scripts/patch/patch_sglang_0513_token_metadata.sh"
fi

case "${NUM_TASKS}" in
    16) "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data_16.py" ;;
    64) "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data_64.py" ;;
    *)  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" ;;
esac

if [ "${PREPARE_IMAGES}" = "1" ]; then
    if [ "${NUM_TASKS}" = "16" ] || [ "${NUM_TASKS}" = "64" ]; then
        # 16/64-task mode: convert locally-loaded Docker images to Apptainer SIF
        "${PYTHON_BIN}" "${SCRIPT_DIR}/build_sifs_from_docker.py" \
            --agent-cli-dir "${AGENT_CLI_DIR}" \
            --node-dist "${NODE_DIST:-${PROJECT_ROOT}/required_packages/node-v22.11.0-linux-x64.tar.xz}" \
            --image-dir "${APPTAINER_IMAGE_DIR}" \
            --cache-dir "${APPTAINER_CACHEDIR}" \
            --tmp-dir "${APPTAINER_TMPDIR}" \
            --jobs "${APPTAINER_PREPARE_JOBS}"
    else
        # Full 293 mode: pull from remote registry (original behavior)
        "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_apptainer_images.py" \
            --agent-cli-dir "${AGENT_CLI_DIR}" \
            --image-dir "${APPTAINER_IMAGE_DIR}" \
            --cache-dir "${APPTAINER_CACHEDIR}" \
            --tmp-dir "${APPTAINER_TMPDIR}" \
            --jobs "${APPTAINER_PREPARE_JOBS}"
    fi
fi

if [ "${CONVERT_WEIGHTS}" = "1" ] || { [ "${CONVERT_WEIGHTS}" = "auto" ] && ! checkpoint_ready; }; then
    HF_CHECKPOINT="${HF_CHECKPOINT}" \
    TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
    SLIME_DIR="${SLIME_DIR}" \
    MEGATRON_DIR="${MEGATRON_DIR}" \
        bash "${SCRIPT_DIR}/convert_weights.sh"
fi

maybe_login_wandb

HF_CHECKPOINT="${HF_CHECKPOINT}" \
REF_LOAD="${REF_LOAD}" \
TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
SAVE_DIR="${SAVE_DIR}" \
RUN_ID="${RUN_ID}" \
SAVE_ROOT="${SAVE_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
SLIME_DIR="${SLIME_DIR}" \
MEGATRON_DIR="${MEGATRON_DIR}" \
PROMPT_DATA="${PROMPT_DATA}" \
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE}" \
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS}" \
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU}" \
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH}" \
AGENT_CLI_DIR="${AGENT_CLI_DIR}" \
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR}" \
    bash "${SCRIPT_DIR}/run.sh"
