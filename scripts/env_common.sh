#!/usr/bin/env bash
# Shared environment hygiene, persistent XLA compilation caches, and TPU VFIO lock guards.
# Source this file before launching any server, pytest suite, or benchmark process:
#   source scripts/env_common.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# 1. Persistent Compilation Cache Setup (Contract #4)
# Prefer fast local NVMe SSD or GKE tmpfs (/tmp/tpu_logs) if present, else /var/tmp.
if [[ -d "/mnt/localssd" && -w "/mnt/localssd" ]]; then
  CACHE_BASE="/mnt/localssd/fp8blockdiff_cache"
elif [[ -d "/scratch" && -w "/scratch" ]]; then
  CACHE_BASE="/scratch/fp8blockdiff_cache"
elif [[ -d "/tmp/tpu_logs" && -w "/tmp/tpu_logs" ]]; then
  CACHE_BASE="/tmp/tpu_logs/fp8blockdiff_cache"
else
  CACHE_BASE="/var/tmp/fp8blockdiff_cache_${USER:-tpu}"
fi

export FP8_BLOCKDIFF_CACHE_BASE="${CACHE_BASE}"
export VLLM_XLA_CACHE_PATH="${CACHE_BASE}/vllm_xla_cache"
export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm_cache"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${CACHE_BASE}/tier3_pjrt_cache"
export XLA_CACHE_DIR="${CACHE_BASE}/xla_cache"
export JAX_COMPILATION_CACHE_DIR="${CACHE_BASE}/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor_cache"
export HF_HOME="${HF_HOME:-${CACHE_BASE}/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export FP8_BLOCKDIFF_TP_SIZE="${FP8_BLOCKDIFF_TP_SIZE:-1}"
if [[ "${FP8_BLOCKDIFF_TP_SIZE}" == "1" ]]; then
  export FP8_BLOCKDIFF_CHIP_ID="${FP8_BLOCKDIFF_CHIP_ID:-1}"
  export TPU_VISIBLE_CHIPS="${FP8_BLOCKDIFF_CHIP_ID}"
  export TPU_VISIBLE_DEVICES="${FP8_BLOCKDIFF_CHIP_ID}"
  export ALLOW_MULTIPLE_LIBTPU_LOAD="true"
  export TPU_CHIPS_PER_HOST_BOUNDS="1,1,1"
  export CHIPS_PER_HOST_BOUNDS="1,1,1"
  export TPU_HOST_BOUNDS="1,1,1"
  export TPU_PROCESS_PORT="${FP8_BLOCKDIFF_TPU_PORT:-8476}"
  export TPU_PROCESS_ADDRESSES="localhost:${TPU_PROCESS_PORT}"
  export TPU_RUNTIME_METRICS_PORTS="8441,8442"
fi
if [[ -f "/etc/ssl/certs/ca-certificates.crt" ]]; then
  export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
  export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
  export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
fi

mkdir -p \
  "${VLLM_XLA_CACHE_PATH}" \
  "${VLLM_CACHE_ROOT}" \
  "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" \
  "${CACHE_BASE}/torch_tpu_shm_cache" \
  "${XLA_CACHE_DIR}" \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${HF_HOME}" \
  "${REPO_ROOT}/artifacts/logs" \
  "${REPO_ROOT}/artifacts/traces" \
  "${REPO_ROOT}/artifacts/plots"

# Redirect /dev/shm/torch_tpu_cache to large tmpfs cache when /dev/shm is small (64 MiB)
if [[ -d "/dev/shm" && -w "/dev/shm" && ! -L "/dev/shm/torch_tpu_cache" ]]; then
  rm -rf "/dev/shm/torch_tpu_cache" 2>/dev/null || true
  ln -s "${CACHE_BASE}/torch_tpu_shm_cache" "/dev/shm/torch_tpu_cache" 2>/dev/null || true
fi

# Ensure XLA persistent disk cache flags are appended idempotently
if [[ "${XLA_FLAGS:-}" != *"xla_gpu_kernel_cache_file"* && "${LIBTPU_INIT_ARGS:-}" == "" ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}"
fi

# 2. TPU Lock Hygiene Helpers (Contract #3)
list_tpu_lock_holders() {
  local holders=""
  local cand_pids=""
  cand_pids="$(pgrep -u "${USER:-$(id -un)}" -f "vllm\.entrypoints|vllm serve|EngineCore|ray::|VLLM::|fp8blockdiff" 2>/dev/null || true)"
  for pid in ${cand_pids}; do
    if ls -l "/proc/${pid}/fd" 2>/dev/null | grep -q "/dev/vfio/"; then
      holders="${holders} ${pid}"
    fi
  done
  echo "${holders}" | tr ' ' '\n' | sed '/^$/d' | sort -u
}

cleanup_stale_tpu_processes() {
  echo "[env_common] Checking for stale vLLM / Ray / TPU lock-holding worker processes..."
  local self_pid="$$"
  local parent_pid="${PPID:-0}"

  # Gracefully stop stale vLLM / Ray worker processes owned by current user
  local stale_pids
  stale_pids="$(pgrep -u "${USER:-$(id -un)}" -f "vllm\.entrypoints|vllm serve|EngineCore|ray::|VLLM::" 2>/dev/null || true)"
  for pid in ${stale_pids}; do
    if [[ "${pid}" != "${self_pid}" && "${pid}" != "${parent_pid}" ]]; then
      echo "[env_common] Sending SIGTERM to stale worker PID ${pid}..."
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  sleep 1

  # Force kill if any still remain
  stale_pids="$(pgrep -u "${USER:-$(id -un)}" -f "vllm\.entrypoints|vllm serve|EngineCore|ray::|VLLM::" 2>/dev/null || true)"
  for pid in ${stale_pids}; do
    if [[ "${pid}" != "${self_pid}" && "${pid}" != "${parent_pid}" ]]; then
      echo "[env_common] Sending SIGKILL to lingering worker PID ${pid}..."
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done

  # Check VFIO / libtpu lock holders
  local vfio_pids
  vfio_pids="$(list_tpu_lock_holders)"
  for pid in ${vfio_pids}; do
    if [[ -n "${pid}" && "${pid}" != "${self_pid}" && "${pid}" != "${parent_pid}" ]]; then
      local owner
      owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
      if [[ "${owner}" == "${USER:-$(id -un)}" ]]; then
        echo "[env_common] Releasing TPU lock held by PID ${pid}..."
        kill -TERM "${pid}" 2>/dev/null || true
        sleep 1
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    fi
  done

  # Remove stale unlocked /tmp/libtpu_lockfile if no live process holds it
  if [[ "${ALLOW_MULTIPLE_LIBTPU_LOAD:-}" != "true" && "${ALLOW_MULTIPLE_LIBTPU_LOAD:-}" != "1" && -e "/tmp/libtpu_lockfile" ]]; then
    local remaining
    remaining="$(lsof -t /tmp/libtpu_lockfile 2>/dev/null || true)"
    if [[ -z "${remaining}" && -O "/tmp/libtpu_lockfile" ]]; then
      rm -f /tmp/libtpu_lockfile
    fi
  fi

  assert_tpu_locks_free
}

assert_tpu_locks_free() {
  local self_pid="$$"
  local holders
  holders="$(list_tpu_lock_holders)"
  local active=""
  for pid in ${holders}; do
    if [[ -n "${pid}" && "${pid}" != "${self_pid}" ]]; then
      active="${active} ${pid}"
    fi
  done
  if [[ -n "${active}" ]]; then
    echo "[env_common] ERROR: TPU VFIO / libtpu locks are still held by PIDs:${active}" >&2
    ps -fp ${active} >&2 || true
    return 1
  fi
  echo "[env_common] TPU lock hygiene verified (VLLM_XLA_CACHE_PATH=${VLLM_XLA_CACHE_PATH})."
}
