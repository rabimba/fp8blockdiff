"""Cloud TPU v7 (Ironwood / v7x) hardware discovery, VFIO lock hygiene,
persistent XLA cache enforcement, and process-group lifecycle management.

Strictly enforces Non-Negotiable Execution Contract Rules #1, #3, and #4:
- Never substitutes CPU/GPU/NPU execution when a Cloud TPU is required.
- Verifies /dev/vfio/* and libtpu locks before and after worker processes.
- Ensures VLLM_XLA_CACHE_PATH and XLA persistent cache directories are active.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
from typing import Any, IO


REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = REPO_ROOT / "STATUS.json"


class TPUHardwareUnavailableError(RuntimeError):
    """Raised when a Cloud TPU operation is invoked on a host without TPU hardware."""


class TPULockConflictError(RuntimeError):
    """Raised when /dev/vfio/* or libtpu locks are held by another process."""


@dataclasses.dataclass(frozen=True)
class TPUTopology:
    """Detected Cloud TPU slice topology and memory properties."""

    chip_count: int
    device_kind: str
    hbm_bytes_per_chip: int
    vfio_devices: tuple[str, ...]
    backend_provider: str
    hostname: str
    python_version: str

    @property
    def hbm_gb_per_chip(self) -> float:
        return round(self.hbm_bytes_per_chip / (1024**3), 2)

    @property
    def total_hbm_gb(self) -> float:
        return round((self.hbm_bytes_per_chip * self.chip_count) / (1024**3), 2)

    @property
    def is_ironwood_v7(self) -> bool:
        lowered = self.device_kind.lower()
        return any(tok in lowered for tok in ("v7", "7x", "ironwood", "tpu7"))

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["hbm_gb_per_chip"] = self.hbm_gb_per_chip
        data["total_hbm_gb"] = self.total_hbm_gb
        data["is_ironwood_v7"] = self.is_ironwood_v7
        return data


def ensure_persistent_cache_env() -> dict[str, str]:
    """Export persistent compilation cache directories on fast local disk (Contract #4)."""
    candidates = [
        Path("/mnt/localssd/fp8blockdiff_cache"),
        Path("/scratch/fp8blockdiff_cache"),
        Path(f"/var/tmp/fp8blockdiff_cache_{os.environ.get('USER', 'tpu')}"),
    ]
    cache_base: Path | None = None
    existing = os.environ.get("FP8_BLOCKDIFF_CACHE_BASE")
    if existing:
        cache_base = Path(existing)
    else:
        for cand in candidates:
            parent = cand.parent
            if parent.exists() and os.access(parent, os.W_OK):
                cache_base = cand
                break
        if cache_base is None:
            cache_base = REPO_ROOT / ".cache" / "xla_persistent"

    env_map = {
        "FP8_BLOCKDIFF_CACHE_BASE": str(cache_base),
        "VLLM_XLA_CACHE_PATH": os.environ.get(
            "VLLM_XLA_CACHE_PATH", str(cache_base / "vllm_xla_cache")
        ),
        "XLA_CACHE_DIR": os.environ.get("XLA_CACHE_DIR", str(cache_base / "xla_cache")),
        "JAX_COMPILATION_CACHE_DIR": os.environ.get(
            "JAX_COMPILATION_CACHE_DIR", str(cache_base / "jax_cache")
        ),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": os.environ.get(
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0"
        ),
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR", str(cache_base / "torchinductor_cache")
        ),
        "PJRT_DEVICE": os.environ.get("PJRT_DEVICE", "TPU"),
    }
    for key, val in env_map.items():
        os.environ[key] = val
        if key.endswith("_PATH") or key.endswith("_DIR") or key.endswith("_BASE"):
            Path(val).mkdir(parents=True, exist_ok=True)
    return env_map


def list_vfio_tpu_nodes() -> list[str]:
    """Return numeric /dev/vfio/<N> group nodes exposed to Cloud TPU VMs."""
    nodes: list[str] = []
    for path in sorted(glob.glob("/dev/vfio/*")):
        base = os.path.basename(path)
        if base.isdigit():
            nodes.append(path)
    return nodes


def scan_tpu_lock_holders(exclude_current_process: bool = True) -> list[dict[str, Any]]:
    """Inspect /proc/*/fd directly to identify any process holding /dev/vfio/<N> or libtpu locks."""
    my_pid = os.getpid()
    holders: dict[int, dict[str, Any]] = {}
    target_prefixes = tuple(list_vfio_tpu_nodes() + ["/tmp/libtpu_lockfile"])
    if not target_prefixes:
        return []

    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        parts = fd_dir.split("/")
        if len(parts) < 3 or not parts[2].isdigit():
            continue
        pid = int(parts[2])
        if exclude_current_process and pid == my_pid:
            continue
        try:
            entries = os.listdir(fd_dir)
        except OSError:
            continue
        for entry in entries:
            try:
                target = os.readlink(os.path.join(fd_dir, entry))
            except OSError:
                continue
            if any(target == p or target.startswith(p + " ") for p in target_prefixes):
                if pid not in holders:
                    cmdline = _read_proc_cmdline(pid)
                    holders[pid] = {
                        "pid": pid,
                        "cmdline": cmdline,
                        "locked_paths": [],
                    }
                if target not in holders[pid]["locked_paths"]:
                    holders[pid]["locked_paths"].append(target)

    return sorted(holders.values(), key=lambda item: item["pid"])


def _read_proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return "<exited>"


def cleanup_stale_tpu_locks(allow_kill: bool = True, timeout_sec: float = 8.0) -> list[int]:
    """Verify no stale Python/Ray/vLLM processes hold /dev/vfio/* or libtpu locks (Contract #3)."""
    killed_pids: list[int] = []
    my_pid = os.getpid()
    my_uid = os.getuid()

    # Also detect stale background vllm / ray worker processes even before they grab VFIO
    stale_worker_markers = ("vllm.entrypoints", "vllm serve", "EngineCore", "ray::", "VLLM::")
    candidate_pids: set[int] = set()

    for holder in scan_tpu_lock_holders(exclude_current_process=True):
        candidate_pids.add(int(holder["pid"]))

    for proc_dir in glob.glob("/proc/[0-9]*"):
        base = os.path.basename(proc_dir)
        if not base.isdigit():
            continue
        pid = int(base)
        if pid == my_pid:
            continue
        try:
            st = os.stat(proc_dir)
        except OSError:
            continue
        if st.st_uid != my_uid:
            continue
        cmdline = _read_proc_cmdline(pid)
        if any(marker in cmdline for marker in stale_worker_markers):
            candidate_pids.add(pid)

    if candidate_pids and allow_kill:
        for pid in sorted(candidate_pids):
            try:
                os.kill(pid, signal.SIGTERM)
                killed_pids.append(pid)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            alive = [p for p in killed_pids if Path(f"/proc/{p}").exists()]
            if not alive:
                break
            time.sleep(0.25)

        for pid in killed_pids:
            if Path(f"/proc/{pid}").exists():
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        time.sleep(0.25)

    # Clean up stale unlocked /tmp/libtpu_lockfile if owned by current user
    lockfile = Path("/tmp/libtpu_lockfile")
    if lockfile.exists() and not scan_tpu_lock_holders(exclude_current_process=False):
        try:
            if lockfile.stat().st_uid == my_uid:
                lockfile.unlink(missing_ok=True)
        except OSError:
            pass

    remaining = scan_tpu_lock_holders(exclude_current_process=True)
    if remaining:
        raise TPULockConflictError(
            f"TPU VFIO / libtpu locks are actively held by other processes: {json.dumps(remaining)}"
        )
    return killed_pids


def probe_tpu_topology(strict: bool = True) -> TPUTopology | None:
    """Detect Cloud TPU topology via torch_tpu and/or jax[tpu].

    Never treats CPU, CUDA GPUs, or laptop NPUs (/dev/accel/accel0) as Cloud TPUs.
    """
    vfio_nodes = tuple(list_vfio_tpu_nodes())
    hostname = socket.gethostname()
    py_ver = platform.python_version()

    # 1. Probe torch_tpu native backend first
    try:
        import torch
        import torch_tpu  # noqa: F401

        if hasattr(torch, "tpu") and torch.tpu.is_available():
            count = int(torch.tpu.device_count())
            if count > 0:
                kind = "Cloud TPU (torch_tpu)"
                hbm_bytes = 192 * (1024**3)  # fallback if property API omits total_memory
                if hasattr(torch.tpu, "get_device_name"):
                    try:
                        kind = str(torch.tpu.get_device_name(0))
                    except Exception:
                        pass
                if hasattr(torch.tpu, "get_device_properties"):
                    try:
                        props = torch.tpu.get_device_properties(0)
                        if hasattr(props, "name") and props.name:
                            kind = str(props.name)
                        if hasattr(props, "total_memory") and int(props.total_memory) > 0:
                            hbm_bytes = int(props.total_memory)
                    except Exception:
                        pass
                # Cross-enrich device kind & exact HBM bytes from jax.devices() if available in-proc
                # Note: Avoid initializing both PJRT clients concurrently if they contend on locks;
                # only enrich from sysfs / tpu-info metadata if needed.
                enriched_kind, enriched_hbm = _enrich_from_sysfs_or_env(kind, hbm_bytes)
                return TPUTopology(
                    chip_count=count,
                    device_kind=enriched_kind,
                    hbm_bytes_per_chip=enriched_hbm,
                    vfio_devices=vfio_nodes,
                    backend_provider="torch_tpu",
                    hostname=hostname,
                    python_version=py_ver,
                )
    except ImportError:
        pass

    # 2. Probe JAX TPU backend if torch_tpu is not yet imported/installed
    try:
        import jax

        tpu_devs = [d for d in jax.devices() if getattr(d, "platform", "") == "tpu"]
        if tpu_devs:
            kind = str(getattr(tpu_devs[0], "device_kind", "Cloud TPU"))
            hbm_bytes = 0
            if hasattr(tpu_devs[0], "memory_stats"):
                try:
                    stats = tpu_devs[0].memory_stats() or {}
                    hbm_bytes = int(stats.get("bytes_limit", 0))
                except Exception:
                    hbm_bytes = 0
            enriched_kind, enriched_hbm = _enrich_from_sysfs_or_env(
                kind, hbm_bytes or 192 * (1024**3)
            )
            return TPUTopology(
                chip_count=len(tpu_devs),
                device_kind=enriched_kind,
                hbm_bytes_per_chip=enriched_hbm,
                vfio_devices=vfio_nodes,
                backend_provider="jax_tpu",
                hostname=hostname,
                python_version=py_ver,
            )
    except Exception:
        pass

    if strict:
        raise TPUHardwareUnavailableError(
            f"No Cloud TPU device detected on host '{hostname}' "
            f"(VFIO numeric nodes={list(vfio_nodes)}). "
            "Contract #1 strictly forbids substituting CPU/GPU/NPU execution for TPU v7 runs. "
            "Clone and execute this phase on the Cloud TPU v7 (Ironwood / v7x) VM."
        )
    return None


def require_tpu_device() -> TPUTopology:
    """Strictly require an active Cloud TPU slice; raise TPUHardwareUnavailableError otherwise."""
    topo = probe_tpu_topology(strict=True)
    assert topo is not None
    return topo


def _enrich_from_sysfs_or_env(default_kind: str, default_hbm_bytes: int) -> tuple[str, int]:
    """Read GCE metadata or TPU env hints without grabbing libtpu locks."""
    env_type = os.environ.get("TPU_ACCELERATOR_TYPE", "") or os.environ.get("TPU_TYPE", "")
    kind = default_kind
    if env_type and env_type.lower() not in kind.lower():
        kind = f"{default_kind} ({env_type})"
    hbm = default_hbm_bytes
    if hbm <= 0 and any(k in kind.lower() for k in ("v7", "7x", "ironwood")):
        hbm = 192 * (1024**3)
    return kind, hbm


class ManagedProcessGroup:
    """Context manager launching a child server in its own process group and guaranteeing
    clean SIGTERM -> SIGKILL teardown plus TPU VFIO lock release in a finally block.
    """

    def __init__(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        cwd: Path | None = None,
        shutdown_grace_sec: float = 10.0,
    ) -> None:
        self.cmd = list(cmd)
        self.env = dict(os.environ if env is None else env)
        self.log_path = log_path
        self.cwd = cwd or REPO_ROOT
        self.shutdown_grace_sec = shutdown_grace_sec
        self.proc: subprocess.Popen[str] | None = None
        self._log_handle: IO[str] | None = None

    def __enter__(self) -> subprocess.Popen[str]:
        cleanup_stale_tpu_locks(allow_kill=True)
        ensure_persistent_cache_env()
        stdout_target: int | IO[str] = subprocess.PIPE
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("w", encoding="utf-8", buffering=1)
            stdout_target = self._log_handle

        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            env=self.env,
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        return self.proc

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self.terminate()
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            cleanup_stale_tpu_locks(allow_kill=True)

    def terminate(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=self.shutdown_grace_sec)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(self.proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=5.0)


def snapshot_xla_compilation_metrics() -> dict[str, int]:
    """Capture XLA graph compilation counters across torch_tpu, torch._dynamo, and disk cache."""
    metrics: dict[str, int] = {
        "dynamo_frame_compiles": 0,
        "tpu_backend_compiles": 0,
        "xla_cache_files": 0,
    }
    try:
        import torch._dynamo.utils as dynamo_utils

        counters = getattr(dynamo_utils, "counters", None)
        if counters and "stats" in counters:
            metrics["dynamo_frame_compiles"] = int(
                counters["stats"].get("unique_graphs", 0)
            )
    except Exception:
        pass

    # Query torch_tpu / torch_xla metric report if exposed
    try:
        import torch

        if hasattr(torch, "tpu"):
            for attr in ("metrics_report", "get_compilation_count", "_compilation_count"):
                fn = getattr(torch.tpu, attr, None)
                if callable(fn):
                    res = fn()
                    if isinstance(res, int):
                        metrics["tpu_backend_compiles"] = res
                        break
                    if isinstance(res, str):
                        metrics["tpu_backend_compiles"] = _extract_compile_count_from_report(res)
                        break
    except Exception:
        pass

    # Count persistent cache entries on disk
    for env_key in ("XLA_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR", "VLLM_XLA_CACHE_PATH"):
        p = os.environ.get(env_key)
        if p and os.path.isdir(p):
            for _root, _dirs, files in os.walk(p):
                metrics["xla_cache_files"] += len(files)

    return metrics


def _extract_compile_count_from_report(report: str) -> int:
    total = 0
    lines = report.splitlines()
    for idx, line in enumerate(lines):
        if "CompileTime" in line or "UncachedCompile" in line or "XlaCompile" in line:
            for look_ahead in lines[idx : min(idx + 4, len(lines))]:
                if "TotalSamples:" in look_ahead:
                    tail = look_ahead.split("TotalSamples:", 1)[1].strip().split()[0]
                    if tail.isdigit():
                        total += int(tail)
                        break
    return total


def update_phase_status(
    phase_key: str,
    status: str,
    overall_state: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically update STATUS.json so takeover agents and validators track exact gate state."""
    if STATUS_PATH.exists():
        data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    else:
        data = {"phases": {}}

    if overall_state is not None:
        data["overall_state"] = overall_state

    phases = data.setdefault("phases", {})
    entry = phases.setdefault(phase_key, {})
    entry["status"] = status
    if status == "passed":
        entry["completed_at"] = datetime.now(timezone.utc).isoformat()
    if extra:
        entry.update(extra)

    tmp_path = STATUS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(STATUS_PATH)
