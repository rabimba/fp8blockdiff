#!/usr/bin/env python3
"""PHASE 0 — HARDWARE & ECOSYSTEM DISCOVERY GATE.

Executes strictly on the Cloud TPU v7 (Ironwood / v7x) VM:
1. Verifies Python >= 3.12, TPU v7 chip count, topology, and ~192 GB/chip HBM3E.
2. Audits (and optionally installs missing) packages: torch, torch_tpu,
   vllm / vllm-torchtpu, jax, pallas, openxla/tokamax, tpu-info, xprof.
3. Executes real BF16 and FP8 (float8_e4m3fn) matrix multiplications on
   torch.device("tpu") under both eager execution and torch.compile(backend="tpu").
4. Introspects installed vllm / vllm-torchtpu / torch_tpu / tokamax source trees
   for speculative decoding (DFlash, DSpark, EAGLE-3, n-gram) and
   torch.tpu.pallas.jax_op signatures, writing docs/api_discovery.md and
   artifacts/logs/phase0_probe.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata as importlib_metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tpu_utils import (  # noqa: E402
    cleanup_stale_tpu_locks,
    ensure_persistent_cache_env,
    require_tpu_device,
    update_phase_status,
)


OFFICIAL_INSTALL_RECIPES: list[tuple[str, list[str]]] = [
    (
        "vllm-torchtpu",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "vllm-torchtpu",
        ],
    ),
    (
        "tokamax",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "tokamax",
        ],
    ),
    (
        "tokamax_git_fallback",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/openxla/tokamax.git",
        ],
    ),
    (
        "tpu-info",
        [sys.executable, "-m", "pip", "install", "tpu-info"],
    ),
    (
        "xprof",
        [sys.executable, "-m", "pip", "install", "xprof"],
    ),
    (
        "host_bench_deps",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "datasets>=2.20.0",
            "seaborn>=0.13.2",
            "matplotlib>=3.9.0",
            "scipy>=1.12.0",
            "httpx>=0.27.0",
            "aiohttp>=3.9.0",
            "pyyaml>=6.0.1",
            "pytest>=8.0.0",
        ],
    ),
]


def check_python_version() -> str:
    if sys.version_info < (3, 12):
        raise RuntimeError(
            f"Python 3.12+ is required by the Execution Contract; found {sys.version}"
        )
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_dist_version(dist_names: list[str], module_fallback: str | None = None) -> str | None:
    for dist in dist_names:
        try:
            return importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            continue
    if module_fallback:
        try:
            mod = importlib.import_module(module_fallback)
            return str(getattr(mod, "__version__", "installed-unknown-version"))
        except ImportError:
            return None
    return None


def audit_and_optionally_install_packages(auto_install: bool = True) -> dict[str, Any]:
    package_checks = {
        "torch": (["torch"], "torch"),
        "torch_tpu": (["torch-tpu", "torch_tpu"], "torch_tpu"),
        "vllm": (["vllm"], "vllm"),
        "vllm_torchtpu": (["vllm-torchtpu", "vllm_torchtpu", "tpu-inference"], "vllm_torchtpu"),
        "jax": (["jax"], "jax"),
        "pallas": (["jax"], "jax.experimental.pallas"),
        "tokamax": (["tokamax", "openxla-tokamax"], "tokamax"),
        "tpu_info": (["tpu-info"], None),
        "xprof": (["xprof", "tensorboard-plugin-profile"], None),
        "datasets": (["datasets"], "datasets"),
        "seaborn": (["seaborn"], "seaborn"),
    }

    versions: dict[str, str | None] = {
        key: get_dist_version(dists, mod) for key, (dists, mod) in package_checks.items()
    }

    if auto_install:
        if versions["vllm_torchtpu"] is None and versions["vllm"] is None:
            print("[phase0] Installing missing vllm-torchtpu from official index...")
            subprocess.run(OFFICIAL_INSTALL_RECIPES[0][1], check=True)
        if versions["tokamax"] is None:
            print("[phase0] Installing missing openxla/tokamax...")
            rc = subprocess.run(OFFICIAL_INSTALL_RECIPES[1][1], check=False).returncode
            if rc != 0:
                subprocess.run(OFFICIAL_INSTALL_RECIPES[2][1], check=True)
        if versions["tpu_info"] is None:
            subprocess.run(OFFICIAL_INSTALL_RECIPES[3][1], check=False)
        if versions["xprof"] is None:
            subprocess.run(OFFICIAL_INSTALL_RECIPES[4][1], check=False)
        if versions["datasets"] is None or versions["seaborn"] is None:
            subprocess.run(OFFICIAL_INSTALL_RECIPES[5][1], check=True)

        # Re-audit after installation
        versions = {
            key: get_dist_version(dists, mod) for key, (dists, mod) in package_checks.items()
        }

    missing_critical = [
        k for k in ("torch", "torch_tpu", "vllm", "jax", "pallas") if versions.get(k) is None
    ]
    if missing_critical:
        raise RuntimeError(
            f"Critical TorchTPU packages are missing after discovery/install: {missing_critical}. "
            f"Detected versions: {json.dumps(versions, indent=2)}"
        )
    return versions


def capture_tpu_info_cli() -> str:
    try:
        proc = subprocess.run(
            ["tpu-info"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return (proc.stdout + "\n" + proc.stderr).strip()
    except FileNotFoundError:
        return "tpu-info binary not found in PATH"
    except subprocess.TimeoutExpired:
        return "tpu-info timed out"


def verify_bf16_and_fp8_matmul_on_tpu() -> dict[str, Any]:
    """Execute real BF16 and FP8 matmuls on torch.device('tpu') + torch.compile(backend='tpu')."""
    import torch
    import torch_tpu  # noqa: F401

    device = torch.device("tpu")
    m, k, n = 1024, 2048, 2048
    flops = 2.0 * m * k * n

    torch.manual_seed(42)
    a_fp32 = torch.randn((m, k), dtype=torch.float32) * 0.1
    b_fp32 = torch.randn((k, n), dtype=torch.float32) * 0.1
    ref_cpu = a_fp32 @ b_fp32

    # 1. Eager BF16 Matmul on TPU
    a_bf16 = a_fp32.to(dtype=torch.bfloat16).to(device)
    b_bf16 = b_fp32.to(dtype=torch.bfloat16).to(device)
    out_bf16_eager = torch.matmul(a_bf16, b_bf16)
    _sync_tpu(torch)
    out_bf16_cpu = out_bf16_eager.to("cpu", dtype=torch.float32)
    if not torch.isfinite(out_bf16_cpu).all():
        raise RuntimeError("Non-finite output in eager BF16 TPU matmul!")
    bf16_eager_cos = float(
        torch.nn.functional.cosine_similarity(
            out_bf16_cpu.flatten(), ref_cpu.flatten(), dim=0
        ).item()
    )

    # 2. Compiled BF16 Matmul via torch.compile(backend="tpu")
    def bf16_mm_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)

    compiled_bf16_mm = torch.compile(bf16_mm_fn, backend="tpu")
    _ = compiled_bf16_mm(a_bf16, b_bf16)  # compile warmup
    _sync_tpu(torch)

    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        out_bf16_compiled = compiled_bf16_mm(a_bf16, b_bf16)
    _sync_tpu(torch)
    bf16_elapsed = (time.perf_counter() - t0) / iters
    bf16_tflops = (flops / bf16_elapsed) / 1e12
    bf16_compiled_cos = float(
        torch.nn.functional.cosine_similarity(
            out_bf16_compiled.to("cpu", dtype=torch.float32).flatten(),
            ref_cpu.flatten(),
            dim=0,
        ).item()
    )

    # 3. FP8 Matmul Verification (probe native _scaled_mm and float8_e4m3fn matmul on TPU)
    fp8_dtype = torch.float8_e4m3fn
    a_fp8 = a_fp32.to(dtype=fp8_dtype).to(device)
    # Column-major transpose layout required by torch._scaled_mm: (N, K).t() -> (K, N)
    b_col_major_fp8 = b_fp32.t().contiguous().to(dtype=fp8_dtype).to(device)
    scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)

    fp8_dispatch_op = "torch._scaled_mm"

    def fp8_mm_fn(
        x_fp8: torch.Tensor,
        y_col_fp8: torch.Tensor,
        sa: torch.Tensor,
        sb: torch.Tensor,
    ) -> torch.Tensor:
        res = torch._scaled_mm(
            x_fp8,
            y_col_fp8.t(),
            scale_a=sa,
            scale_b=sb,
            out_dtype=torch.bfloat16,
        )
        # PyTorch versions return either Tensor or (Tensor, amax)
        return res[0] if isinstance(res, tuple) else res

    try:
        out_fp8_eager = fp8_mm_fn(a_fp8, b_col_major_fp8, scale_a, scale_b)
        _sync_tpu(torch)
    except Exception as scaled_mm_exc:
        # Inspect if torch_tpu exposes direct torch.matmul on float8_e4m3fn
        fp8_dispatch_op = f"torch.matmul(float8_e4m3fn) [scaled_mm note: {scaled_mm_exc}]"
        b_row_fp8 = b_fp32.to(dtype=fp8_dtype).to(device)

        def fp8_mm_fn(  # type: ignore[misc]
            x_fp8: torch.Tensor,
            y_col_fp8: torch.Tensor,
            sa: torch.Tensor,
            sb: torch.Tensor,
        ) -> torch.Tensor:
            del y_col_fp8, sa, sb
            return torch.matmul(x_fp8, b_row_fp8).to(torch.bfloat16)

        out_fp8_eager = fp8_mm_fn(a_fp8, b_col_major_fp8, scale_a, scale_b)
        _sync_tpu(torch)

    out_fp8_eager_cpu = out_fp8_eager.to("cpu", dtype=torch.float32)
    if not torch.isfinite(out_fp8_eager_cpu).all():
        raise RuntimeError("Non-finite output in eager FP8 TPU matmul!")

    compiled_fp8_mm = torch.compile(fp8_mm_fn, backend="tpu")
    _ = compiled_fp8_mm(a_fp8, b_col_major_fp8, scale_a, scale_b)
    _sync_tpu(torch)

    t1 = time.perf_counter()
    for _ in range(iters):
        out_fp8_compiled = compiled_fp8_mm(a_fp8, b_col_major_fp8, scale_a, scale_b)
    _sync_tpu(torch)
    fp8_elapsed = (time.perf_counter() - t1) / iters
    fp8_tflops = (flops / fp8_elapsed) / 1e12
    fp8_compiled_cos = float(
        torch.nn.functional.cosine_similarity(
            out_fp8_compiled.to("cpu", dtype=torch.float32).flatten(),
            ref_cpu.flatten(),
            dim=0,
        ).item()
    )

    if bf16_compiled_cos < 0.995 or fp8_compiled_cos < 0.985:
        raise RuntimeError(
            f"TPU matmul cosine similarity below health threshold: "
            f"BF16={bf16_compiled_cos:.5f}, FP8={fp8_compiled_cos:.5f}"
        )

    return {
        "matrix_shape_mkn": [m, k, n],
        "bf16_eager_cosine_sim": round(bf16_eager_cos, 6),
        "bf16_compiled_cosine_sim": round(bf16_compiled_cos, 6),
        "bf16_compiled_latency_ms": round(bf16_elapsed * 1000.0, 4),
        "bf16_compiled_tflops": round(bf16_tflops, 2),
        "fp8_dispatch_op": fp8_dispatch_op,
        "fp8_compiled_cosine_sim": round(fp8_compiled_cos, 6),
        "fp8_compiled_latency_ms": round(fp8_elapsed * 1000.0, 4),
        "fp8_compiled_tflops": round(fp8_tflops, 2),
    }


def _sync_tpu(torch_mod: Any) -> None:
    if hasattr(torch_mod, "tpu") and hasattr(torch_mod.tpu, "synchronize"):
        torch_mod.tpu.synchronize()


def introspect_ecosystem_apis() -> dict[str, Any]:
    """Inspect live installed modules & source files for speculative decoding and jax_op."""
    import torch

    report: dict[str, Any] = {
        "pallas_jax_op": {},
        "tpu_profiler": {},
        "tokamax": {},
        "vllm_speculative": {},
        "vllm_cli_speculative_help": "",
    }

    # 1. Inspect torch.tpu.pallas.jax_op and mesh helpers
    pallas_mod = getattr(getattr(torch, "tpu", None), "pallas", None)
    if pallas_mod is not None:
        pallas_file = getattr(pallas_mod, "__file__", "<builtin>")
        attrs = [a for a in dir(pallas_mod) if not a.startswith("_")]
        sigs: dict[str, str] = {}
        for attr in attrs:
            obj = getattr(pallas_mod, attr)
            if callable(obj):
                try:
                    sigs[attr] = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    sigs[attr] = "(...)"
        report["pallas_jax_op"] = {
            "module_path": pallas_file,
            "public_attributes": attrs,
            "callable_signatures": sigs,
        }

    # 2. Inspect torch.tpu.profiler
    prof_mod = getattr(getattr(torch, "tpu", None), "profiler", None)
    if prof_mod is not None:
        prof_attrs = [a for a in dir(prof_mod) if not a.startswith("_")]
        prof_sigs: dict[str, str] = {}
        for attr in prof_attrs:
            obj = getattr(prof_mod, attr)
            if callable(obj):
                try:
                    prof_sigs[attr] = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    prof_sigs[attr] = "(...)"
        report["tpu_profiler"] = {
            "module_path": getattr(prof_mod, "__file__", "<builtin>"),
            "public_attributes": prof_attrs,
            "callable_signatures": prof_sigs,
        }

    # 3. Inspect openxla/tokamax
    try:
        import tokamax

        tok_root = Path(tokamax.__file__).resolve().parent
        tok_files = [
            str(p.relative_to(tok_root.parent))
            for p in sorted(tok_root.rglob("*.py"))[:80]
        ]
        report["tokamax"] = {
            "module_path": str(tok_root),
            "public_attributes": [a for a in dir(tokamax) if not a.startswith("_")],
            "sample_files": tok_files,
        }
    except ImportError:
        report["tokamax"] = {"installed": False}

    # 4. Scan vllm & vllm_torchtpu / tpu_inference source trees for DFlash, DSpark, EAGLE-3, ngram
    keywords = ("dflash", "dspark", "eagle", "ngram", "spec_decode", "speculative", "pallas")
    discovered_files: list[dict[str, Any]] = []
    for pkg_name in ("vllm_torchtpu", "tpu_inference", "vllm"):
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            continue
        if not getattr(pkg, "__file__", None):
            continue
        pkg_root = Path(pkg.__file__).resolve().parent
        for py_file in sorted(pkg_root.rglob("*.py")):
            rel = str(py_file.relative_to(pkg_root.parent))
            rel_lower = rel.lower()
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            text_lower = text.lower()
            matched = [
                kw
                for kw in ("dflash", "dspark", "eagle3", "eagle", "ngram", "jax_op")
                if kw in rel_lower or kw in text_lower
            ]
            if matched and (
                any(k in rel_lower for k in keywords)
                or "dflash" in matched
                or "dspark" in matched
                or "jax_op" in matched
            ):
                discovered_files.append(
                    {
                        "package": pkg_name,
                        "file": str(py_file),
                        "relative_path": rel,
                        "matched_keywords": matched,
                    }
                )

    # Also inspect SpeculativeConfig fields directly from vllm.config
    spec_config_fields: list[str] = []
    spec_methods_doc = ""
    try:
        from vllm.config import SpeculativeConfig

        if hasattr(SpeculativeConfig, "__dataclass_fields__"):
            spec_config_fields = list(SpeculativeConfig.__dataclass_fields__.keys())
        elif hasattr(SpeculativeConfig, "model_fields"):
            spec_config_fields = list(SpeculativeConfig.model_fields.keys())
        spec_methods_doc = inspect.getdoc(SpeculativeConfig) or ""
    except Exception as exc:
        spec_methods_doc = f"Could not import vllm.config.SpeculativeConfig: {exc}"

    report["vllm_speculative"] = {
        "speculative_config_fields": spec_config_fields,
        "speculative_config_doc": spec_methods_doc[:3000],
        "discovered_source_files": discovered_files[:120],
    }

    # 5. Capture `vllm serve --help` speculative section
    try:
        help_proc = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        help_lines = [
            line
            for line in (help_proc.stdout + "\n" + help_proc.stderr).splitlines()
            if any(
                tok in line.lower()
                for tok in ("speculative", "kv-cache-dtype", "async-scheduling", "quantization")
            )
        ]
        report["vllm_cli_speculative_help"] = "\n".join(help_lines[:80])
    except Exception as exc:
        report["vllm_cli_speculative_help"] = f"CLI help probe note: {exc}"

    return report


def write_api_discovery_markdown(probe_payload: dict[str, Any], out_path: Path) -> None:
    topo = probe_payload["topology"]
    pkgs = probe_payload["packages"]
    mm = probe_payload["matmul_verification"]
    apis = probe_payload["api_introspection"]

    pallas_info = apis.get("pallas_jax_op", {})
    prof_info = apis.get("tpu_profiler", {})
    tok_info = apis.get("tokamax", {})
    spec_info = apis.get("vllm_speculative", {})

    lines = [
        "# Phase 0: Hardware Topology & TorchTPU Ecosystem API Discovery",
        "",
        f"- **Generated Timestamp (UTC)**: `{probe_payload['timestamp_utc']}`",
        f"- **Host**: `{topo['hostname']}` (Python `{topo['python_version']}`)",
        f"- **TPU Device Kind**: `{topo['device_kind']}` (`chip_count={topo['chip_count']}`, `HBM3E/chip={topo['hbm_gb_per_chip']} GiB`, `total_HBM3E={topo['total_hbm_gb']} GiB`)",
        f"- **VFIO Nodes**: `{', '.join(topo['vfio_devices']) or 'managed via libtpu'}`",
        "",
        "## 1. Installed Package Versions",
        "",
        "| Component | Installed Version |",
        "| :--- | :--- |",
    ]
    for k, v in pkgs.items():
        lines.append(f"| `{k}` | `{v}` |")

    lines.extend(
        [
            "",
            "## 2. TPU v7 Device Matmul Verification (`BF16` & `FP8`)",
            "",
            f"- **Shape `(M, K, N)`**: `{tuple(mm['matrix_shape_mkn'])}`",
            f"- **BF16 `torch.compile(backend='tpu')`**: `{mm['bf16_compiled_tflops']} TFLOPS` (`{mm['bf16_compiled_latency_ms']} ms`, cosine sim vs FP32 = `{mm['bf16_compiled_cosine_sim']}`)",
            f"- **FP8 Dispatch Op**: `{mm['fp8_dispatch_op']}`",
            f"- **FP8 `torch.compile(backend='tpu')`**: `{mm['fp8_compiled_tflops']} TFLOPS` (`{mm['fp8_compiled_latency_ms']} ms`, cosine sim vs FP32 = `{mm['fp8_compiled_cosine_sim']}`)",
            "",
            "## 3. `torch.tpu.pallas.jax_op` & Custom Kernel Entry Points",
            "",
            f"- **Module Path**: `{pallas_info.get('module_path', 'N/A')}`",
            "- **Discovered Callable Signatures**:",
            "```python",
        ]
    )
    for fn_name, sig in sorted(pallas_info.get("callable_signatures", {}).items()):
        lines.append(f"torch.tpu.pallas.{fn_name}{sig}")
    lines.extend(
        [
            "```",
            "",
            "## 4. `torch.tpu.profiler` Entry Points",
            "",
            f"- **Module Path**: `{prof_info.get('module_path', 'N/A')}`",
            "```python",
        ]
    )
    for fn_name, sig in sorted(prof_info.get("callable_signatures", {}).items()):
        lines.append(f"torch.tpu.profiler.{fn_name}{sig}")
    lines.extend(
        [
            "```",
            "",
            "## 5. `openxla/tokamax` Discovery",
            "",
            f"- **Module Root**: `{tok_info.get('module_path', 'not installed')}`",
            f"- **Top-Level Exports**: `{', '.join(tok_info.get('public_attributes', []))}`",
            "",
            "## 6. `vllm` / `vllm-torchtpu` Speculative Decoding Entry Points",
            "",
            f"- **`SpeculativeConfig` Fields**: `{', '.join(spec_info.get('speculative_config_fields', []))}`",
            "",
            "### Discovered Source Files Matching `DFlash` / `DSpark` / `EAGLE-3` / `ngram` / `jax_op`",
            "",
            "| Package | Relative Path | Matched Capabilities |",
            "| :--- | :--- | :--- |",
        ]
    )
    for entry in spec_info.get("discovered_source_files", [])[:60]:
        lines.append(
            f"| `{entry['package']}` | `{entry['relative_path']}` | `{', '.join(entry['matched_keywords'])}` |"
        )

    lines.extend(
        [
            "",
            "### Relevant `vllm` CLI Flags (`--help` Excerpt)",
            "",
            "```text",
            apis.get("vllm_cli_speculative_help", ""),
            "```",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 TPU v7 Discovery & Health Gate")
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not attempt to pip install missing packages",
    )
    args = parser.parse_args()

    py_ver = check_python_version()
    cache_env = ensure_persistent_cache_env()
    killed = cleanup_stale_tpu_locks(allow_kill=True)
    print(f"[phase0] Python {py_ver} verified. Cleaned stale PIDs: {killed}")

    packages = audit_and_optionally_install_packages(auto_install=not args.no_install)
    tpu_info_raw = capture_tpu_info_cli()
    topology = require_tpu_device()
    print(f"[phase0] Detected TPU Topology: {json.dumps(topology.to_dict(), indent=2)}")

    matmul_health = verify_bf16_and_fp8_matmul_on_tpu()
    print(f"[phase0] TPU Matmul Health: {json.dumps(matmul_health, indent=2)}")

    api_introspection = introspect_ecosystem_apis()

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": py_ver,
        "persistent_cache_env": cache_env,
        "topology": topology.to_dict(),
        "tpu_info_cli": tpu_info_raw,
        "packages": packages,
        "matmul_verification": matmul_health,
        "api_introspection": api_introspection,
    }

    probe_json = REPO_ROOT / "artifacts" / "logs" / "phase0_probe.json"
    probe_json.parent.mkdir(parents=True, exist_ok=True)
    probe_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    api_md = REPO_ROOT / "docs" / "api_discovery.md"
    write_api_discovery_markdown(payload, api_md)

    update_phase_status(
        "phase0_discovery",
        status="passed",
        overall_state="PHASE_0_PASSED",
        extra={
            "chip_count": topology.chip_count,
            "device_kind": topology.device_kind,
            "hbm_gb_per_chip": topology.hbm_gb_per_chip,
        },
    )
    print(f"[phase0] Gate PASSED. Wrote {probe_json} and {api_md}.")


if __name__ == "__main__":
    main()
