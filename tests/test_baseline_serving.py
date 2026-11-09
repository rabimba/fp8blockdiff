"""PHASE 1 — BASELINE FP8 SERVING & DETERMINISTIC CORRECTNESS GATE.

Stands up vllm-torchtpu in FP8 (--kv-cache-dtype fp8, --async-scheduling,
VLLM_XLA_CACHE_PATH set) and verifies:
1. OpenAI-compatible /health and /v1/completions endpoint health.
2. Exact greedy decoding determinism (temperature=0.0) across repeated runs.
3. Baseline TTFT, TPOT, and output tokens/sec on a 16-prompt real smoke set,
   saved to artifacts/logs/phase1_baseline_smoke.json.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np
import pytest
import yaml

from src.datasets.prompt_suites import load_phase1_smoke_16_prompts
from src.serving.vllm_launcher import (
    ManagedVLLMServer,
    VLLMServerConfig,
    run_concurrent_prompt_batch,
)
from src.tpu_utils import REPO_ROOT, require_tpu_device, update_phase_status


pytestmark = pytest.mark.tpu_required


def _load_baseline_server_config() -> VLLMServerConfig:
    cfg_path = REPO_ROOT / "configs" / "models.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    prof = raw["profiles"][raw["active_profile"]]
    srv = raw["serving"]

    manifest_path = REPO_ROOT / "artifacts" / "logs" / "model_manifest.json"
    model_id = prof["target_fp8"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_id = manifest["target_fp8"]["repo_id"]

    return VLLMServerConfig(
        model=model_id,
        mode_id="phase1_baseline_fp8",
        kv_cache_dtype=srv["kv_cache_dtype"],
        async_scheduling=bool(srv["async_scheduling"]),
        max_model_len=int(srv["max_model_len"]),
        max_num_seqs=int(srv["max_num_seqs"]),
        gpu_memory_utilization=float(srv["gpu_memory_utilization"]),
        disable_prefix_caching=bool(srv["disable_prefix_caching_in_benchmarks"]),
        host=str(srv["host"]),
        port=int(srv["port"]),
        startup_timeout_sec=float(srv["startup_timeout_sec"]),
    )


@pytest.fixture(scope="module")
def baseline_fp8_server() -> Iterator[ManagedVLLMServer]:
    require_tpu_device()
    cfg = _load_baseline_server_config()
    with ManagedVLLMServer(cfg) as server:
        yield server


def test_openai_endpoint_health(baseline_fp8_server: ManagedVLLMServer) -> None:
    base_url = baseline_fp8_server.cfg.base_url
    with httpx.Client(timeout=15.0) as client:
        h_resp = client.get(f"{base_url}/health")
        assert h_resp.status_code == 200

        m_resp = client.get(f"{base_url}/v1/models")
        assert m_resp.status_code == 200
        models_data = m_resp.json()
        model_ids = [item["id"] for item in models_data.get("data", [])]
        assert baseline_fp8_server.cfg.model in model_ids


def test_greedy_decoding_determinism(baseline_fp8_server: ManagedVLLMServer) -> None:
    prompts = load_phase1_smoke_16_prompts(seed=42)[:4]
    base_url = baseline_fp8_server.cfg.base_url
    model = baseline_fp8_server.cfg.model

    runs: list[list[str]] = []
    for _ in range(3):
        telemetry, _ = asyncio.run(
            run_concurrent_prompt_batch(
                base_url=base_url,
                model=model,
                prompts=prompts,
                concurrency=1,
                max_tokens=96,
                temperature=0.0,
            )
        )
        texts = [item.generated_text for item in telemetry]
        assert all(len(t.strip()) > 0 for t in texts), "Empty completion returned!"
        runs.append(texts)

    assert runs[0] == runs[1], "Greedy run 0 and run 1 diverged!"
    assert runs[1] == runs[2], "Greedy run 1 and run 2 diverged!"


def test_record_16_prompt_baseline_smoke(baseline_fp8_server: ManagedVLLMServer) -> None:
    smoke_prompts = load_phase1_smoke_16_prompts(seed=42)
    assert len(smoke_prompts) == 16

    base_url = baseline_fp8_server.cfg.base_url
    model = baseline_fp8_server.cfg.model

    # Warmup 2 prompts so XLA steady-state execution is measured
    _ = asyncio.run(
        run_concurrent_prompt_batch(
            base_url=base_url,
            model=model,
            prompts=smoke_prompts[:2],
            concurrency=2,
            max_tokens=64,
            temperature=0.0,
        )
    )

    telemetry, wall_sec = asyncio.run(
        run_concurrent_prompt_batch(
            base_url=base_url,
            model=model,
            prompts=smoke_prompts,
            concurrency=4,
            max_tokens=128,
            temperature=0.0,
        )
    )

    ttfts = np.array([t.ttft_ms for t in telemetry], dtype=np.float64)
    tpots = np.array([t.tpot_ms for t in telemetry if t.tpot_ms > 0], dtype=np.float64)
    total_out_tokens = int(sum(t.completion_tokens for t in telemetry))
    output_tok_per_sec = float(total_out_tokens / max(wall_sec, 1e-6))

    assert np.all(np.isfinite(ttfts)) and float(ttfts.mean()) > 0.0
    assert total_out_tokens >= 16

    assert baseline_fp8_server.topology is not None
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "topology": baseline_fp8_server.topology.to_dict(),
        "model": model,
        "kv_cache_dtype": baseline_fp8_server.cfg.kv_cache_dtype,
        "async_scheduling": baseline_fp8_server.cfg.async_scheduling,
        "num_prompts": len(smoke_prompts),
        "concurrency": 4,
        "max_tokens": 128,
        "wall_time_sec": round(wall_sec, 4),
        "total_output_tokens": total_out_tokens,
        "output_tokens_per_sec": round(output_tok_per_sec, 3),
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 3),
        "ttft_p95_ms": round(float(np.percentile(ttfts, 95)), 3),
        "ttft_mean_ms": round(float(np.mean(ttfts)), 3),
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 3) if len(tpots) else 0.0,
        "tpot_p95_ms": round(float(np.percentile(tpots, 95)), 3) if len(tpots) else 0.0,
        "tpot_mean_ms": round(float(np.mean(tpots)), 3) if len(tpots) else 0.0,
    }

    out_path = REPO_ROOT / "artifacts" / "logs" / "phase1_baseline_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    update_phase_status(
        "phase1_baseline_serving",
        status="passed",
        overall_state="PHASE_1_PASSED",
        extra={
            "output_tokens_per_sec": summary["output_tokens_per_sec"],
            "ttft_p50_ms": summary["ttft_p50_ms"],
            "tpot_p50_ms": summary["tpot_p50_ms"],
        },
    )
