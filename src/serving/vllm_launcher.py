"""vllm-torchtpu server launcher and OpenAI-compatible streaming client.

Enforces Non-Negotiable Execution Contract Rules #2, #3, and #4:
- Dynamically inspects installed `vllm.entrypoints.openai.api_server --help`
  before building CLI flags so unknown/changed flags are never guessed.
- Launches inside ManagedProcessGroup with guaranteed try/finally cleanup and
  /dev/vfio/* lock verification.
- Exports VLLM_XLA_CACHE_PATH and XLA persistent compilation caches.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import aiohttp
import httpx

from src.tpu_utils import (
    ManagedProcessGroup,
    REPO_ROOT,
    TPUTopology,
    ensure_persistent_cache_env,
    require_tpu_device,
)


@dataclasses.dataclass
class VLLMServerConfig:
    """Configuration for a managed vllm-torchtpu OpenAI server instance."""

    model: str
    mode_id: str = "baseline_fp8"
    tensor_parallel_size: int | None = None
    kv_cache_dtype: str = "fp8"
    dtype: str = "auto"
    async_scheduling: bool = True
    max_model_len: int = 4096
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.88
    disable_prefix_caching: bool = True
    speculative_config: dict[str, Any] | None = None
    profiler_dir: Path | None = None
    extra_env: dict[str, str] | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    startup_timeout_sec: float = 900.0
    poll_interval_sec: float = 3.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclasses.dataclass
class RequestTelemetry:
    """Per-request streaming latency and token generation telemetry."""

    prompt_id: str
    suite: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tpot_ms: float
    inter_token_latencies_ms: list[float]


_CACHED_CLI_HELP: str | None = None


def get_vllm_cli_help_text() -> str:
    """Inspect actual installed `vllm.entrypoints.openai.api_server --help` output (Contract #2)."""
    global _CACHED_CLI_HELP
    if _CACHED_CLI_HELP is not None:
        return _CACHED_CLI_HELP

    proc = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    _CACHED_CLI_HELP = proc.stdout + "\n" + proc.stderr
    return _CACHED_CLI_HELP


def build_vllm_server_cmd(cfg: VLLMServerConfig, topology: TPUTopology) -> list[str]:
    """Construct verified CLI arguments against live installed vllm-torchtpu flags."""
    help_text = get_vllm_cli_help_text()
    env_tp = int(os.environ.get("FP8_BLOCKDIFF_TP_SIZE", "0"))
    tp_size = cfg.tensor_parallel_size or (env_tp if env_tp > 0 else topology.chip_count)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        str(cfg.max_model_len),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
    ]

    if "--dtype" in help_text and cfg.dtype:
        cmd.extend(["--dtype", cfg.dtype])

    if "--kv-cache-dtype" in help_text and cfg.kv_cache_dtype:
        cmd.extend(["--kv-cache-dtype", cfg.kv_cache_dtype])

    if "--gpu-memory-utilization" in help_text:
        cmd.extend(["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)])

    if cfg.async_scheduling:
        if "--async-scheduling" in help_text:
            cmd.append("--async-scheduling")

    if cfg.disable_prefix_caching:
        if "--no-enable-prefix-caching" in help_text:
            cmd.append("--no-enable-prefix-caching")

    if cfg.speculative_config is not None:
        if "--speculative-config" not in help_text:
            raise RuntimeError(
                "Installed vLLM entrypoint does not advertise --speculative-config in --help."
            )
        cmd.extend(["--speculative-config", json.dumps(cfg.speculative_config)])

    if cfg.profiler_dir is not None and "--profiler-config" in help_text:
        prof_cfg = {
            "profiler": "torch",
            "torch_profiler_dir": str(cfg.profiler_dir.resolve()),
        }
        cmd.extend(["--profiler-config", json.dumps(prof_cfg)])

    return cmd


class ManagedVLLMServer:
    """Context manager standing up vllm-torchtpu, waiting for health readiness,
    and guaranteeing clean worker & VFIO lock teardown on exit.
    """

    def __init__(self, cfg: VLLMServerConfig) -> None:
        self.cfg = cfg
        self.topology: TPUTopology | None = None
        self._group: ManagedProcessGroup | None = None
        self.log_path = REPO_ROOT / "artifacts" / "logs" / f"vllm_server_{cfg.mode_id}.log"

    def __enter__(self) -> "ManagedVLLMServer":
        self.topology = require_tpu_device()
        base_env = dict(os.environ)
        base_env.update(ensure_persistent_cache_env())
        if self.cfg.profiler_dir is not None:
            self.cfg.profiler_dir.mkdir(parents=True, exist_ok=True)
            base_env["VLLM_TORCH_PROFILER_DIR"] = str(self.cfg.profiler_dir)
        if self.cfg.extra_env:
            base_env.update(self.cfg.extra_env)

        cmd = build_vllm_server_cmd(self.cfg, self.topology)
        candidate_chips = ["1", "2", "3", "0"]
        last_exc: Exception | None = None
        for attempt in range(6):
            env = dict(base_env)
            if env.get("FP8_BLOCKDIFF_TP_SIZE", "1") == "1" and self.topology.chip_count >= 2:
                chip = candidate_chips[attempt % len(candidate_chips)]
                pport = str(8475 + int(chip))
                mport1 = str(8439 + 2 * int(chip))
                mport2 = str(8440 + 2 * int(chip))
                env["TPU_VISIBLE_CHIPS"] = chip
                env["TPU_VISIBLE_DEVICES"] = chip
                env["TPU_PROCESS_PORT"] = pport
                env["TPU_PROCESS_ADDRESSES"] = f"localhost:{pport}"
                env["TPU_RUNTIME_METRICS_PORTS"] = f"{mport1},{mport2}"
            self._group = ManagedProcessGroup(cmd=cmd, env=env, log_path=self.log_path)
            proc = self._group.__enter__()
            try:
                self._wait_until_healthy(proc)
                return self
            except Exception as exc:
                self._group.__exit__(*sys.exc_info())
                self._group = None
                last_exc = exc
                if "Failed to acquire a TPU device node" in str(exc) and attempt < 5:
                    time.sleep(10.0)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._group is not None:
            self._group.__exit__(exc_type, exc_val, exc_tb)
            self._group = None

    def _wait_until_healthy(self, proc: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + self.cfg.startup_timeout_sec
        health_url = f"{self.cfg.base_url}/health"
        models_url = f"{self.cfg.base_url}/v1/models"

        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                rc = proc.poll()
                if rc is not None:
                    tail = self._read_log_tail(60)
                    raise RuntimeError(
                        f"vllm-torchtpu server ({self.cfg.mode_id}) exited prematurely "
                        f"with code {rc}. Last log lines:\n{tail}"
                    )
                try:
                    resp = client.get(health_url)
                    if resp.status_code == 200:
                        m_resp = client.get(models_url)
                        if m_resp.status_code == 200:
                            return
                except httpx.HTTPError:
                    pass
                time.sleep(self.cfg.poll_interval_sec)

        tail = self._read_log_tail(60)
        raise TimeoutError(
            f"Timed out after {self.cfg.startup_timeout_sec}s waiting for "
            f"{health_url}. Last log lines:\n{tail}"
        )

    def _read_log_tail(self, num_lines: int = 50) -> str:
        if not self.log_path.exists():
            return "<no server log file>"
        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-num_lines:])

    def start_profile(self) -> bool:
        """Trigger vLLM server-side Torch/TPU profiler if VLLM_TORCH_PROFILER_DIR is active."""
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.cfg.base_url}/start_profile")
            return resp.status_code == 200

    def stop_profile(self) -> bool:
        """Stop vLLM server-side Torch/TPU profiler and flush .xplane.pb traces."""
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{self.cfg.base_url}/stop_profile")
            return resp.status_code == 200

    def scrape_prometheus_spec_metrics(self) -> dict[str, float]:
        """Parse speculative decoding acceptance & draft counters from vLLM /metrics endpoint."""
        metrics: dict[str, float] = {}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.cfg.base_url}/metrics")
                if resp.status_code != 200:
                    return metrics
                text = resp.text
        except httpx.HTTPError:
            return metrics

        metric_Regex = re.compile(
            r"^(vllm:[a-zA-Z0-9_]+)(?:\{[^}]*\})?\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$"
        )
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = metric_Regex.match(line)
            if m:
                name, val_str = m.group(1), m.group(2)
                if any(k in name for k in ("spec_decode", "accepted", "draft", "kv_cache")):
                    metrics[name] = metrics.get(name, 0.0) + float(val_str)
        return metrics


async def stream_single_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_id: str,
    suite: str,
    prompt_text: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> RequestTelemetry:
    """Execute a streaming /v1/completions request and measure exact TTFT and TPOT."""
    payload = {
        "model": model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url}/v1/completions"

    chunk_times: list[float] = []
    pieces: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    t_start = time.perf_counter()
    first_token_time: float | None = None

    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status} from {url}: {body[:500]}")

        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            event = json.loads(data_str)
            now = time.perf_counter()

            choices = event.get("choices") or []
            if choices:
                token_text = choices[0].get("text") or ""
                if token_text:
                    if first_token_time is None:
                        first_token_time = now
                    chunk_times.append(now)
                    pieces.append(token_text)

            usage = event.get("usage")
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens)

    t_end = time.perf_counter()
    full_text = "".join(pieces)
    if completion_tokens <= 0:
        completion_tokens = max(len(pieces), 1)

    total_latency_ms = (t_end - t_start) * 1000.0
    ttft_ms = (
        (first_token_time - t_start) * 1000.0
        if first_token_time is not None
        else total_latency_ms
    )

    itl_ms: list[float] = []
    for i in range(1, len(chunk_times)):
        itl_ms.append((chunk_times[i] - chunk_times[i - 1]) * 1000.0)

    if completion_tokens > 1 and first_token_time is not None:
        tpot_ms = ((t_end - first_token_time) * 1000.0) / float(completion_tokens - 1)
    else:
        tpot_ms = 0.0

    return RequestTelemetry(
        prompt_id=prompt_id,
        suite=suite,
        generated_text=full_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        tpot_ms=tpot_ms,
        inter_token_latencies_ms=itl_ms,
    )


async def run_concurrent_prompt_batch(
    base_url: str,
    model: str,
    prompts: list[Any],
    concurrency: int,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[list[RequestTelemetry], float]:
    """Run a batch of EvalPrompt items with bounded active concurrency B."""
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def _worker(p: Any) -> RequestTelemetry:
            async with sem:
                return await stream_single_completion(
                    session=session,
                    base_url=base_url,
                    model=model,
                    prompt_id=p.prompt_id,
                    suite=p.suite,
                    prompt_text=p.prompt_text,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        t0 = time.perf_counter()
        results = await asyncio.gather(*[_worker(p) for p in prompts])
        wall_sec = time.perf_counter() - t0

    return list(results), wall_sec
