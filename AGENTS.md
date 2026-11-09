# AGENTS.md — Autonomous Takeover Instructions for Cloud TPU v7 (Ironwood / v7x)

> **ATTENTION CODING AGENT (`agy`, Claude Code, Codex, Gemini CLI, OpenCode, Cursor):**
> Read this file, `HANDOFF.md`, and `STATUS.json` **before running any commands**.
> You are taking over an already-architected, host-verified codebase on a **Cloud TPU v7 (Ironwood / `tpu7x`) VM** to execute **Phases 0 through 4** on real TPU v7 hardware and push the completed empirical artifacts and reports back to GitHub.

---

## 1. Mission & Current State

- **Project**: *"FP8 Block-Diffusion Speculative Decoding on Systolic Arrays"* (TorchTPU Sprint + Research Repository).
- **What Was Already Completed (Stage A — Pre-TPU Scaffold)**:
  - The complete repository (`src/`, `kernels/`, `benchmarks/`, `tests/`, `scripts/`, `configs/`) was implemented and verified for syntax, tiled online Shannon entropy & prefix-verification math, roofline optimization equations, and binary `.xplane.pb` (`XSpace`) wire-format parsing (`pytest -v tests/test_host_math_and_xprof_parser.py`).
  - Because Stage A was built on a non-TPU laptop (`rabimba-aipc`), **zero TPU benchmarks, traces, or numbers were mocked or fabricated**. `STATUS.json` records `overall_state = "AWAITING_TPU_V7_EXECUTION"`.
- **Your Mission Now (Stage B — On Cloud TPU v7 VM)**:
  - Execute **Phase 0 $\rightarrow$ Phase 1 $\rightarrow$ Phase 2 $\rightarrow$ Phase 3 $\rightarrow$ Phase 4** sequentially on this Cloud TPU v7 machine.
  - Inspect live installed `torch_tpu`, `vllm-torchtpu`, and `openxla/tokamax` APIs in Phase 0 (`docs/api_discovery.md`) and adapt any binding nuances in `kernels/pallas_block_verify.py`, `src/serving/vllm_launcher.py`, or `src/spec_decode/vllm_pallas_plugin.py` if live signatures differ.
  - Do **not** stop until `bash scripts/run_all.sh` exits with code `0`, `STATUS.json` shows `"ALL_PHASES_COMPLETE_AND_VERIFIED"`, and all real logs, `.xplane.pb` traces, plots, `README.md`, and `docs/paper_results_summary.md` are committed and pushed.

---

## 2. Non-Negotiable Execution Contract

1. **ZERO MOCKS OR PLACEHOLDERS**: Never fabricate benchmark numbers, never substitute CPU execution for TPU execution, never leave `TODO` stubs, and never catch-and-ignore exceptions to fake a passing test. Every metric in the final report must be parsed programmatically from real JSON/CSV logs and real XProf (`.xplane.pb`) traces generated on this TPU v7 machine during your session.
2. **INSPECT BEFORE CODING**: Run Phase 0 first to inspect the actual installed Python packages, module paths, and CLI `--help` outputs on this VM (`docs/api_discovery.md` + `artifacts/logs/phase0_probe.json`) so you never guess API signatures.
3. **TPU LOCK HYGIENE**: Before starting any new server or benchmark process, verify that no stale Python, Ray, or vLLM worker processes hold `/dev/vfio/*` or `libtpu` locks (`source scripts/env_common.sh && cleanup_stale_tpu_processes`). Always terminate child processes cleanly in `try/finally` blocks (`ManagedProcessGroup` / `ManagedVLLMServer`).
4. **PERSISTENT COMPILATION CACHE**: Always source `scripts/env_common.sh` (or call `ensure_persistent_cache_env()`) so `VLLM_XLA_CACHE_PATH`, `XLA_CACHE_DIR`, and `JAX_COMPILATION_CACHE_DIR` point to persistent local disk and TPU graph compilations are reused across runs.
5. **ITERATIVE SMOKE-TO-SCALE**: Validate every pipeline stage end-to-end with a fast smoke test (`benchmarks/run_spec_matrix.py --smoke`, ~2-minute turnaround) before launching full-scale sweeps (`--full`). Fix every error until all `pytest` checks and shell scripts exit with code `0`.

---

## 3. Repository Map

| Path | Responsibility |
| :--- | :--- |
| `STATUS.json` | Machine-readable gate status ledger (`phase0` .. `phase4`). |
| `HANDOFF.md` | Detailed step-by-step operational runbook & TPU v7 debugging guide. |
| `scripts/env_common.sh` | Exports `VLLM_XLA_CACHE_PATH` / XLA caches + `cleanup_stale_tpu_processes`. |
| `src/tpu_utils.py` | TPU v7 topology detector, `/proc/*/fd` VFIO lock scanner, `ManagedProcessGroup`, XLA compile counter. |
| `scripts/phase0_discover.py` | **Phase 0 Gate**: Hardware/package probe, BF16 & FP8 matmul on `torch.device("tpu")` + `torch.compile(backend="tpu")`, live API discovery -> `docs/api_discovery.md`. |
| `configs/models.yaml` | Target (`Qwen/Qwen3-8B-FP8`, `Qwen/Qwen3-8B`) + DFlash (`z-lab/Qwen3-8B-DFlash-b16`) + EAGLE-3 (`RedHatAI/Qwen3-8B-speculator.eagle3`) registry. |
| `scripts/download_and_verify_models.py` | Verifies & caches HF checkpoints -> `artifacts/logs/model_manifest.json`. |
| `src/serving/vllm_launcher.py` | `vllm-torchtpu` server manager (`--kv-cache-dtype fp8`, `--async-scheduling`) & async streaming OpenAI client. |
| `tests/test_baseline_serving.py` | **Phase 1 Gate**: Endpoint health, 3-run greedy determinism, 16-prompt TTFT/TPOT/tok/s smoke test. |
| `kernels/pallas_block_verify.py` | **Core Contribution**: Static-shape `128`-tile Pallas online Shannon entropy + prefix block verification kernel wrapped via `torch.tpu.pallas.jax_op`. |
| `src/spec_decode/entropy_adaptive_controller.py` | Stateful static-tile adaptive $K^*(B, H)$ block horizon controller. |
| `src/spec_decode/vllm_pallas_plugin.py` | `vllm.general_plugins` hook integrating Pallas adaptive verification into Mode `(d)`. |
| `tests/test_pallas_block_verify.py` | **Phase 2 Gate**: Pallas `jax_op` vs TPU eager reference across `B=[1,4,16,32]`, `K_pad=[16,32,64]` + **0 XLA recompiles** across 20 dynamic steps. |
| `src/datasets/prompt_suites.py` | Real HumanEval, GSM8K, and UltraChat/ShareGPT loaders & functional/exact-match graders. |
| `benchmarks/run_spec_matrix.py` | **Phase 3 Gate**: `--smoke` and `--full` 4-mode $\times$ 3-suite $\times$ `B=[1,4,8,16,32]` matrix + `.xplane.pb` trace capture + lossless quality parity. |
| `scripts/parse_xprof.py` | Programmatic `.xplane.pb` parser (`xprof` + pure-Python `XSpace` wire-format decoder) -> `xprof_summary.json` & `spec_matrix_summary.csv`. |
| `benchmarks/fit_roofline_model.py` | **Phase 4**: Fits TPU v7 roofline (4,614 TFLOPS FP8, 7.38 TB/s HBM3E) & renders 4 plots in `artifacts/plots/`. |
| `scripts/generate_final_reports.py` | Populates `README.md` and `docs/paper_results_summary.md` directly from `artifacts/logs/*.json`. |
| `scripts/validate_artifacts.py` | Strict provenance, binary `.xplane.pb`, PNG plot, and zero-mock validator. |
| `scripts/run_all.sh` | End-to-end automated pipeline (must exit `0`). |

---

## 4. Sequential Execution Gates (Do Not Skip or Reorder)

### Step 0: Editable Install & Phase 0 Gate
```bash
pip install -e . --no-build-isolation --no-deps
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 scripts/phase0_discover.py"
```
- **Action Required After Phase 0**: Read `docs/api_discovery.md` and `artifacts/logs/phase0_probe.json`!
  - Verify `torch.tpu.pallas.jax_op` signature matches `_bind_torch_tpu_jax_op` in `kernels/pallas_block_verify.py`.
  - Verify `vllm` `--speculative-config` fields for `dflash` and `eagle3` match `_build_speculative_config` in `benchmarks/run_spec_matrix.py`, and verify `RejectionSampler` module path in `src/spec_decode/vllm_pallas_plugin.py`. Patch any live API differences immediately before Phase 1.

### Step 1: Phase 1 — Baseline FP8 Serving & Determinism Gate
```bash
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 scripts/download_and_verify_models.py --include-bf16-contrast"
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && pytest -v -s tests/test_baseline_serving.py"
```

### Step 2: Phase 2 — Custom Pallas `jax_op` Verification & Zero-Recompile Gate
```bash
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && pytest -v -s tests/test_pallas_block_verify.py"
```

### Step 3: Phase 3 — Speculative Decoding Matrix & XProf Profiling Gate
```bash
# Always pass the 2-minute smoke gate FIRST:
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 benchmarks/run_spec_matrix.py --smoke"

# Then run the full sweep (supports --resume if interrupted) + XProf parser:
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 benchmarks/run_spec_matrix.py --full --resume"
bash -c "source scripts/env_common.sh && python3 scripts/parse_xprof.py"
```

### Step 4: Phase 4 — Roofline Fit, Plots, Final Reports & End-to-End Validation
```bash
bash -c "source scripts/env_common.sh && python3 benchmarks/fit_roofline_model.py && python3 scripts/generate_final_reports.py"
bash scripts/run_all.sh
```
- Once `bash scripts/run_all.sh` exits with code `0`, stage, commit, and push all generated `artifacts/logs/*`, `artifacts/plots/*`, `artifacts/traces/*` (compressing `.xplane.pb` > 50 MB to `.xplane.pb.gz` if needed for GitHub's 100 MB file limit—`parse_xprof.py` and `validate_artifacts.py` natively support `.xplane.pb.gz`!), `docs/api_discovery.md`, `docs/paper_results_summary.md`, `README.md`, and `STATUS.json`.
