# Cloud TPU v7 (Ironwood / `tpu7x`) End-to-End Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement and execute this plan task-by-task on the live Cloud TPU v7 (`tpu7x`) pod `rkaranjai-nanodit-dev`.

**Goal:** Execute all experiments across Phases 0–4 of *FP8 Block-Diffusion Speculative Decoding on Systolic Arrays* on real Cloud TPU v7 (`tpu7x`) hardware (`rkaranjai-nanodit-dev`), resolve all live `torch_tpu 0.2.0` and `vllm 0.31.0` (`vllm-torchtpu-core 0.0.0`) API deltas, verify `bash scripts/run_all.sh` exits `0`, and sync/commit/push all empirical artifacts and reports back to GitHub.

**Architecture:**
- Local control plane: `/Users/rkaranjai/Documents/GitHub/fp8blockdiff` connected via SSH bastion tunnel (`127.0.0.1:8443 -> 10.183.51.108:443`) to GKE cluster `dev51-sgau-gke-cluster` (`unset https_proxy http_proxy && kubectl ...`).
- Remote execution plane: Pod `rkaranjai-nanodit-dev` on node `gke-tpu-83df725d-h97k` (`tpu7x`, 4 physical TPU7x chips / 8 TensorCores, 192 GiB HBM3E/chip, 466 GiB `tmpfs` at `/tmp/tpu_logs`).
- Five sequential gates (`Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4`) governed by `STATUS.json` and validated by `scripts/validate_artifacts.py`.

**Tech Stack:** Python 3.12.15, `torch 2.13.0+cpu`, `torch_tpu 0.2.0`, `vllm 0.31.0+empty`, `vllm-torchtpu-core 0.0.0`, `jax 0.11.2`, `libtpu 0.0.50`, `tpu-info 0.7.1`, `xprof`, `scipy`, `matplotlib`, `seaborn`, `datasets`, `pytest`.

---

### Task 1: Update `AGENTS.md` & Adapt Live `torch_tpu 0.2.0` / `vllm 0.31.0` Bindings

**Files:**
- Modify: `AGENTS.md`
- Modify: `scripts/env_common.sh`
- Modify: `src/tpu_utils.py`
- Modify: `scripts/phase0_discover.py`
- Modify: `kernels/pallas_block_verify.py`
- Modify: `src/serving/vllm_launcher.py`
- Modify: `src/spec_decode/vllm_pallas_plugin.py`
- Modify: `benchmarks/run_spec_matrix.py`

- [ ] **Step 1: Update `AGENTS.md` with verified repository & live `torch_tpu 0.2.0` / `vllm 0.31.0` invariants**
- [ ] **Step 2: Patch `scripts/env_common.sh` and `src/tpu_utils.py` for `torch.accelerator` APIs, dual-core TPU7x HBM sizing (`~192 GiB/chip`), `TpuBackend._num_compiled_executables`, and `/tmp/tpu_logs` cache paths (`HF_HOME`, `VLLM_CACHE_ROOT`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)**
- [ ] **Step 3: Patch `scripts/phase0_discover.py` to use `torch.accelerator.synchronize()` and single-chip isolation when probing `torch.device("tpu:0")`**
- [ ] **Step 4: Patch `kernels/pallas_block_verify.py` so `jax` is imported at module scope (`jax_op` calls `inspect.signature(..., eval_str=True)`), `jax_op(op_name, _jax_entry, ...)` passes positional `name, fn`, and `get_pallas_mesh` is only called when `torch.distributed.is_initialized()` and `world_size > 1`**
- [ ] **Step 5: Patch `src/serving/vllm_launcher.py`, `src/spec_decode/vllm_pallas_plugin.py`, and `benchmarks/run_spec_matrix.py` for `vllm 0.31.0` `--profiler-config '{"profiler":"torch","torch_profiler_dir":"..."}'`, `TpuProfilerConfig(run_dir=...)`, and `vllm_torchtpu.layers.adapter.sample.rejection_sampler.RejectionSampler` hook**
- [ ] **Step 6: Run local host unit tests (`pytest -v tests/test_host_math_and_xprof_parser.py`) to confirm zero regressions**

---

### Task 2: Sync Workspace to `rkaranjai-nanodit-dev` & Execute Phase 0 Gate

**Files:**
- Generate on pod & sync back: `docs/api_discovery.md`, `artifacts/logs/phase0_probe.json`, `STATUS.json`

- [ ] **Step 1: Sync repository to `rkaranjai-nanodit-dev:/workspace/fp8blockdiff` and install missing Python test/plot/dataset packages (`pytest`, `datasets`, `matplotlib`, `seaborn`, `xprof`, `pyyaml`, `httpx`, `aiohttp`, `huggingface-hub`) + `pip install -e . --no-build-isolation --no-deps`**
- [ ] **Step 2: Run Phase 0 discovery gate (`source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 scripts/phase0_discover.py --no-install`)**
- [ ] **Step 3: Inspect generated `docs/api_discovery.md` and `artifacts/logs/phase0_probe.json` to verify `chip_count >= 1`, FP8 (`float8_e4m3fn`) eager & compiled matmuls on TPU v7, and `jax_op` / `vllm` discovery**

---

### Task 3: Execute Phase 1 — Model Verification & Baseline FP8 Serving Determinism Gate

**Files:**
- Generate on pod & sync back: `artifacts/logs/model_manifest.json`, `artifacts/logs/phase1_baseline_smoke.json`, `STATUS.json`

- [ ] **Step 1: Run `python3 scripts/download_and_verify_models.py --include-bf16-contrast` using `HF_HOME=/tmp/tpu_logs/hf_cache` and verify `Qwen/Qwen3-8B-FP8`, `Qwen/Qwen3-8B`, `z-lab/Qwen3-8B-DFlash-b16`, and `RedHatAI/Qwen3-8B-speculator.eagle3`**
- [ ] **Step 2: Run Phase 1 pytest gate (`pytest -v -s tests/test_baseline_serving.py`) verifying `/health`, 3-run greedy determinism (`temperature=0.0`), and 16-prompt baseline TTFT/TPOT/tok/s telemetry**

---

### Task 4: Execute Phase 2 — Custom Static-Tile Pallas `jax_op` & Zero-Recompile Gate

**Files:**
- Generate on pod & sync back: `artifacts/logs/phase2_pallas_verify.json`, `STATUS.json`

- [ ] **Step 1: Run Phase 2 pytest gate (`pytest -v -s tests/test_pallas_block_verify.py`) testing numerical equivalence across `B in [1, 4, 16, 32]`, `K_pad in [16, 32, 64]`, and 20 dynamic steps with `new_tpu_xla_compilations == 0`**

---

### Task 5: Execute Phase 3 — Speculative Decoding Matrix, Quality Parity & XProf Trace Capture

**Files:**
- Generate on pod & sync back: `artifacts/logs/spec_matrix_runs.json`, `artifacts/logs/quality_equivalence.json`, `artifacts/logs/xprof_summary.json`, `artifacts/logs/spec_matrix_summary.csv`, `artifacts/traces/**/*.xplane.pb[.gz]`, `STATUS.json`

- [ ] **Step 1: Run the fast 2-minute smoke gate (`python3 benchmarks/run_spec_matrix.py --smoke`) to validate all modes (`autoreg_fp8`, `dflash_fp8_k16`, `dflash_pallas_adaptive_fp8`) end-to-end**
- [ ] **Step 2: Run the full matrix sweep (`python3 benchmarks/run_spec_matrix.py --full --resume`) across modes (`autoreg_fp8`, `autoreg_bf16`, `eagle3_fp8`, `dflash_fp8_k8`, `dflash_fp8_k16`, `dflash_fp8_k32`, `dflash_pallas_adaptive_fp8`), 3 real datasets (HumanEval, GSM8K, UltraChat), and batch sizes `B in [1, 4, 8, 16, 32]` with real `.xplane.pb` trace capture at `B in {1, 8, 32}`**
- [ ] **Step 3: Run `python3 scripts/parse_xprof.py` to decode all `.xplane.pb[.gz]` traces into `artifacts/logs/xprof_summary.json` and `artifacts/logs/spec_matrix_summary.csv`**

---

### Task 6: Execute Phase 4 — Roofline Fit, Plots, Reports, `run_all.sh` Gate & Git Push

**Files:**
- Generate on pod & sync back: `artifacts/logs/roofline_fit.json`, `artifacts/plots/*.{png,pdf}`, `README.md`, `docs/paper_results_summary.md`, `STATUS.json`

- [ ] **Step 1: Run `python3 benchmarks/fit_roofline_model.py` and `python3 scripts/generate_final_reports.py`**
- [ ] **Step 2: Run `bash scripts/run_all.sh` on `rkaranjai-nanodit-dev` and verify exit code `0` and `STATUS.json` == `"ALL_PHASES_COMPLETE_AND_VERIFIED"`**
- [ ] **Step 3: Sync all generated `artifacts/`, `docs/`, `README.md`, and `STATUS.json` back to local repo, compress any `.xplane.pb` > 50 MB to `.xplane.pb.gz`, run `python3 scripts/validate_artifacts.py` and `pytest -v tests/test_host_math_and_xprof_parser.py` locally, and commit/push to GitHub**
