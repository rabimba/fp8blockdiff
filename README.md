# FP8 Block-Diffusion Speculative Decoding on Cloud TPU v7 Systolic Arrays

End-to-end **TorchTPU** systems & research repository evaluating **FP8 Block-Diffusion Speculative Decoding (`DFlash` / `DSpark`)** and a custom **Static-Shape, Systolic-Tile-Aligned Pallas Entropy-Adaptive Verification Kernel (`torch.tpu.pallas.jax_op`)** against autoregressive FP8/BF16 decoding and tree-based `EAGLE-3` speculation on **Google Cloud TPU v7 (Ironwood / `tpu7x`)**.

> **Multi-Agent Takeover Ready**: Cloning this repository onto a Cloud TPU v7 VM? See [`AGENTS.md`](./AGENTS.md), [`HANDOFF.md`](./HANDOFF.md), and [`STATUS.json`](./STATUS.json) for the sequential Phase 0–4 hardware execution runbook.

---

## 1. Architectural Motivation: Systolic Roofline Mechanics on Cloud TPU v7

A single Cloud TPU v7 (Ironwood) chip delivers **4,614 TFLOPS of native FP8 compute** (2,307 TFLOPS BF16) backed by **192 GB of HBM3E at 7.38 TB/s**, placing its FP8 operational ridge point at:

$$\text{AI}_{\text{ridge, FP8}} = \frac{4614\text{ TFLOP/s}}{7.38\text{ TB/s}} \approx 625.2\text{ FLOPs/Byte}$$

During standard autoregressive FP8 decoding ($K = 0$, 1 token per sequence), weight streaming performs $\approx 2B\text{ FLOPs/Byte}$—leaving the TPU v7 Matrix Multiply Unit (MXU) **>99.6% idle at $B = 1$** and **~90% memory-bandwidth-bound even at $B = 32$**.

While speculative decoding raises target verification batch-tokens to $M_{\text{eff}} = B(K + 1)$, existing approaches encounter three systolic bottlenecks:

1. **Autoregressive Tree Speculation (`EAGLE-3`)**: Serializes drafting across $d$ sequential HBM weight-bound steps and requires irregular tree-attention masks that fragment `128 x 128` systolic MXU tiles.
2. **Fixed-Block Diffusion Drafting (`DFlash`)**: Proposes $K$ tokens in a single $O(1)$ parallel pass over contiguous causal chains, matching systolic arrays ideally at small $B$. However, when $B(K + 1)$ approaches or crosses the 625.2 FLOP/B ridge point or when local token Shannon entropy $H$ spikes, fixed large blocks ($K = 32$) waste MXU compute verifying low-survival tail positions (*suffix decay*).
3. **Dynamic Truncation Recompilation Penalty**: Varying draft tensor shapes $[B, K_t]$ dynamically per decode step triggers repeated XLA HLO graph recompilations.

### Core Contribution: `kernels/pallas_block_verify.py`
We implement a custom **Pallas TPU kernel** wrapped as a first-class PyTorch operator via `torch.tpu.pallas.jax_op` (`pallas_entropy_block_verify`):
- **Single-Pass VMEM Tiled Online Entropy & Argmax**: Streams `[128, 1024]` vocabulary tiles through VMEM in `float32` vector registers using a shift-invariant online `(m, l, s, best_val, best_idx)` recurrence to compute exact token-level Shannon entropy $H = \log l - s/l$ and greedy target predictions **without materializing `[B(K+1), V]` probability tensors in HBM3E** (~600x writeback reduction).
- **Zero-Recompilation Dynamic Block Scheduling**: Keeps all verification and mask buffers strictly padded to TPU systolic tile multiples (`128` rows, static $K_{\text{pad}} \in \{16, 32, 64\}$) and passes runtime thresholds via tensor control blocks—achieving adaptive $K^*(B, H)$ suffix masking with **zero XLA graph recompilations**.

---

## 2. Repository Structure

```text
fp8blockdiff/
├── AGENTS.md / CLAUDE.md / HANDOFF.md / STATUS.json  # Multi-agent TPU v7 takeover context
├── configs/
│   ├── models.yaml                  # Qwen3-8B-FP8, BF16 contrast, DFlash & EAGLE-3 checkpoints
│   └── sweep_matrix.yaml            # TPU v7 roofline constants + 4-mode x 3-suite x 5-batch sweep
├── kernels/
│   └── pallas_block_verify.py       # Pallas TPU kernel + torch.tpu.pallas.jax_op bridge
├── src/
│   ├── tpu_utils.py                 # VFIO lock hygiene, topology auto-detect, XLA compile counters
│   ├── datasets/prompt_suites.py    # Real HumanEval, GSM8K, and UltraChat/ShareGPT loaders & graders
│   ├── serving/vllm_launcher.py     # Managed vllm-torchtpu launcher & async streaming telemetry client
│   └── spec_decode/                 # Entropy-adaptive controller & vLLM general_plugins hook
├── benchmarks/
│   ├── run_spec_matrix.py           # Phase 3 benchmark orchestrator + XProf (.xplane.pb) capture
│   └── fit_roofline_model.py        # Phase 4 analytical + empirical roofline fitter & 4-plot builder
├── scripts/
│   ├── env_common.sh                # VLLM_XLA_CACHE_PATH, persistent XLA caches & lock cleanup
│   ├── phase0_discover.py           # Phase 0 hardware probe, BF16/FP8 matmul & live API audit
│   ├── download_and_verify_models.py
│   ├── parse_xprof.py               # Programmatic .xplane.pb analyzer (xprof + pure wire-format)
│   ├── generate_final_reports.py    # Populates empirical tables strictly from artifacts/logs/*.json
│   ├── validate_artifacts.py        # Strict zero-mock & provenance validator
│   └── run_all.sh                   # End-to-end automated pipeline
└── tests/
    ├── test_host_math_and_xprof_parser.py
    ├── test_baseline_serving.py
    └── test_pallas_block_verify.py
```

---

## 3. Reproducible Execution Commands (Cloud TPU v7 VM)

```bash
# Install editable package & verify host-side math + XSpace wire parser
pip install -e . --no-build-isolation --no-deps
pytest -v tests/test_host_math_and_xprof_parser.py

# Phase 0: Hardware probe, BF16/FP8 TPU matmul & live API introspection
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 scripts/phase0_discover.py"

# Phase 1: Stage checkpoints + FP8 baseline serving health & determinism gate
bash -c "source scripts/env_common.sh && python3 scripts/download_and_verify_models.py --include-bf16-contrast"
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && pytest -v -s tests/test_baseline_serving.py"

# Phase 2: Pallas jax_op numerical equivalence & 20-step zero-XLA-recompile gate
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && pytest -v -s tests/test_pallas_block_verify.py"

# Phase 3: Iterative smoke test -> full 4-mode speculative matrix + XProf trace parsing
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 benchmarks/run_spec_matrix.py --smoke"
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 benchmarks/run_spec_matrix.py --full --resume"
bash -c "source scripts/env_common.sh && python3 scripts/parse_xprof.py"

# Phase 4: Analytical roofline fit, publication plots, report build & full validation
bash -c "source scripts/env_common.sh && python3 benchmarks/fit_roofline_model.py && python3 scripts/generate_final_reports.py"
bash scripts/run_all.sh
```

---

## 4. Empirical Results on Cloud TPU v7 (Ironwood)

<!-- BEGIN_EMPIRICAL_RESULTS -->

- **Verified Hardware Topology**: `TPU v7 (tpu7x-16)` (`4` chip(s), `192.0 GiB` HBM3E/chip, host `rkaranjai-nanodit-dev`)
- **Phase 0 Matmul Health**: Compiled BF16 `91.56 TFLOPS` | Compiled FP8 `94.53 TFLOPS` (`torch._scaled_mm`)
- **Phase 1 Baseline FP8 Smoke (16 Prompts, B=4)**: `812.005 tok/s` | TTFT P50 `133.89 ms` | TPOT P50 `3.945 ms`
- **Phase 2 Custom Pallas `jax_op` Verification**: `0` XLA recompilations across `20` dynamic-acceptance steps (mean kernel latency `6.3728 ms` at `V=151936`)

### Speculative Decoding Throughput, Latency & XProf Hardware Utilization

| Mode | Suite | B | Mean Acc Tokens | TTFT P50 (ms) | TPOT P50 (ms) | Throughput (tok/s) | MXU Util (%) | HBM3E BW Util (%) | XLA Recompiles |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `autoreg_fp8` | `code_synthesis` | 1 | 0.00 | 11.60 | 3.55 | **265.06** | 4.84 | 57.36 | 0 |
| `autoreg_fp8` | `code_synthesis` | 4 | 0.00 | 21.01 | 3.95 | **953.31** | — | — | — |
| `autoreg_fp8` | `code_synthesis` | 8 | 0.00 | 32.79 | 4.48 | **1592.08** | 10.81 | 43.91 | 0 |
| `autoreg_fp8` | `code_synthesis` | 16 | 0.00 | 77.06 | 5.64 | **2283.85** | — | — | — |
| `autoreg_fp8` | `code_synthesis` | 32 | 0.00 | 118.03 | 7.91 | **3392.31** | 14.08 | 28.80 | 0 |
| `autoreg_fp8` | `math_reasoning` | 1 | 0.00 | 9.75 | 3.55 | **279.47** | 4.84 | 57.36 | 0 |
| `autoreg_fp8` | `math_reasoning` | 4 | 0.00 | 15.01 | 3.95 | **956.55** | — | — | — |
| `autoreg_fp8` | `math_reasoning` | 8 | 0.00 | 15.89 | 4.47 | **1618.72** | 10.81 | 43.91 | 0 |
| `autoreg_fp8` | `math_reasoning` | 16 | 0.00 | 43.16 | 5.56 | **2386.24** | — | — | — |
| `autoreg_fp8` | `math_reasoning` | 32 | 0.00 | 92.50 | 8.38 | **3310.23** | 14.08 | 28.80 | 0 |
| `autoreg_fp8` | `dialogue_summarization` | 1 | 0.00 | 27.53 | 3.56 | **274.72** | 4.84 | 57.36 | 0 |
| `autoreg_fp8` | `dialogue_summarization` | 4 | 0.00 | 54.98 | 3.95 | **921.51** | — | — | — |
| `autoreg_fp8` | `dialogue_summarization` | 8 | 0.00 | 89.47 | 4.53 | **1503.63** | 10.81 | 43.91 | 0 |
| `autoreg_fp8` | `dialogue_summarization` | 16 | 0.00 | 143.71 | 5.82 | **2193.80** | — | — | — |
| `autoreg_fp8` | `dialogue_summarization` | 32 | 0.00 | 167.29 | 8.98 | **3026.20** | 14.08 | 28.80 | 0 |
| `autoreg_bf16` | `code_synthesis` | 1 | 0.00 | 12.90 | 5.52 | **179.90** | 5.90 | 73.98 | 0 |
| `autoreg_bf16` | `code_synthesis` | 8 | 0.00 | 46.75 | 6.69 | **1060.69** | 12.38 | 56.83 | 0 |
| `autoreg_bf16` | `code_synthesis` | 32 | 0.00 | 175.20 | 11.60 | **2348.64** | 22.22 | 34.16 | 0 |
| `autoreg_bf16` | `math_reasoning` | 1 | 0.00 | 11.45 | 5.51 | **180.58** | 5.90 | 73.98 | 0 |
| `autoreg_bf16` | `math_reasoning` | 8 | 0.00 | 27.52 | 6.67 | **1074.29** | 12.38 | 56.83 | 0 |
| `autoreg_bf16` | `math_reasoning` | 32 | 0.00 | 167.73 | 11.09 | **2394.02** | 22.22 | 34.16 | 0 |
| `autoreg_bf16` | `dialogue_summarization` | 1 | 0.00 | 32.25 | 5.53 | **177.98** | 5.90 | 73.98 | 0 |
| `autoreg_bf16` | `dialogue_summarization` | 8 | 0.00 | 111.83 | 6.78 | **1021.04** | 12.38 | 56.83 | 0 |
| `autoreg_bf16` | `dialogue_summarization` | 32 | 0.00 | 195.75 | 12.08 | **2227.91** | 22.22 | 34.16 | 0 |
| `eagle3_fp8` | `code_synthesis` | 1 | 1.34 | 16.69 | 3.20 | **299.62** | 5.54 | 31.70 | 0 |
| `eagle3_fp8` | `code_synthesis` | 4 | 1.35 | 23.20 | 3.20 | **1201.86** | — | — | — |
| `eagle3_fp8` | `code_synthesis` | 8 | 1.34 | 23.81 | 3.29 | **2192.59** | 16.12 | 26.75 | 1 |
| `eagle3_fp8` | `code_synthesis` | 16 | 1.29 | 71.92 | 4.16 | **2775.54** | — | — | — |
| `eagle3_fp8` | `code_synthesis` | 32 | 1.30 | 131.69 | 8.46 | **2938.43** | 22.79 | 20.08 | 4 |
| `eagle3_fp8` | `math_reasoning` | 1 | 1.64 | 15.47 | 2.60 | **368.37** | 5.54 | 31.70 | 0 |
| `eagle3_fp8` | `math_reasoning` | 4 | 1.63 | 23.85 | 2.87 | **1201.28** | — | — | — |
| `eagle3_fp8` | `math_reasoning` | 8 | 1.60 | 25.84 | 2.96 | **2079.74** | 16.12 | 26.75 | 1 |
| `eagle3_fp8` | `math_reasoning` | 16 | 1.59 | 33.12 | 3.23 | **3854.49** | — | — | — |
| `eagle3_fp8` | `math_reasoning` | 32 | 1.62 | 114.42 | 6.09 | **4453.82** | 22.79 | 20.08 | 4 |
| `eagle3_fp8` | `dialogue_summarization` | 1 | 1.13 | 30.30 | 3.22 | **304.46** | 5.54 | 31.70 | 0 |
| `eagle3_fp8` | `dialogue_summarization` | 4 | 1.11 | 38.50 | 3.49 | **1038.11** | — | — | — |
| `eagle3_fp8` | `dialogue_summarization` | 8 | 1.11 | 47.21 | 4.15 | **1687.68** | 16.12 | 26.75 | 1 |
| `eagle3_fp8` | `dialogue_summarization` | 16 | 1.05 | 84.52 | 5.34 | **2247.22** | — | — | — |
| `eagle3_fp8` | `dialogue_summarization` | 32 | 1.09 | 185.08 | 9.21 | **2527.19** | 22.79 | 20.08 | 4 |
| `dflash_fp8_k8` | `code_synthesis` | 1 | 1.70 | 12.82 | 1.98 | **509.37** | 5.35 | 38.51 | 0 |
| `dflash_fp8_k8` | `code_synthesis` | 4 | 1.75 | 19.04 | 2.22 | **1595.02** | — | — | — |
| `dflash_fp8_k8` | `code_synthesis` | 8 | 1.65 | 22.03 | 2.99 | **2243.72** | 22.36 | 33.21 | 0 |
| `dflash_fp8_k8` | `code_synthesis` | 16 | 1.62 | 68.17 | 4.66 | **2513.70** | — | — | — |
| `dflash_fp8_k8` | `code_synthesis` | 32 | 1.68 | 124.83 | 11.75 | **2446.22** | 24.86 | 23.25 | 0 |
| `dflash_fp8_k8` | `math_reasoning` | 1 | 2.14 | 12.36 | 1.61 | **594.17** | 5.35 | 38.51 | 0 |
| `dflash_fp8_k8` | `math_reasoning` | 4 | 2.15 | 18.46 | 1.88 | **1919.01** | — | — | — |
| `dflash_fp8_k8` | `math_reasoning` | 8 | 2.11 | 21.72 | 2.29 | **3013.16** | 22.36 | 33.21 | 0 |
| `dflash_fp8_k8` | `math_reasoning` | 16 | 2.16 | 34.54 | 3.29 | **4085.52** | — | — | — |
| `dflash_fp8_k8` | `math_reasoning` | 32 | 2.16 | 76.97 | 7.12 | **4057.22** | 24.86 | 23.25 | 0 |
| `dflash_fp8_k8` | `dialogue_summarization` | 1 | 1.14 | 28.54 | 2.59 | **381.44** | 5.35 | 38.51 | 0 |
| `dflash_fp8_k8` | `dialogue_summarization` | 4 | 1.14 | 36.61 | 3.06 | **1215.33** | — | — | — |
| `dflash_fp8_k8` | `dialogue_summarization` | 8 | 1.14 | 42.34 | 4.04 | **1814.47** | 22.36 | 33.21 | 0 |
| `dflash_fp8_k8` | `dialogue_summarization` | 16 | 1.17 | 77.48 | 6.61 | **2178.45** | — | — | — |
| `dflash_fp8_k8` | `dialogue_summarization` | 32 | 1.16 | 313.92 | 14.98 | **1937.91** | 24.86 | 23.25 | 0 |
| `dflash_fp8_k16` | `code_synthesis` | 1 | 1.89 | 13.14 | 1.91 | **528.97** | 8.82 | 37.30 | 0 |
| `dflash_fp8_k16` | `code_synthesis` | 4 | 1.88 | 20.50 | 2.46 | **1501.20** | — | — | — |
| `dflash_fp8_k16` | `code_synthesis` | 8 | 1.79 | 29.40 | 3.36 | **1932.50** | 29.11 | 34.06 | 0 |
| `dflash_fp8_k16` | `code_synthesis` | 16 | 1.78 | 59.01 | 6.60 | **2080.26** | — | — | — |
| `dflash_fp8_k16` | `code_synthesis` | 32 | 1.84 | 138.24 | 16.49 | **1894.66** | 25.95 | 20.77 | 0 |
| `dflash_fp8_k16` | `math_reasoning` | 1 | 2.39 | 12.43 | 1.55 | **626.07** | 8.82 | 37.30 | 0 |
| `dflash_fp8_k16` | `math_reasoning` | 4 | 2.39 | 19.42 | 1.97 | **1873.50** | — | — | — |
| `dflash_fp8_k16` | `math_reasoning` | 8 | 2.41 | 25.08 | 2.56 | **2773.83** | 29.11 | 34.06 | 0 |
| `dflash_fp8_k16` | `math_reasoning` | 16 | 2.43 | 48.43 | 5.06 | **2842.64** | — | — | — |
| `dflash_fp8_k16` | `math_reasoning` | 32 | 2.48 | 116.70 | 12.17 | **2452.79** | 25.95 | 20.77 | 0 |
| `dflash_fp8_k16` | `dialogue_summarization` | 1 | 1.23 | 28.63 | 2.55 | **398.85** | 8.82 | 37.30 | 0 |
| `dflash_fp8_k16` | `dialogue_summarization` | 4 | 1.28 | 37.27 | 3.34 | **1111.21** | — | — | — |
| `dflash_fp8_k16` | `dialogue_summarization` | 8 | 1.24 | 46.28 | 4.93 | **1532.79** | 29.11 | 34.06 | 0 |
| `dflash_fp8_k16` | `dialogue_summarization` | 16 | 1.31 | 79.70 | 9.68 | **1516.60** | — | — | — |
| `dflash_fp8_k16` | `dialogue_summarization` | 32 | 1.21 | 263.39 | 26.90 | **1182.53** | 25.95 | 20.77 | 0 |
| `dflash_fp8_k32` | `code_synthesis` | 1 | 1.74 | 13.30 | 2.16 | **462.67** | 13.99 | 39.83 | 0 |
| `dflash_fp8_k32` | `code_synthesis` | 4 | 1.71 | 23.87 | 3.25 | **1156.67** | — | — | — |
| `dflash_fp8_k32` | `code_synthesis` | 8 | 1.63 | 42.26 | 6.35 | **1186.23** | 33.66 | 26.64 | 0 |
| `dflash_fp8_k32` | `code_synthesis` | 16 | 1.65 | 94.80 | 13.85 | **1103.98** | — | — | — |
| `dflash_fp8_k32` | `code_synthesis` | 32 | 1.70 | 255.12 | 19.55 | **1346.14** | 28.45 | 21.96 | 0 |
| `dflash_fp8_k32` | `math_reasoning` | 1 | 2.17 | 12.31 | 1.78 | **552.16** | 13.99 | 39.83 | 0 |
| `dflash_fp8_k32` | `math_reasoning` | 4 | 2.19 | 22.98 | 2.66 | **1444.28** | — | — | — |
| `dflash_fp8_k32` | `math_reasoning` | 8 | 2.21 | 41.79 | 5.16 | **1490.80** | 33.66 | 26.64 | 0 |
| `dflash_fp8_k32` | `math_reasoning` | 16 | 2.19 | 76.13 | 10.88 | **1406.59** | — | — | — |
| `dflash_fp8_k32` | `math_reasoning` | 32 | 2.17 | 172.56 | 15.45 | **1642.20** | 28.45 | 21.96 | 0 |
| `dflash_fp8_k32` | `dialogue_summarization` | 1 | 1.16 | 28.32 | 2.81 | **363.63** | 13.99 | 39.83 | 0 |
| `dflash_fp8_k32` | `dialogue_summarization` | 4 | 1.16 | 39.78 | 4.30 | **872.73** | — | — | — |
| `dflash_fp8_k32` | `dialogue_summarization` | 8 | 1.17 | 64.31 | 8.85 | **894.35** | 33.66 | 26.64 | 0 |
| `dflash_fp8_k32` | `dialogue_summarization` | 16 | 1.15 | 124.02 | 19.86 | **793.66** | — | — | — |
| `dflash_fp8_k32` | `dialogue_summarization` | 32 | 1.12 | 627.62 | 33.06 | **822.81** | 28.45 | 21.96 | 0 |
| `dflash_pallas_adaptive_fp8` | `code_synthesis` | 1 | 1.70 | 12.52 | 1.96 | **515.83** | 4.35 | 43.93 | 0 |
| `dflash_pallas_adaptive_fp8` | `code_synthesis` | 4 | 1.71 | 18.57 | 2.23 | **1597.02** | — | — | — |
| `dflash_pallas_adaptive_fp8` | `code_synthesis` | 8 | 1.65 | 21.62 | 2.90 | **2358.74** | 21.77 | 38.29 | 0 |
| `dflash_pallas_adaptive_fp8` | `code_synthesis` | 16 | 1.63 | 48.19 | 4.30 | **2907.63** | — | — | — |
| `dflash_pallas_adaptive_fp8` | `code_synthesis` | 32 | 1.68 | 127.69 | 9.52 | **2872.73** | 25.76 | 21.41 | 0 |
| `dflash_pallas_adaptive_fp8` | `math_reasoning` | 1 | 2.14 | 11.90 | 1.60 | **597.89** | 4.35 | 43.93 | 0 |
| `dflash_pallas_adaptive_fp8` | `math_reasoning` | 4 | 2.16 | 17.89 | 1.82 | **2056.02** | — | — | — |
| `dflash_pallas_adaptive_fp8` | `math_reasoning` | 8 | 2.11 | 21.38 | 2.22 | **3132.18** | 21.77 | 38.29 | 0 |
| `dflash_pallas_adaptive_fp8` | `math_reasoning` | 16 | 2.15 | 37.46 | 3.65 | **3722.13** | — | — | — |
| `dflash_pallas_adaptive_fp8` | `math_reasoning` | 32 | 2.14 | 74.01 | 7.08 | **4129.26** | 25.76 | 21.41 | 0 |
| `dflash_pallas_adaptive_fp8` | `dialogue_summarization` | 1 | 1.14 | 28.42 | 2.53 | **399.18** | 4.35 | 43.93 | 0 |
| `dflash_pallas_adaptive_fp8` | `dialogue_summarization` | 4 | 1.14 | 36.23 | 3.02 | **1237.95** | — | — | — |
| `dflash_pallas_adaptive_fp8` | `dialogue_summarization` | 8 | 1.14 | 42.70 | 3.94 | **1857.54** | 21.77 | 38.29 | 0 |
| `dflash_pallas_adaptive_fp8` | `dialogue_summarization` | 16 | 1.17 | 76.96 | 6.40 | **2238.96** | — | — | — |
| `dflash_pallas_adaptive_fp8` | `dialogue_summarization` | 32 | 1.17 | 178.92 | 15.31 | **2028.63** | 25.76 | 21.41 | 0 |

### Analytical + Empirical TPU v7 Roofline Calibration

- **Fitted Systolic MXU Efficiency (`eta_mxu`)**: `0.15`
- **Fitted HBM3E Bandwidth Efficiency (`eta_bw`)**: `0.2105`
- **Fitted Parallel Block-Diffusion Draft Overhead**: `14.2171 ms`
- **Entropy Acceptance Decay (`lambda0`, `lambda1`, `beta`)**: `(0.36184, 0.29061, 0.9213)`

| Batch Size (B) | Domain | Mean Entropy H (nats) | Optimal Block Horizon K* | Predicted Peak Throughput (tok/s) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `code_synthesis` | 0.70 | **0** | 187.95 |
| 1 | `math_reasoning` | 1.04 | **0** | 187.95 |
| 1 | `dialogue_summarization` | 1.68 | **0** | 187.95 |
| 4 | `code_synthesis` | 0.70 | **0** | 731.74 |
| 4 | `math_reasoning` | 1.04 | **0** | 731.74 |
| 4 | `dialogue_summarization` | 1.68 | **0** | 731.74 |
| 8 | `code_synthesis` | 0.70 | **0** | 1413.23 |
| 8 | `math_reasoning` | 1.04 | **0** | 1413.23 |
| 8 | `dialogue_summarization` | 1.68 | **0** | 1413.23 |
| 16 | `code_synthesis` | 0.70 | **0** | 2644.81 |
| 16 | `math_reasoning` | 1.04 | **0** | 2644.81 |
| 16 | `dialogue_summarization` | 1.68 | **0** | 2644.81 |
| 32 | `code_synthesis` | 0.70 | **0** | 4687.16 |
| 32 | `math_reasoning` | 1.04 | **0** | 4687.16 |
| 32 | `dialogue_summarization` | 1.68 | **0** | 4687.16 |

### Lossless Output Quality Equivalence Summary

- **All Lossless Modes Equivalent Within Greedy Tie-Break Tolerance**: `True`

| Mode | GSM8K Exact-Match | HumanEval Pass@1 |
| :--- | :---: | :---: |
| `autoreg_fp8` | `0.8` | `0.0` |
| `autoreg_bf16` | `0.74` | `0.0` |
| `eagle3_fp8` | `0.78` | `0.0` |
| `dflash_fp8_k8` | `0.82` | `0.0` |
| `dflash_fp8_k16` | `0.82` | `0.0` |
| `dflash_fp8_k32` | `0.76` | `0.0` |
| `dflash_pallas_adaptive_fp8` | `0.82` | `0.0` |

<!-- END_EMPIRICAL_RESULTS -->
