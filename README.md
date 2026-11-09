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

> **Status (`PENDING_TPU_V7_EXECUTION`)**: Per Non-Negotiable Execution Contract #1 (Zero Mocks / Zero Fabricated Metrics), this section intentionally contains no placeholder numbers prior to Cloud TPU v7 execution. Running `python3 scripts/generate_final_reports.py` in Phase 4 populates this block programmatically from `artifacts/logs/*.json` and `artifacts/traces/*.xplane.pb`.

<!-- END_EMPIRICAL_RESULTS -->
