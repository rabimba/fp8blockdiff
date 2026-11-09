# HANDOFF.md — Complete Operational Runbook for Taking Over on Cloud TPU v7

This repository was engineered in two clean stages so that **zero metrics or traces were mocked off-device**:

- **Stage A (Completed)**: Full implementation of the repository layout, TPU lock/cache hygiene utilities, Phase 0 hardware & API introspector, Phase 1 FP8 `vllm-torchtpu` serving harness & determinism tests, Phase 2 static-tile Pallas online-entropy & block-verification kernel (`torch.tpu.pallas.jax_op`), Phase 3 4-mode $\times$ 3-dataset $\times$ 5-batch matrix runner & pure-Python/XProf `.xplane.pb` parser, and Phase 4 analytical TPU v7 roofline fitter, plot generator, report builder, and strict validator. Verified via `pytest -v tests/test_host_math_and_xprof_parser.py`.
- **Stage B (Ready for Takeover on Cloud TPU v7 VM)**: Clone this repository on your **Cloud TPU v7 (Ironwood / `v7x`) VM** and let your coding agent execute Phases 0–4 end-to-end on real TPU v7 silicon.

---

## Quick Copy-Paste Prompt for Your Coding Agent on the TPU v7 VM

After running `git clone https://github.com/rabimba/fp8blockdiff.git && cd fp8blockdiff` on your Cloud TPU v7 VM, paste this single prompt into Antigravity (`agy`), Claude Code, Codex, Gemini CLI, or OpenCode:

```text
You are an autonomous Staff ML Systems Engineer working directly on this Cloud TPU v7 (Ironwood / v7x) VM. Read AGENTS.md, HANDOFF.md, and STATUS.json first. Uphold every rule of the Non-Negotiable Execution Contract in AGENTS.md (zero mocks, inspect live installed packages in Phase 0 and adapt any signature deltas before Phase 1, strict /dev/vfio/* lock hygiene, persistent VLLM_XLA_CACHE_PATH, smoke-before-full sweeps). Execute Phase 0, Phase 1, Phase 2, Phase 3, and Phase 4 sequentially until `bash scripts/run_all.sh` exits with code 0, then git commit and git push all verified logs, compressed traces, plots, docs/api_discovery.md, docs/paper_results_summary.md, README.md, and STATUS.json.
```

---

## Detailed Gate-by-Gate Playbook & Adaptation Notes

### Gate 0 — Hardware Probe,Matmul Health & Live API Introspection
1. Install the local package in editable mode (registers the `vllm.general_plugins` hook for Mode `(d)` without altering TorchTPU wheels):
   ```bash
   pip install -e . --no-build-isolation --no-deps
   ```
2. Run Phase 0:
   ```bash
   bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 scripts/phase0_discover.py"
   ```
3. **Inspect `docs/api_discovery.md` immediately after Gate 0 passes**:
   - **`torch.tpu.pallas.jax_op`**: Check Section 3 of `docs/api_discovery.md`. `kernels/pallas_block_verify.py` (`_bind_torch_tpu_jax_op`) already introspects `inspect.signature(torch.tpu.pallas.jax_op)` for `op`/`fn`, `mesh`, `input_partition_specs`/`in_specs`, and `output_partition_specs`/`out_specs`. If your TorchTPU build uses a custom decorator style or requires `trace_with_jax`, adjust `_bind_torch_tpu_jax_op` directly to match `docs/api_discovery.md`.
   - **`vllm-torchtpu` Speculative Config**: Check Section 6 of `docs/api_discovery.md` (`SpeculativeConfig` fields and discovered `dflash` / `dspark` / `eagle3` files). Confirm whether `method` strings are `"dflash"`, `"eagle3"`, `"ngram"` (or version-specific aliases) and verify `_build_speculative_config` in `benchmarks/run_spec_matrix.py`.
   - **`RejectionSampler` Hook**: Check the exact module path of `RejectionSampler` in `docs/api_discovery.md` and confirm it is listed in `candidate_targets` inside `src/spec_decode/vllm_pallas_plugin.py`.

### Gate 1 — Checkpoint Staging & Baseline FP8 Serving Determinism
1. Stage and verify Hugging Face checkpoints (`Qwen/Qwen3-8B-FP8`, `Qwen/Qwen3-8B`, `z-lab/Qwen3-8B-DFlash-b16`, `RedHatAI/Qwen3-8B-speculator.eagle3`):
   ```bash
   bash -c "source scripts/env_common.sh && python3 scripts/download_and_verify_models.py --include-bf16-contrast"
   ```
   *(Tip: If you have multiple TPU v7 chips and want to switch to the 30B MoE profile after validating 8B, pass `--profile qwen3_30b_a3b` or set `active_profile: qwen3_30b_a3b` in `configs/models.yaml`.)*
2. Run the Phase 1 pytest suite:
   ```bash
   bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && pytest -v -s tests/test_baseline_serving.py"
   ```
   - Verifies `/health`, `/v1/models`, 3-pass greedy determinism (`temperature=0.0`), and writes `artifacts/logs/phase1_baseline_smoke.json`.

### Gate 2 — Custom Pallas Static-Shape Kernel & Zero-Recompile Verification
```bash
bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && pytest -v -s tests/test_pallas_block_verify.py"
```
- **Troubleshooting TPU VMEM / Mosaic Lowering**:
  - All Pallas `BlockSpec` shapes in `kernels/pallas_block_verify.py` use `(128, 1024)` inputs and `(128, 128)` VMEM scratch/outputs (~3.5 MiB total VMEM footprint, well inside TPU v7's scoped VMEM budget).
  - If a newer JAX/Mosaic release changes `pltpu.PrefetchScalarGridSpec` argument order (`in_specs`, `out_specs`, `scratch_shapes`, `grid`, `dimension_semantics`), inspect `inspect.signature(pltpu.PrefetchScalarGridSpec)` on the VM and align keyword names in `kernels/pallas_block_verify.py`.

### Gate 3 — 4-Mode Speculative Matrix, XProf `.xplane.pb` Capture & Quality Parity
1. **Mandatory Fast Smoke Test First (~2 minutes)**:
   ```bash
   bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 benchmarks/run_spec_matrix.py --smoke"
   ```
2. **Full Benchmark Sweep (`B in [1, 4, 8, 16, 32]` x 3 Real Prompt Suites x All Modes)**:
   ```bash
   bash -c "source scripts/env_common.sh && cleanup_stale_tpu_processes && python3 benchmarks/run_spec_matrix.py --full --resume"
   ```
3. **Parse `.xplane.pb` Hardware Traces**:
   ```bash
   bash -c "source scripts/env_common.sh && python3 scripts/parse_xprof.py"
   ```
   *(Note on GitHub 100 MB file limit: If raw `.xplane.pb` traces exceed 50 MB, compress them with `gzip artifacts/traces/*/*.xplane.pb`—both `scripts/parse_xprof.py` and `scripts/validate_artifacts.py` natively read `.xplane.pb.gz`!)*

### Gate 4 — Analytical TPU v7 Roofline Fit, Publication Plots & `run_all.sh`
```bash
bash -c "source scripts/env_common.sh && python3 benchmarks/fit_roofline_model.py && python3 scripts/generate_final_reports.py"
bash scripts/run_all.sh
```
- Confirm `bash scripts/run_all.sh` exits with code `0` and `STATUS.json` shows `"overall_state": "ALL_PHASES_COMPLETE_AND_VERIFIED"`.
- Commit and push all results:
  ```bash
  git add STATUS.json README.md docs/ artifacts/logs/ artifacts/plots/ artifacts/traces/
  git commit -m "Complete Phase 0-4 TPU v7 empirical benchmarks, XProf traces, and roofline analysis"
  git push origin main
  ```
