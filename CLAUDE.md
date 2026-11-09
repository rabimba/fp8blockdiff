# CLAUDE.md — Claude Code Entry Point

Read [`AGENTS.md`](./AGENTS.md), [`HANDOFF.md`](./HANDOFF.md), and [`STATUS.json`](./STATUS.json) immediately upon opening this repository.

## Summary of Takeover Task on Cloud TPU v7 (Ironwood / v7x)
1. This repository implements **"FP8 Block-Diffusion Speculative Decoding on Systolic Arrays"** (`torch_tpu`, `vllm-torchtpu`, Pallas/Tokamax `torch.tpu.pallas.jax_op`, XProf `.xplane.pb`, and TPU v7 roofline modeling).
2. All core modules, Pallas kernels, benchmark runners, XProf wire-format parsers, roofline solvers, and unit tests are implemented, with **zero fabricated or CPU-substituted TPU metrics** (`STATUS.json` = `AWAITING_TPU_V7_EXECUTION`).
3. Follow the **Non-Negotiable Execution Contract** and sequential gates in [`AGENTS.md`](./AGENTS.md) (`Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4`) on your Cloud TPU v7 VM until `bash scripts/run_all.sh` exits with code `0`, then commit and push the real empirical artifacts and reports.
