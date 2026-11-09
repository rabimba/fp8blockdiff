# GEMINI.md — Gemini CLI & Antigravity Entry Point

Read [`AGENTS.md`](../AGENTS.md), [`HANDOFF.md`](../HANDOFF.md), and [`STATUS.json`](../STATUS.json) at repository root before executing any tasks.

- **Current State**: Full codebase implemented and host-math/wire-parser verified; awaiting real hardware execution of **Phases 0–4** on the **Cloud TPU v7 (Ironwood / `tpu7x`) VM**.
- **Strict Rules**: Zero mocks, zero CPU-for-TPU substitution, inspect live `torch_tpu` / `vllm-torchtpu` / `tokamax` signatures in Phase 0 (`docs/api_discovery.md`) before Phase 1, maintain `/dev/vfio/*` lock hygiene via `scripts/env_common.sh`, reuse persistent XLA compilation caches, and run `bash scripts/run_all.sh` to exit code `0` before committing and pushing final results.
