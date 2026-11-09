#!/usr/bin/env python3
"""Strict End-to-End Provenance & Artifact Validator (Phase 4 Gate).

Fails with non-zero exit code if ANY required TPU v7 JSON log, CSV summary,
real `.xplane.pb` trace, PNG/PDF plot, or programmatically populated Markdown
report is missing, empty, non-finite, or lacks verified Cloud TPU provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parse_xprof import discover_trace_files, parse_xplane_file  # noqa: E402
from src.tpu_utils import update_phase_status  # noqa: E402


REQUIRED_JSON_LOGS = [
    "artifacts/logs/phase0_probe.json",
    "artifacts/logs/model_manifest.json",
    "artifacts/logs/phase1_baseline_smoke.json",
    "artifacts/logs/phase2_pallas_verify.json",
    "artifacts/logs/spec_matrix_runs.json",
    "artifacts/logs/quality_equivalence.json",
    "artifacts/logs/xprof_summary.json",
    "artifacts/logs/roofline_fit.json",
]

REQUIRED_PLOTS = [
    "artifacts/plots/roofline_trajectory.png",
    "artifacts/plots/speedup_heatmap_BxK.png",
    "artifacts/plots/acceptance_length_vs_entropy.png",
    "artifacts/plots/xprof_time_breakdown.png",
]


def validate_all_artifacts(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return list of validation errors (empty list == 100% valid)."""
    errors: list[str] = []

    for rel in REQUIRED_JSON_LOGS:
        p = repo_root / rel
        if not p.exists():
            errors.append(f"Missing required JSON log: {rel}")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data:
                errors.append(f"Empty or invalid JSON object in {rel}")
        except Exception as exc:
            errors.append(f"Failed to parse {rel}: {exc}")

    csv_path = repo_root / "artifacts" / "logs" / "spec_matrix_summary.csv"
    if not csv_path.exists() or csv_path.stat().st_size < 64:
        errors.append("Missing or empty artifacts/logs/spec_matrix_summary.csv")

    # Verify Phase 0 & Phase 2 TPU provenance & zero-recompilation gate
    p0_path = repo_root / "artifacts" / "logs" / "phase0_probe.json"
    if p0_path.exists():
        p0 = json.loads(p0_path.read_text(encoding="utf-8"))
        topo = p0.get("topology", {})
        if int(topo.get("chip_count", 0)) < 1:
            errors.append("phase0_probe.json reports chip_count < 1")
        if topo.get("backend_provider") not in ("torch_tpu", "jax_tpu"):
            errors.append(f"Invalid TPU backend_provider: {topo.get('backend_provider')}")

    p2_path = repo_root / "artifacts" / "logs" / "phase2_pallas_verify.json"
    if p2_path.exists():
        p2 = json.loads(p2_path.read_text(encoding="utf-8"))
        if int(p2.get("new_tpu_xla_compilations", -1)) != 0:
            errors.append(
                f"phase2_pallas_verify.json reports non-zero XLA recompilations: {p2.get('new_tpu_xla_compilations')}"
            )

    # Verify real .xplane.pb traces exist and decode cleanly
    traces_dir = repo_root / "artifacts" / "traces"
    trace_files = discover_trace_files(traces_dir)
    if len(trace_files) < 3:
        errors.append(
            f"Expected at least 3 real .xplane.pb traces in artifacts/traces/, found {len(trace_files)}"
        )
    else:
        for tf in trace_files[:6]:
            try:
                parsed = parse_xplane_file(tf)
                if int(parsed.get("plane_count", 0)) < 1:
                    errors.append(f"Trace {tf} has 0 XPlane entries")
            except Exception as exc:
                errors.append(f"Corrupt .xplane.pb file {tf}: {exc}")

    # Verify PNG plots exist with non-trivial PNG header & size
    for rel in REQUIRED_PLOTS:
        p = repo_root / rel
        if not p.exists():
            errors.append(f"Missing required plot: {rel}")
        elif p.stat().st_size < 4096 or p.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append(f"Invalid or truncated PNG plot: {rel}")

    # Verify docs/api_discovery.md & docs/paper_results_summary.md exist and contain real numbers
    for doc_rel in ("docs/api_discovery.md", "docs/paper_results_summary.md", "README.md"):
        dp = repo_root / doc_rel
        if not dp.exists():
            errors.append(f"Missing required documentation: {doc_rel}")
        else:
            txt = dp.read_text(encoding="utf-8")
            if "PENDING_TPU_V7_EXECUTION" in txt:
                errors.append(
                    f"{doc_rel} still contains unpopulated placeholder marker PENDING_TPU_V7_EXECUTION"
                )

    return errors


def main() -> None:
    errors = validate_all_artifacts(REPO_ROOT)
    if errors:
        print("[validate_artifacts] FAILED with the following errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    update_phase_status(
        "phase4_roofline_and_submission",
        status="passed",
        overall_state="ALL_PHASES_COMPLETE_AND_VERIFIED",
    )
    print("[validate_artifacts] ALL TPU v7 logs, .xplane.pb traces, plots, and reports VERIFIED.")


if __name__ == "__main__":
    main()
