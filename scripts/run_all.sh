#!/usr/bin/env bash
# End-to-End Reproducible Execution & Verification Pipeline (Phase 4 Gate).
# Sequence:
#   1. TPU VFIO & worker process lock cleanup
#   2. Pytest unit & TPU correctness suite (host math + baseline serving + Pallas zero-recompile)
#   3. Cached end-to-end speculative benchmark + XProf trace parse + Roofline fit + Report generation
#   4. Strict programmatic artifact & provenance validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Step 1: Persistent compilation cache & TPU lock cleanup
source "${SCRIPT_DIR}/env_common.sh"
cleanup_stale_tpu_processes

echo "=================================================================="
echo "[run_all] Step 1/4: Phase 0 Hardware & Ecosystem Discovery Check"
echo "=================================================================="
python3 scripts/phase0_discover.py --no-install

echo "=================================================================="
echo "[run_all] Step 2/4: Model Manifest Check & Pytest Correctness Suite"
echo "=================================================================="
if [[ ! -f "artifacts/logs/model_manifest.json" ]]; then
  python3 scripts/download_and_verify_models.py
fi

cleanup_stale_tpu_processes
pytest -v -s tests/test_host_math_and_xprof_parser.py
cleanup_stale_tpu_processes
pytest -v -s tests/test_pallas_block_verify.py
cleanup_stale_tpu_processes
pytest -v -s tests/test_baseline_serving.py

echo "=================================================================="
echo "[run_all] Step 3/4: Cached End-to-End Speculative Matrix & Roofline"
echo "=================================================================="
cleanup_stale_tpu_processes
if [[ ! -f "artifacts/logs/spec_matrix_runs.json" ]]; then
  python3 benchmarks/run_spec_matrix.py --smoke
  cleanup_stale_tpu_processes
  python3 benchmarks/run_spec_matrix.py --full --resume
else
  # Re-verify cached end-to-end serving pipeline via fast smoke pass while preserving full logs
  python3 benchmarks/run_spec_matrix.py --smoke
fi

cleanup_stale_tpu_processes
python3 scripts/parse_xprof.py
python3 benchmarks/fit_roofline_model.py
python3 scripts/generate_final_reports.py

echo "=================================================================="
echo "[run_all] Step 4/4: Programmatic Artifact & Provenance Validation"
echo "=================================================================="
python3 scripts/validate_artifacts.py

echo "[run_all] SUCCESS — All phases verified with exit code 0."
