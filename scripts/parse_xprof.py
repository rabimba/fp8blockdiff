#!/usr/bin/env python3
"""Programmatic OpenXLA / TorchTPU XProf (.xplane.pb) trace analyzer.

Extracts real hardware telemetry from `.xplane.pb` (or `.xplane.pb.gz`) traces
captured during Phase 3 at B in {1, 8, 32}:
- MXU utilization %
- HBM3E bandwidth utilization % (and achieved GB/s)
- Host-device synchronization count
- XLA compilation count during steady-state window
- Fine-grained TPU device time breakdown (Target GEMM/MXU, KV/Attention,
  Draft head, Pallas Verification, Host/Collective overhead)

Merges XProf metrics with `artifacts/logs/spec_matrix_runs.json` to emit
`artifacts/logs/xprof_summary.json` and `artifacts/logs/spec_matrix_summary.csv`.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import struct
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TPU_V7_FP8_PEAK_TFLOPS = 4614.0
TPU_V7_BF16_PEAK_TFLOPS = 2307.0
TPU_V7_HBM3E_GBPS = 7380.0


# ---------------------------------------------------------------------------
# Pure-Python Protocol Buffers Wire-Format Reader for OpenXLA `xplane.proto`
# (Zero fragile TensorFlow C++ version dependency; works alongside `xprof`)
# ---------------------------------------------------------------------------

def _read_varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    n = len(buf)
    while pos < n:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift >= 70:
            raise ValueError("Malformed protobuf varint (>10 bytes)")
    raise ValueError("Truncated protobuf varint")


def _to_int64(u64: int) -> int:
    return u64 - (1 << 64) if (u64 & (1 << 63)) else u64


def _iter_proto_fields(buf: memoryview) -> list[tuple[int, int, Any]]:
    fields: list[tuple[int, int, Any]] = []
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:  # varint
            val, pos = _read_varint(buf, pos)
            fields.append((field_num, wire_type, val))
        elif wire_type == 1:  # 64-bit fixed / double
            raw8 = bytes(buf[pos : pos + 8])
            pos += 8
            fields.append((field_num, wire_type, raw8))
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            sub = buf[pos : pos + length]
            pos += length
            fields.append((field_num, wire_type, sub))
        elif wire_type == 5:  # 32-bit fixed
            raw4 = bytes(buf[pos : pos + 4])
            pos += 4
            fields.append((field_num, wire_type, raw4))
        else:
            raise ValueError(f"Unsupported protobuf wire_type={wire_type} at field={field_num}")
    return fields


def _parse_map_id_to_name(entry_buf: memoryview) -> tuple[int, str]:
    key_id = 0
    meta_name = ""
    for fnum, wtype, val in _iter_proto_fields(entry_buf):
        if fnum == 1 and wtype == 0:
            key_id = _to_int64(val)
        elif fnum == 2 and wtype == 2:
            for mfnum, mwtype, mval in _iter_proto_fields(val):
                if mfnum == 2 and mwtype == 2:
                    meta_name = bytes(mval).decode("utf-8", errors="replace")
    return key_id, meta_name


def _parse_xstat(stat_buf: memoryview) -> tuple[int, float | str]:
    meta_id = 0
    value: float | str = 0.0
    for fnum, wtype, val in _iter_proto_fields(stat_buf):
        if fnum == 1 and wtype == 0:
            meta_id = _to_int64(val)
        elif fnum == 2 and wtype == 1:  # double_value
            value = float(struct.unpack("<d", val)[0])
        elif fnum == 3 and wtype == 0:  # uint64_value
            value = float(val)
        elif fnum == 4 and wtype == 0:  # int64_value
            value = float(_to_int64(val))
        elif fnum == 5 and wtype == 2:  # str_value
            value = bytes(val).decode("utf-8", errors="replace")
        elif fnum == 7 and wtype == 0:  # ref_value
            value = float(val)
    return meta_id, value


def parse_xspace_bytes(raw_bytes: bytes, precision: str = "fp8", chip_count: int = 1) -> dict[str, Any]:
    """Decode an OpenXLA XSpace binary protobuf and aggregate TPU hardware metrics."""
    if not raw_bytes:
        raise ValueError("Empty .xplane.pb payload")

    view = memoryview(raw_bytes)
    plane_count = 0
    total_flops = 0.0
    total_bytes_accessed = 0.0
    direct_mxu_pcts: list[float] = []
    direct_hbm_pcts: list[float] = []

    host_sync_count = 0
    xla_compile_count = 0

    cat_ps = {
        "target_mxu_ps": 0.0,
        "kv_attention_ps": 0.0,
        "draft_model_ps": 0.0,
        "pallas_verify_ps": 0.0,
        "collective_or_sync_ps": 0.0,
        "other_device_ps": 0.0,
    }

    for fnum, wtype, val in _iter_proto_fields(view):
        if fnum != 1 or wtype != 2:
            continue
        plane_count += 1
        plane_name = ""
        event_meta: dict[int, str] = {}
        stat_meta: dict[int, str] = {}
        line_blobs: list[memoryview] = []
        plane_stats: list[memoryview] = []

        for pfnum, pwtype, pval in _iter_proto_fields(val):
            if pfnum == 2 and pwtype == 2:
                plane_name = bytes(pval).decode("utf-8", errors="replace")
            elif pfnum == 3 and pwtype == 2:
                line_blobs.append(pval)
            elif pfnum == 4 and pwtype == 2:
                eid, ename = _parse_map_id_to_name(pval)
                event_meta[eid] = ename
            elif pfnum == 5 and pwtype == 2:
                sid, sname = _parse_map_id_to_name(pval)
                stat_meta[sid] = sname
            elif pfnum == 6 and pwtype == 2:
                plane_stats.append(pval)

        is_device_plane = any(
            tok in plane_name.lower() for tok in ("/device:tpu", "tpu:", "tensorcore", "custom:tpu")
        )

        for ps_blob in plane_stats:
            sid, sval = _parse_xstat(ps_blob)
            sname = stat_meta.get(sid, "").lower()
            if isinstance(sval, float):
                if "matrix_unit_utilization" in sname or "mxu_utilization" in sname:
                    direct_mxu_pcts.append(sval)
                elif "memory_bandwidth_utilization" in sname or "hbm_utilization" in sname:
                    direct_hbm_pcts.append(sval)

        for lblob in line_blobs:
            for lfnum, lwtype, lval in _iter_proto_fields(lblob):
                if lfnum != 7 or lwtype != 2:
                    continue
                ev_meta_id = 0
                dur_ps = 0
                ev_stats: list[tuple[int, float | str]] = []
                for efnum, ewtype, eval_ in _iter_proto_fields(lval):
                    if efnum == 1 and ewtype == 0:
                        ev_meta_id = _to_int64(eval_)
                    elif efnum == 3 and ewtype == 0:
                        dur_ps = _to_int64(eval_)
                    elif efnum == 4 and ewtype == 2:
                        ev_stats.append(_parse_xstat(eval_))

                ev_name = event_meta.get(ev_meta_id, "").lower()

                if not is_device_plane:
                    if any(
                        k in ev_name
                        for k in (
                            "xlacompile",
                            "pjrtclient::compile",
                            "compileprogram",
                            "tpucompile",
                        )
                    ):
                        xla_compile_count += 1
                    if any(
                        k in ev_name
                        for k in (
                            "transferfromdevice",
                            "transferfromserver",
                            "synctensors",
                            "blockuntilready",
                            "waitforexecution",
                        )
                    ):
                        host_sync_count += 1
                else:
                    # Categorize TPU device execution duration
                    if "pallas_block_verify" in ev_name or "fp8blockdiff" in ev_name:
                        cat_ps["pallas_verify_ps"] += max(dur_ps, 0)
                    elif any(k in ev_name for k in ("dflash", "eagle", "draft", "speculator")):
                        cat_ps["draft_model_ps"] += max(dur_ps, 0)
                    elif any(
                        k in ev_name
                        for k in ("attention", "splash", "paged", "flash", "kv_cache")
                    ):
                        cat_ps["kv_attention_ps"] += max(dur_ps, 0)
                    elif any(
                        k in ev_name
                        for k in ("all-reduce", "all-gather", "reduce-scatter", "collective")
                    ):
                        cat_ps["collective_or_sync_ps"] += max(dur_ps, 0)
                    elif any(
                        k in ev_name
                        for k in ("dot", "convolution", "fusion", "matmul", "gemm", "custom-call")
                    ):
                        cat_ps["target_mxu_ps"] += max(dur_ps, 0)
                    else:
                        cat_ps["other_device_ps"] += max(dur_ps, 0)

                for sid, sval in ev_stats:
                    if not isinstance(sval, float):
                        continue
                    sname = stat_meta.get(sid, "").lower()
                    if sname in ("flops", "model_flops", "raw_flops"):
                        total_flops += max(sval, 0.0)
                    elif sname in ("bytes_accessed", "memory_bytes", "hbm_bytes_accessed"):
                        total_bytes_accessed += max(sval, 0.0)
                    elif "matrix_unit_utilization" in sname or "mxu_utilization" in sname:
                        direct_mxu_pcts.append(sval)
                    elif "memory_bandwidth_utilization" in sname or "hbm_utilization" in sname:
                        direct_hbm_pcts.append(sval)

    total_device_ps = sum(cat_ps.values())
    total_device_sec = total_device_ps * 1e-12

    peak_tflops = (
        TPU_V7_FP8_PEAK_TFLOPS if precision.lower() == "fp8" else TPU_V7_BF16_PEAK_TFLOPS
    ) * max(chip_count, 1)
    peak_gbps = TPU_V7_HBM3E_GBPS * max(chip_count, 1)

    achieved_tflops = (total_flops / max(total_device_sec, 1e-9)) / 1e12 if total_device_sec > 0 else 0.0
    achieved_gbps = (
        (total_bytes_accessed / max(total_device_sec, 1e-9)) / 1e9
        if total_device_sec > 0
        else 0.0
    )

    computed_mxu_pct = min(100.0, (achieved_tflops / peak_tflops) * 100.0) if peak_tflops > 0 else 0.0
    computed_hbm_pct = min(100.0, (achieved_gbps / peak_gbps) * 100.0) if peak_gbps > 0 else 0.0

    mxu_pct = float(sum(direct_mxu_pcts) / len(direct_mxu_pcts)) if direct_mxu_pcts else computed_mxu_pct
    hbm_pct = float(sum(direct_hbm_pcts) / len(direct_hbm_pcts)) if direct_hbm_pcts else computed_hbm_pct

    return {
        "plane_count": plane_count,
        "total_device_time_ms": round(total_device_ps * 1e-9, 4),
        "achieved_tflops": round(achieved_tflops, 3),
        "achieved_hbm_gbps": round(achieved_gbps, 3),
        "mxu_utilization_pct": round(mxu_pct, 3),
        "hbm_bw_utilization_pct": round(hbm_pct, 3),
        "host_device_sync_count": host_sync_count,
        "xla_compilation_count": xla_compile_count,
        "breakdown_ms": {
            "target_mxu_ms": round(cat_ps["target_mxu_ps"] * 1e-9, 4),
            "kv_attention_ms": round(cat_ps["kv_attention_ps"] * 1e-9, 4),
            "draft_model_ms": round(cat_ps["draft_model_ps"] * 1e-9, 4),
            "pallas_verify_ms": round(cat_ps["pallas_verify_ps"] * 1e-9, 4),
            "collective_or_sync_ms": round(cat_ps["collective_or_sync_ps"] * 1e-9, 4),
            "other_device_ms": round(cat_ps["other_device_ps"] * 1e-9, 4),
        },
    }


def _enrich_with_xprof_plugin_if_available(
    trace_file: Path, base_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Query official `xprof` C++ converter (`overview_page`) if installed on TPU VM."""
    try:
        from xprof.convert import _pywrap_profiler_plugin  # type: ignore[import-not-found]

        raw_data, _ = _pywrap_profiler_plugin.xspace_to_tools_data(
            [str(trace_file)], "overview_page"
        )
        if raw_data:
            payload = json.loads(
                raw_data.decode("utf-8") if isinstance(raw_data, bytes) else raw_data
            )
            text_blob = json.dumps(payload)
            # Extract official overview MXU / HBM utilization if present in JSON table
            if isinstance(payload, list):
                for section in payload:
                    if isinstance(section, dict) and "p" in section:
                        props = section["p"]
                        if "mxu_utilization_percent" in props:
                            base_metrics["mxu_utilization_pct"] = round(
                                float(props["mxu_utilization_percent"]), 3
                            )
                        if "memory_bw_utilization_percent" in props:
                            base_metrics["hbm_bw_utilization_pct"] = round(
                                float(props["memory_bw_utilization_percent"]), 3
                            )
            del text_blob
    except Exception:
        pass
    return base_metrics


def parse_xplane_file(
    trace_path: Path, precision: str = "fp8", chip_count: int = 1
) -> dict[str, Any]:
    if trace_path.suffix == ".gz":
        raw = gzip.decompress(trace_path.read_bytes())
    else:
        raw = trace_path.read_bytes()
    metrics = parse_xspace_bytes(raw, precision=precision, chip_count=chip_count)
    if trace_path.suffix != ".gz":
        metrics = _enrich_with_xprof_plugin_if_available(trace_path, metrics)
    metrics["trace_file"] = str(trace_path.relative_to(REPO_ROOT))
    metrics["trace_size_bytes"] = len(raw)
    return metrics


def discover_trace_files(traces_root: Path) -> list[Path]:
    files = sorted(traces_root.rglob("*.xplane.pb")) + sorted(
        traces_root.rglob("*.xplane.pb.gz")
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse TPU v7 XProf .xplane.pb traces")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "traces",
    )
    parser.add_argument(
        "--runs-json",
        type=Path,
        default=REPO_ROOT / "artifacts" / "logs" / "spec_matrix_runs.json",
    )
    args = parser.parse_args()

    trace_files = discover_trace_files(args.traces_dir)
    if not trace_files:
        raise FileNotFoundError(
            f"No .xplane.pb traces found under {args.traces_dir}. "
            "Run Phase 3 (`benchmarks/run_spec_matrix.py`) on the Cloud TPU v7 VM first."
        )

    chip_count = 1
    probe_path = REPO_ROOT / "artifacts" / "logs" / "phase0_probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        chip_count = int(probe.get("topology", {}).get("chip_count", 1))

    parsed_by_tag: dict[str, dict[str, Any]] = {}
    for tf in trace_files:
        # Directory convention: artifacts/traces/<mode_id>_B<batch>/...xplane.pb
        run_tag = tf.parent.name if tf.parent != args.traces_dir else tf.stem.split(".")[0]
        precision = "bf16" if "bf16" in run_tag.lower() else "fp8"
        parsed = parse_xplane_file(tf, precision=precision, chip_count=chip_count)
        parsed_by_tag[run_tag] = parsed

    xprof_out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chip_count": chip_count,
        "num_traces_parsed": len(parsed_by_tag),
        "traces": parsed_by_tag,
    }
    xprof_json_path = REPO_ROOT / "artifacts" / "logs" / "xprof_summary.json"
    xprof_json_path.parent.mkdir(parents=True, exist_ok=True)
    xprof_json_path.write_text(json.dumps(xprof_out, indent=2) + "\n", encoding="utf-8")

    # Merge with spec_matrix_runs.json into spec_matrix_summary.csv
    if args.runs_json.exists():
        runs_payload = json.loads(args.runs_json.read_text(encoding="utf-8"))
        rows = runs_payload.get("runs", [])
        csv_path = REPO_ROOT / "artifacts" / "logs" / "spec_matrix_summary.csv"
        fieldnames = [
            "mode_id",
            "precision",
            "suite",
            "batch_size",
            "mean_accepted_tokens",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "tpot_p50_ms",
            "tpot_p95_ms",
            "throughput_tok_s",
            "mxu_utilization_pct",
            "hbm_bw_utilization_pct",
            "achieved_tflops",
            "achieved_hbm_gbps",
            "host_device_sync_count",
            "xla_compilation_count",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                tag = f"{r['mode_id']}_B{r['batch_size']}"
                xp = parsed_by_tag.get(tag, {})
                writer.writerow(
                    {
                        "mode_id": r["mode_id"],
                        "precision": r["precision"],
                        "suite": r["suite"],
                        "batch_size": r["batch_size"],
                        "mean_accepted_tokens": r.get("mean_accepted_tokens", 0.0),
                        "ttft_p50_ms": r["ttft_p50_ms"],
                        "ttft_p95_ms": r["ttft_p95_ms"],
                        "tpot_p50_ms": r["tpot_p50_ms"],
                        "tpot_p95_ms": r["tpot_p95_ms"],
                        "throughput_tok_s": r["throughput_tok_s"],
                        "mxu_utilization_pct": xp.get("mxu_utilization_pct", ""),
                        "hbm_bw_utilization_pct": xp.get("hbm_bw_utilization_pct", ""),
                        "achieved_tflops": xp.get("achieved_tflops", ""),
                        "achieved_hbm_gbps": xp.get("achieved_hbm_gbps", ""),
                        "host_device_sync_count": xp.get("host_device_sync_count", ""),
                        "xla_compilation_count": xp.get("xla_compilation_count", ""),
                    }
                )
        print(f"[parse_xprof] Wrote merged CSV summary to {csv_path}")

    print(f"[parse_xprof] Parsed {len(parsed_by_tag)} trace(s) -> {xprof_json_path}")


if __name__ == "__main__":
    main()
