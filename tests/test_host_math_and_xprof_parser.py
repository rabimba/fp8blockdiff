"""Host-runnable deterministic unit tests for:
1. Tiled VMEM online Shannon entropy & tie-consistent argmax recurrence math.
2. Lossless contiguous prefix verification & entropy-adaptive block horizon logic.
3. Pure-Python OpenXLA `XSpace` (`.xplane.pb`) binary protobuf wire-format parser.
4. Analytical Cloud TPU v7 (Ironwood) roofline ridge-point & K*(B, H) optimizer.
5. Strict zero-mock enforcement (`validate_artifacts` rejects unpopulated workspace,
   and `pallas_entropy_block_verify` refuses CPU tensors).

Does NOT claim or mock TPU execution; runs in <3 seconds on any host and on TPU v7.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest
import torch

from benchmarks.fit_roofline_model import (
    compute_verification_arithmetic_intensity,
    solve_optimal_block_size_k,
)
from kernels.pallas_block_verify import (
    align_up,
    pallas_entropy_block_verify,
    pytorch_reference_block_verify,
    tiled_online_entropy_and_argmax_numpy,
)
from scripts.parse_xprof import parse_xspace_bytes
from scripts.validate_artifacts import validate_all_artifacts


pytestmark = pytest.mark.host_only


# --- Minimal Protobuf Wire Encoder for In-Memory XSpace Round-Trip Test ---

def _enc_varint(val: int) -> bytes:
    if val < 0:
        val += 1 << 64
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _enc_tag(field_num: int, wire_type: int) -> bytes:
    return _enc_varint((field_num << 3) | wire_type)


def _enc_len_delim(field_num: int, payload: bytes) -> bytes:
    return _enc_tag(field_num, 2) + _enc_varint(len(payload)) + payload


def _enc_str(field_num: int, text: str) -> bytes:
    return _enc_len_delim(field_num, text.encode("utf-8"))


def _enc_int64(field_num: int, val: int) -> bytes:
    return _enc_tag(field_num, 0) + _enc_varint(val)


def _enc_double(field_num: int, val: float) -> bytes:
    return _enc_tag(field_num, 1) + struct.pack("<d", val)


def _enc_meta_map_entry(meta_id: int, name: str) -> bytes:
    meta_msg = _enc_int64(1, meta_id) + _enc_str(2, name)
    return _enc_int64(1, meta_id) + _enc_len_delim(2, meta_msg)


def _enc_xstat_double(meta_id: int, val: float) -> bytes:
    return _enc_int64(1, meta_id) + _enc_double(2, val)


def _enc_xevent(meta_id: int, duration_ps: int, stats: list[bytes]) -> bytes:
    return _enc_int64(1, meta_id) + _enc_int64(3, duration_ps) + b"".join(
        _enc_len_delim(4, s) for s in stats
    )


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("k_pad", [16, 32, 64])
def test_tiled_online_entropy_and_lossless_prefix_math(batch_size: int, k_pad: int) -> None:
    valid_vocab = 3700
    v_pad = align_up(valid_vocab, 1024)
    torch.manual_seed(batch_size * 101 + k_pad)

    # Include large +/- offsets and duplicate tie values to stress numerical stability
    logits = torch.randn((batch_size, k_pad + 1, v_pad), dtype=torch.float32) * 3.0 + 120.0
    logits[:, 0, 10] = 250.0
    logits[:, 0, 25] = 250.0  # Exact tie: argmax must pick index 10 in both implementations

    flat_np = logits.reshape(-1, v_pad).numpy()
    tiled_ent, tiled_arg = tiled_online_entropy_and_argmax_numpy(
        flat_np, valid_vocab_size=valid_vocab, vocab_tile=1024
    )

    draft = torch.randint(0, valid_vocab, (batch_size, k_pad), dtype=torch.int32)
    greedy = torch.argmax(logits[:, :, :valid_vocab], dim=-1).to(torch.int32)
    for b in range(batch_size):
        k_match = b % (k_pad + 1)
        if k_match > 0:
            draft[b, :k_match] = greedy[b, :k_match]
        if k_match < k_pad:
            draft[b, k_match] = (greedy[b, k_match] + 1) % valid_vocab

    mask = torch.ones((batch_size, k_pad), dtype=torch.int32)
    ctrl = torch.tensor(
        [1.5, float(batch_size), 4.0, float(k_pad), 512.0, 0.0, 0.0, 0.0], dtype=torch.float32
    )
    ref = pytorch_reference_block_verify(
        target_logits=logits,
        draft_token_ids=draft,
        active_draft_mask=mask,
        control_params=ctrl,
        valid_vocab_size=valid_vocab,
    )

    np.testing.assert_allclose(
        tiled_ent.reshape(batch_size, k_pad + 1),
        ref.token_entropies.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        tiled_arg.reshape(batch_size, k_pad + 1),
        greedy.numpy(),
    )
    assert int(tiled_arg.reshape(batch_size, k_pad + 1)[0, 0]) == 10

    # Verify 100% lossless greedy equivalence of accepted prefix + bonus token
    for b in range(batch_size):
        tau = int(ref.accepted_lengths[b].item())
        emitted = draft[b, :tau].tolist() + [int(ref.bonus_token_ids[b].item())]
        expected_greedy = greedy[b, : tau + 1].tolist()
        assert emitted == expected_greedy


def test_uniform_distribution_entropy_identity() -> None:
    vocab = 2048
    logits = np.zeros((4, vocab), dtype=np.float32)
    ent, idx = tiled_online_entropy_and_argmax_numpy(logits, valid_vocab_size=vocab, vocab_tile=1024)
    np.testing.assert_allclose(ent, math.log(vocab), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(idx, np.zeros(4, dtype=np.int32))


def test_xspace_wire_format_parser_roundtrip() -> None:
    # Build an in-memory OpenXLA XSpace binary buffer (1 TPU device plane + 1 Host plane)
    # Device plane: 1ms total duration (1e9 ps):
    #   - dot_gemm: 0.6ms, 2.307e12 FLOPs (2307 TFLOPS = 50% of 4614 FP8 peak), 3.69e9 bytes (3690 GB/s = 50% of 7380 GB/s)
    #   - dflash_draft: 0.25ms
    #   - fp8blockdiff::pallas_block_verify: 0.15ms
    ev_dot = _enc_xevent(
        1,
        600_000_000,
        [_enc_xstat_double(10, 2.307e12), _enc_xstat_double(11, 3.69e9)],
    )
    ev_draft = _enc_xevent(2, 250_000_000, [])
    ev_pallas = _enc_xevent(3, 150_000_000, [])
    dev_line = _enc_int64(1, 0) + _enc_len_delim(7, ev_dot) + _enc_len_delim(7, ev_draft) + _enc_len_delim(7, ev_pallas)

    dev_plane = (
        _enc_int64(1, 0)
        + _enc_str(2, "/device:TPU:0")
        + _enc_len_delim(3, dev_line)
        + _enc_len_delim(4, _enc_meta_map_entry(1, "fusion_dot_gemm"))
        + _enc_len_delim(4, _enc_meta_map_entry(2, "dflash_proposer"))
        + _enc_len_delim(4, _enc_meta_map_entry(3, "fp8blockdiff::pallas_block_verify"))
        + _enc_len_delim(5, _enc_meta_map_entry(10, "flops"))
        + _enc_len_delim(5, _enc_meta_map_entry(11, "bytes_accessed"))
    )

    host_sync_ev = _enc_xevent(20, 5_000_000, [])
    host_line = _enc_int64(1, 1) + _enc_len_delim(7, host_sync_ev) + _enc_len_delim(7, host_sync_ev)
    host_plane = (
        _enc_int64(1, 1)
        + _enc_str(2, "/host:CPU")
        + _enc_len_delim(3, host_line)
        + _enc_len_delim(4, _enc_meta_map_entry(20, "TransferFromDevice"))
    )

    xspace_bytes = _enc_len_delim(1, dev_plane) + _enc_len_delim(1, host_plane)
    parsed = parse_xspace_bytes(xspace_bytes, precision="fp8", chip_count=1)

    assert parsed["plane_count"] == 2
    assert math.isclose(parsed["total_device_time_ms"], 1.0, rel_tol=1e-5)
    assert math.isclose(parsed["mxu_utilization_pct"], 50.0, rel_tol=1e-3)
    assert math.isclose(parsed["hbm_bw_utilization_pct"], 50.0, rel_tol=1e-3)
    assert parsed["host_device_sync_count"] == 2
    assert parsed["xla_compilation_count"] == 0
    assert math.isclose(parsed["breakdown_ms"]["pallas_verify_ms"], 0.15, rel_tol=1e-4)
    assert math.isclose(parsed["breakdown_ms"]["draft_model_ms"], 0.25, rel_tol=1e-4)


def test_analytical_roofline_optimal_block_scaling() -> None:
    ai_b1_k0, _, _ = compute_verification_arithmetic_intensity(batch_size=1, block_k=0)
    ai_b16_k32, _, _ = compute_verification_arithmetic_intensity(batch_size=16, block_k=32)
    assert ai_b1_k0 < 5.0
    assert ai_b16_k32 > 625.2  # Exceeds TPU v7 FP8 ridge point

    k_low_b_low_h, _ = solve_optimal_block_size_k(batch_size=1, entropy_nats=0.45)
    k_high_b_high_h, _ = solve_optimal_block_size_k(batch_size=32, entropy_nats=2.20)
    assert k_low_b_low_h > k_high_b_high_h


def test_zero_mock_guards_refuse_cpu_and_unpopulated_artifacts() -> None:
    logits_cpu = torch.zeros((2, 17, 1024), dtype=torch.float32)
    draft_cpu = torch.zeros((2, 16), dtype=torch.int32)
    mask_cpu = torch.ones((2, 16), dtype=torch.int32)
    ctrl_cpu = torch.tensor([1.5, 2.0, 4.0, 16.0, 512.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="torch.device\\('tpu'\\)"):
        pallas_entropy_block_verify(logits_cpu, draft_cpu, mask_cpu, ctrl_cpu)

    # Confirm artifact validator fails cleanly before real TPU v7 execution
    errs = validate_all_artifacts()
    assert len(errs) > 0
