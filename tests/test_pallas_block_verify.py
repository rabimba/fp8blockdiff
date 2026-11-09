"""PHASE 2 — CUSTOM PALLAS STATIC-SHAPE BLOCK VERIFICATION & ZERO-RECOMPILE GATE.

Tests `kernels/pallas_block_verify.py` on `torch.device("tpu")`:
1. Numerical equivalence (`torch.testing.assert_close` and exact integer equality)
   against pure PyTorch eager reference on TPU across batch sizes [1, 4, 16, 32],
   padded block capacities [16, 32, 64], and varying entropy thresholds.
2. Full compatibility inside `torch.compile(backend="tpu")` with ZERO new XLA
   graph compilations after warmup across 20 consecutive steps with fluctuating
   dynamic acceptance lengths and changing runtime entropy thresholds.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time

import pytest
import torch

from kernels.pallas_block_verify import (
    pallas_entropy_block_verify,
    pytorch_reference_block_verify,
)
from src.tpu_utils import (
    REPO_ROOT,
    ensure_persistent_cache_env,
    require_tpu_device,
    snapshot_xla_compilation_metrics,
    update_phase_status,
)


pytestmark = pytest.mark.tpu_required


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("k_pad", [16, 32, 64])
@pytest.mark.parametrize("entropy_thresh", [0.4, 1.5, 3.5])
def test_pallas_block_verify_matches_tpu_eager_reference(
    batch_size: int,
    k_pad: int,
    entropy_thresh: float,
) -> None:
    ensure_persistent_cache_env()
    topo = require_tpu_device()
    del topo
    device = torch.device("tpu")

    vocab_size = 4096  # Exact multiple of 1024; full 151936 Qwen3 vocab also tested below
    torch.manual_seed(1000 + batch_size * 37 + k_pad)

    # Construct controlled logits with mixed low-entropy (peaked) and high-entropy rows
    logits_cpu = torch.randn((batch_size, k_pad + 1, vocab_size), dtype=torch.float32) * 1.2
    greedy_cpu = torch.argmax(logits_cpu, dim=-1).to(torch.int32)

    # Make prefix of random length per batch row match greedy target tokens
    draft_cpu = torch.randint(0, vocab_size, (batch_size, k_pad), dtype=torch.int32)
    active_mask_cpu = torch.ones((batch_size, k_pad), dtype=torch.int32)
    for b in range(batch_size):
        accept_target = (b * 5 + 3) % (k_pad + 1)
        if accept_target > 0:
            draft_cpu[b, :accept_target] = greedy_cpu[b, :accept_target]
        # Force mismatch right after accept_target if within k_pad
        if accept_target < k_pad:
            draft_cpu[b, accept_target] = (greedy_cpu[b, accept_target] + 1) % vocab_size
        # Also test partially masked active draft tails
        eff_curr = max(4, k_pad - (b % 4) * 2)
        active_mask_cpu[b, eff_curr:] = 0

    ctrl_cpu = torch.tensor(
        [entropy_thresh, float(batch_size), 4.0, float(k_pad), 512.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    logits_tpu = logits_cpu.to(torch.bfloat16).to(device)
    draft_tpu = draft_cpu.to(device)
    mask_tpu = active_mask_cpu.to(device)
    ctrl_tpu = ctrl_cpu.to(device)

    ref_out = pytorch_reference_block_verify(
        target_logits=logits_tpu,
        draft_token_ids=draft_tpu,
        active_draft_mask=mask_tpu,
        control_params=ctrl_tpu,
        valid_vocab_size=vocab_size,
    )
    pallas_out = pallas_entropy_block_verify(
        target_logits=logits_tpu,
        draft_token_ids=draft_tpu,
        active_draft_mask=mask_tpu,
        control_params=ctrl_tpu,
        valid_vocab_size=vocab_size,
        world_size=1,
    )

    torch.testing.assert_close(
        pallas_out.accepted_lengths.to("cpu"),
        ref_out.accepted_lengths.to("cpu"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pallas_out.bonus_token_ids.to("cpu"),
        ref_out.bonus_token_ids.to("cpu"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pallas_out.token_entropies.to("cpu"),
        ref_out.token_entropies.to("cpu"),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        pallas_out.entropy_tail_mask.to("cpu"),
        ref_out.entropy_tail_mask.to("cpu"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pallas_out.next_effective_k.to("cpu"),
        ref_out.next_effective_k.to("cpu"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pallas_out.next_draft_mask.to("cpu"),
        ref_out.next_draft_mask.to("cpu"),
        rtol=0,
        atol=0,
    )


def test_qwen3_full_vocab_and_zero_xla_recompilations_across_20_dynamic_steps() -> None:
    """Verify Qwen3 vocab (151,936) and assert ZERO XLA recompilations over 20 steps."""
    ensure_persistent_cache_env()
    topo = require_tpu_device()
    device = torch.device("tpu")

    batch_size = 8
    k_pad = 32
    qwen3_vocab = 151936

    def compiled_step(
        logits: torch.Tensor,
        draft_ids: torch.Tensor,
        active_mask: torch.Tensor,
        ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        res = pallas_entropy_block_verify(
            target_logits=logits,
            draft_token_ids=draft_ids,
            active_draft_mask=active_mask,
            control_params=ctrl,
            valid_vocab_size=qwen3_vocab,
            world_size=1,
        )
        return (
            res.accepted_lengths,
            res.bonus_token_ids,
            res.next_effective_k,
            res.next_draft_mask,
        )

    step_fn = torch.compile(compiled_step, backend="tpu")

    torch.manual_seed(2026)
    base_logits = torch.randn(
        (batch_size, k_pad + 1, qwen3_vocab), dtype=torch.bfloat16, device=device
    )
    init_draft = torch.randint(0, qwen3_vocab, (batch_size, k_pad), dtype=torch.int32, device=device)
    active_mask = torch.ones((batch_size, k_pad), dtype=torch.int32, device=device)
    ctrl = torch.tensor(
        [1.65, float(batch_size), 4.0, float(k_pad), 512.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )

    # Warmup 2 steps to populate XLA compilation graph once
    for _ in range(2):
        acc, bonus, next_k, active_mask = step_fn(base_logits, init_draft, active_mask, ctrl)
    if hasattr(torch.tpu, "synchronize"):
        torch.tpu.synchronize()

    before_metrics = snapshot_xla_compilation_metrics()

    observed_accept_means: list[float] = []
    step_latencies_ms: list[float] = []
    greedy_ids = torch.argmax(base_logits.to(torch.float32), dim=-1).to(torch.int32)

    # Execute 20 consecutive steps with fluctuating acceptance lengths & entropy thresholds
    for step_idx in range(20):
        target_prefix = (step_idx * 7 + 3) % (k_pad + 1)
        dyn_draft = torch.randint(
            0, qwen3_vocab, (batch_size, k_pad), dtype=torch.int32, device=device
        )
        if target_prefix > 0:
            dyn_draft[:, :target_prefix] = greedy_ids[:, :target_prefix]

        dyn_thresh = 0.6 + 0.2 * float(step_idx % 9)
        ctrl = torch.tensor(
            [dyn_thresh, float(batch_size), 4.0, float(k_pad), 512.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        )

        t0 = time.perf_counter()
        acc, bonus, next_k, active_mask = step_fn(base_logits, dyn_draft, active_mask, ctrl)
        if hasattr(torch.tpu, "synchronize"):
            torch.tpu.synchronize()
        step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        observed_accept_means.append(float(acc.to("cpu").to(torch.float32).mean().item()))
        del bonus, next_k

    after_metrics = snapshot_xla_compilation_metrics()

    new_dynamo_compiles = (
        after_metrics["dynamo_frame_compiles"] - before_metrics["dynamo_frame_compiles"]
    )
    new_tpu_compiles = (
        after_metrics["tpu_backend_compiles"] - before_metrics["tpu_backend_compiles"]
    )

    assert new_dynamo_compiles == 0, (
        f"Expected ZERO new Dynamo graph compilations after warmup, got {new_dynamo_compiles}"
    )
    assert new_tpu_compiles == 0, (
        f"Expected ZERO new TPU XLA graph compilations after warmup, got {new_tpu_compiles}"
    )

    log_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "topology": topo.to_dict(),
        "batch_size": batch_size,
        "k_pad": k_pad,
        "vocab_size": qwen3_vocab,
        "num_dynamic_steps": 20,
        "new_dynamo_compilations": new_dynamo_compiles,
        "new_tpu_xla_compilations": new_tpu_compiles,
        "mean_step_latency_ms": round(sum(step_latencies_ms) / len(step_latencies_ms), 4),
        "min_step_latency_ms": round(min(step_latencies_ms), 4),
        "max_step_latency_ms": round(max(step_latencies_ms), 4),
        "observed_mean_accepted_lengths": [round(x, 2) for x in observed_accept_means],
    }

    out_path = REPO_ROOT / "artifacts" / "logs" / "phase2_pallas_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log_payload, indent=2) + "\n", encoding="utf-8")

    update_phase_status(
        "phase2_pallas_block_verify",
        status="passed",
        overall_state="PHASE_2_PASSED",
        extra={
            "new_tpu_xla_compilations": new_tpu_compiles,
            "mean_step_latency_ms": log_payload["mean_step_latency_ms"],
        },
    )
