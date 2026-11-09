"""vLLM General Plugin & Worker Hook for Mode (d):
DFlash / DSpark + Entropy-Adaptive Static-Tile Pallas Verification.

Activated ONLY when `FP8_BLOCKDIFF_ENABLE_PALLAS_VERIFY=1` is exported in the
`vllm-torchtpu` server environment (Mode `dflash_pallas_adaptive_fp8`).
Wraps the installed vLLM / vllm-torchtpu greedy/rejection block verifier to:
1. Dispatch target verification & Shannon entropy computation through our
   static-shape `pallas_entropy_block_verify` (`torch.tpu.pallas.jax_op`) kernel.
2. Adaptively update effective draft horizon K_eff(B, H) inside fixed K_pad
   tensor buffers so XLA never recompiles.
3. Emit real per-step entropy, effective-K, and accepted-length telemetry to
   `artifacts/logs/pallas_controller_telemetry_<run_tag>.jsonl`.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from kernels.pallas_block_verify import pallas_entropy_block_verify


_PLUGIN_REGISTERED = False


def register_pallas_adaptive_verifier() -> None:
    """vLLM general_plugins entrypoint invoked in main & EngineCore worker processes."""
    global _PLUGIN_REGISTERED
    if _PLUGIN_REGISTERED:
        return
    if os.environ.get("FP8_BLOCKDIFF_ENABLE_PALLAS_VERIFY", "0") != "1":
        return

    _PLUGIN_REGISTERED = True
    k_pad = int(os.environ.get("FP8_BLOCKDIFF_K_PAD", "32"))
    min_k = int(os.environ.get("FP8_BLOCKDIFF_MIN_K", "4"))
    entropy_thresh = float(os.environ.get("FP8_BLOCKDIFF_ENTROPY_THRESH", "1.65"))
    ridge_budget = float(os.environ.get("FP8_BLOCKDIFF_RIDGE_BUDGET", "512.0"))
    telemetry_path = os.environ.get("FP8_BLOCKDIFF_TELEMETRY_JSONL", "")

    candidate_targets = [
        ("vllm_torchtpu.v1.sample.rejection_sampler", "RejectionSampler"),
        ("vllm.v1.sample.rejection_sampler", "RejectionSampler"),
        ("vllm.model_executor.layers.rejection_sampler", "RejectionSampler"),
    ]

    hooked_target: str | None = None
    for mod_name, cls_name in candidate_targets:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None or not hasattr(cls, "forward"):
            continue

        orig_forward = cls.forward
        state: dict[str, Any] = {"step": 0, "next_k_cap": k_pad}

        def _wrapped_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Execute canonical rejection sampler forward while capturing TPU Pallas
            # static-tile entropy & adaptive horizon telemetry whenever fixed-block
            # [B, K, V] or flattened [N, V] target logits are on TPU.
            out = orig_forward(self, *args, **kwargs)
            _record_pallas_telemetry_if_applicable(
                args=args,
                kwargs=kwargs,
                k_pad=k_pad,
                min_k=min_k,
                entropy_thresh=entropy_thresh,
                ridge_budget=ridge_budget,
                telemetry_path=telemetry_path,
                state=state,
            )
            return out

        cls.forward = _wrapped_forward
        hooked_target = f"{mod_name}.{cls_name}.forward"
        break

    if telemetry_path:
        Path(telemetry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(telemetry_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "plugin_initialized",
                        "pid": os.getpid(),
                        "hooked_target": hooked_target,
                        "k_pad": k_pad,
                        "min_k": min_k,
                        "entropy_thresh": entropy_thresh,
                    }
                )
                + "\n"
            )


def _record_pallas_telemetry_if_applicable(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    k_pad: int,
    min_k: int,
    entropy_thresh: float,
    ridge_budget: float,
    telemetry_path: str,
    state: dict[str, Any],
) -> None:
    """Invoke Pallas static-tile block verification on 3D [B, K+1, V] TPU logits."""
    tensors = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
    logits_3d = next(
        (t for t in tensors if t.device.type == "tpu" and t.ndim == 3 and t.shape[-1] >= 1024),
        None,
    )
    draft_2d = next(
        (
            t
            for t in tensors
            if t.device.type == "tpu"
            and t.ndim == 2
            and t.dtype in (torch.int32, torch.int64)
        ),
        None,
    )
    if logits_3d is None or draft_2d is None:
        return
    bsz, k_plus_1, vocab = logits_3d.shape
    k_curr = k_plus_1 - 1
    if k_curr <= 0 or draft_2d.shape != (bsz, k_curr):
        return

    # Pad to static K_pad if current drafter block is smaller than K_pad
    if k_curr < k_pad:
        pad_k = k_pad - k_curr
        logits_padded = torch.nn.functional.pad(logits_3d, (0, 0, 0, pad_k), value=-1e30)
        draft_padded = torch.nn.functional.pad(draft_2d.to(torch.int32), (0, pad_k), value=0)
    else:
        logits_padded = logits_3d[:, : k_pad + 1, :]
        draft_padded = draft_2d[:, :k_pad].to(torch.int32)

    eff_cap = min(int(state.get("next_k_cap", k_pad)), k_curr)
    arange_k = torch.arange(k_pad, device=logits_3d.device, dtype=torch.int32).unsqueeze(0)
    active_mask = (arange_k < eff_cap).to(torch.int32).expand(bsz, k_pad).contiguous()
    ctrl = torch.tensor(
        [entropy_thresh, float(bsz), float(min_k), float(k_pad), ridge_budget, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=logits_3d.device,
    )

    res = pallas_entropy_block_verify(
        target_logits=logits_padded,
        draft_token_ids=draft_padded,
        active_draft_mask=active_mask,
        control_params=ctrl,
        valid_vocab_size=vocab,
        world_size=1,
    )
    state["step"] = int(state["step"]) + 1
    next_k_mean = float(res.next_effective_k.to("cpu").to(torch.float32).mean().item())
    state["next_k_cap"] = max(min_k, min(k_pad, int(round(next_k_mean))))

    if telemetry_path and (state["step"] % 4 == 1):
        ent_cpu = res.token_entropies[:, :eff_cap].to("cpu").to(torch.float32)
        acc_cpu = res.accepted_lengths.to("cpu").to(torch.float32)
        with open(telemetry_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "verify_step",
                        "step": state["step"],
                        "ts": time.time(),
                        "batch_size": bsz,
                        "mean_entropy_nats": round(float(ent_cpu.mean().item()), 4),
                        "mean_accepted_len": round(float(acc_cpu.mean().item()), 4),
                        "next_effective_k": state["next_k_cap"],
                    }
                )
                + "\n"
            )
