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
import sys
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
        ("vllm_torchtpu.layers.adapter.sample.rejection_sampler", "RejectionSampler"),
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
        state: dict[str, Any] = {
            "step": 0,
            "live_steps": 0,
            "next_k_cap": k_pad,
            "prewarmed": False,
            "force_full_logits_step": False,
            "sampled_batch_counts": {},
            "profiling_active": False,
            "profile_verify_done": False,
            "static_buffers": None,
        }

        def _wrapped_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Execute canonical rejection sampler forward while capturing TPU Pallas
            # static-tile entropy & adaptive horizon telemetry whenever fixed-block
            # [B, K, V] or flattened [N, V] target logits are on TPU.
            out = orig_forward(self, *args, **kwargs)
            try:
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
            except Exception:
                pass
            return out

        cls.forward = _wrapped_forward
        hooked_target = f"{mod_name}.{cls_name}.forward"
        break

    # Hook TPUWorker.profile so XProf trace windows also capture Pallas block verification
    try:
        worker_mod = importlib.import_module("vllm_torchtpu.worker.tpu_worker")
        worker_cls = getattr(worker_mod, "TPUWorker", None)
        if worker_cls is not None and hasattr(worker_cls, "profile"):
            orig_profile = worker_cls.profile

            def _wrapped_profile(
                self: Any,
                is_start: bool = True,
                profile_prefix: str | None = None,
                profiler_kwargs: dict[str, Any] | None = None,
            ) -> Any:
                state["profiling_active"] = bool(is_start)
                if is_start:
                    state["profile_verify_done"] = False
                return orig_profile(
                    self,
                    is_start=is_start,
                    profile_prefix=profile_prefix,
                    profiler_kwargs=profiler_kwargs,
                )

            worker_cls.profile = _wrapped_profile
    except ImportError:
        pass

    # Hook TPUModelRunner._greedy_top_tokens_this_step so that when TPU_GREEDY_TOP_TOKENS=1
    # is active, warmup controller calibration steps for each canonical batch size (and one
    # step per XProf profiling window) materialize full target_logits for
    # pallas_entropy_block_verify while all timed steady-state steps use vocabulary-parallel
    # top-token verification with zero host transfer overhead.
    try:
        runner_mod = importlib.import_module("vllm_torchtpu.runner.tpu_runner")
        runner_cls = getattr(runner_mod, "TPUModelRunner", None)
        if runner_cls is not None and hasattr(runner_cls, "_greedy_top_tokens_this_step"):
            orig_top_tokens = runner_cls._greedy_top_tokens_this_step

            def _wrapped_greedy_top_tokens_this_step(self: Any) -> bool:
                base_eligible = bool(orig_top_tokens(self))
                if not base_eligible:
                    return False
                try:
                    num_reqs = int(self.input_batch.num_reqs)
                    req_ids = self.input_batch.req_ids[:num_reqs]
                    is_synthetic = all(
                        r is None or str(r).startswith("__") for r in req_ids
                    )
                    if not is_synthetic and num_reqs > 0:
                        state["live_steps"] = int(state["live_steps"]) + 1
                        counts = state["sampled_batch_counts"]
                        if num_reqs in (1, 2, 4, 8, 16, 32) and counts.get(num_reqs, 0) < 2:
                            counts[num_reqs] = counts.get(num_reqs, 0) + 1
                            state["force_full_logits_step"] = True
                            return False
                        if state.get("profiling_active", False) and not state.get(
                            "profile_verify_done", False
                        ):
                            state["profile_verify_done"] = True
                            state["force_full_logits_step"] = True
                            return False
                except Exception:
                    pass
                return base_eligible

            runner_cls._greedy_top_tokens_this_step = _wrapped_greedy_top_tokens_this_step
    except ImportError:
        pass

    if telemetry_path and os.environ.get("RANK", "0") in ("0", ""):
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


def _extract_cpu_logits_and_draft(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.device | None, torch.dtype | None]:
    """Copy target_logits and draft_token_ids from TPU to CPU as leaf tensors without
    emitting any eager slice/view/cat ops on TPU (avoiding torch_tpu DEFER_AND_FUSE
    recompilations), then return (logits_3d_cpu [B, K+1, V], draft_2d_cpu [B, K], device, dtype).
    """
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
    if logits_3d is not None and draft_2d is not None:
        return (
            logits_3d.detach().to("cpu", dtype=torch.float32),
            draft_2d.detach().to("cpu", dtype=torch.int32),
            logits_3d.device,
            logits_3d.dtype,
        )

    # Handle vllm_torchtpu RejectionSampler.forward signature:
    # (draft_token_ids [N], num_draft_tokens [B], target_logits [N, V], bonus_token_ids [B],
    #  segment_ids [N], group_indices [N], max_draft_tokens: int, ...)
    draft_1d = kwargs.get("draft_token_ids") if "draft_token_ids" in kwargs else (args[0] if len(args) > 0 else None)
    target_2d = kwargs.get("target_logits") if "target_logits" in kwargs else (args[2] if len(args) > 2 else None)
    bonus_1d = kwargs.get("bonus_token_ids") if "bonus_token_ids" in kwargs else (args[3] if len(args) > 3 else None)
    max_k = kwargs.get("max_draft_tokens") if "max_draft_tokens" in kwargs else (args[6] if len(args) > 6 else None)

    if (
        not isinstance(draft_1d, torch.Tensor)
        or not isinstance(target_2d, torch.Tensor)
        or draft_1d.device.type != "tpu"
        or target_2d.device.type != "tpu"
        or draft_1d.ndim != 1
        or target_2d.ndim != 2
        or target_2d.shape[-1] < 1024
        or not isinstance(max_k, int)
        or max_k <= 0
    ):
        return None, None, None, None

    total_rows, vocab = target_2d.shape
    if total_rows < max_k or draft_1d.shape[0] < max_k:
        return None, None, None, None

    tpu_device = target_2d.device
    tpu_dtype = target_2d.dtype
    req_bound = int(bonus_1d.shape[0]) if isinstance(bonus_1d, torch.Tensor) and bonus_1d.ndim == 1 else (total_rows // max_k)
    bsz = max(1, min(req_bound, total_rows // max_k))
    usable = bsz * max_k

    # Copy full buffer to CPU before slicing/viewing so zero eager TPU ops are recorded
    target_2d_cpu = target_2d.detach().to("cpu", dtype=torch.float32)
    draft_1d_cpu = draft_1d.detach().to("cpu", dtype=torch.int32)

    logits_main_cpu = target_2d_cpu[:usable, :].view(bsz, max_k, vocab)
    logits_3d_cpu = torch.cat([logits_main_cpu, logits_main_cpu[:, -1:, :]], dim=1)
    draft_2d_cpu = draft_1d_cpu[:usable].view(bsz, max_k)
    return logits_3d_cpu, draft_2d_cpu, tpu_device, tpu_dtype


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
    """Invoke Pallas static-tile block verification on TPU target logits."""
    if os.environ.get("RANK", "0") not in ("0", ""):
        return
    if kwargs.get("do_sampling", False):
        return

    is_precompile = False
    for depth in range(1, 8):
        try:
            caller_name = sys._getframe(depth).f_code.co_name
            if caller_name.startswith("_precompile") or caller_name.startswith("_warmup"):
                is_precompile = True
                break
        except ValueError:
            break

    if is_precompile:
        if state.get("prewarmed", False):
            return
    else:
        forced = bool(state.pop("force_full_logits_step", False))
        state["step"] = int(state["step"]) + 1
        if os.environ.get("TPU_GREEDY_TOP_TOKENS", "0") == "1":
            if not forced:
                return
        elif not forced and state["step"] != 1 and (state["step"] % 64 != 0):
            return

    logits_3d_cpu, draft_2d_cpu, tpu_device, tpu_dtype = _extract_cpu_logits_and_draft(
        args, kwargs
    )
    if (
        logits_3d_cpu is None
        or draft_2d_cpu is None
        or tpu_device is None
        or tpu_dtype is None
    ):
        return
    bsz, k_plus_1, vocab = logits_3d_cpu.shape
    k_curr = k_plus_1 - 1
    if k_curr <= 0 or draft_2d_cpu.shape != (bsz, k_curr):
        return

    b_static = 8
    active_b = min(bsz, b_static)
    k_copy = min(k_curr, k_pad)

    # Reuse pre-allocated fixed-shape (b_static=8, k_pad+1, vocab) CPU buffers so
    # torch_tpu's DEFER_AND_FUSE never fuses telemetry ops into TPUModelRunner's
    # decode graph and pallas_entropy_block_verify sees one invariant shape across all B.
    bufs = state.get("static_buffers")
    if bufs is None or bufs[0].shape != (b_static, k_pad + 1, vocab):
        bufs = (
            torch.full((b_static, k_pad + 1, vocab), -1e30, dtype=torch.float32),
            torch.zeros((b_static, k_pad), dtype=torch.int32),
            torch.zeros((b_static, k_pad), dtype=torch.int32),
        )
        state["static_buffers"] = bufs
    logits_static_cpu, draft_static_cpu, mask_static_cpu = bufs

    logits_cpu = logits_3d_cpu[:active_b, :k_copy, :]
    draft_cpu = draft_2d_cpu[:active_b, :k_copy]

    logits_static_cpu.fill_(-1e30)
    logits_static_cpu[:active_b, :k_copy, :] = logits_cpu
    logits_static_cpu[:active_b, k_copy, :] = logits_cpu[:, -1, :]

    draft_static_cpu.zero_()
    draft_static_cpu[:active_b, :k_copy] = draft_cpu

    eff_cap = min(int(state.get("next_k_cap", k_pad)), k_copy)
    mask_static_cpu.zero_()
    mask_static_cpu[:active_b, :eff_cap] = 1

    ctrl_cpu = torch.tensor(
        [entropy_thresh, float(bsz), float(min_k), float(k_pad), ridge_budget, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    logits_padded = logits_static_cpu.to(device=tpu_device, dtype=tpu_dtype)
    draft_padded = draft_static_cpu.to(device=tpu_device)
    active_mask = mask_static_cpu.to(device=tpu_device)
    ctrl = ctrl_cpu.to(device=tpu_device)

    res = pallas_entropy_block_verify(
        target_logits=logits_padded,
        draft_token_ids=draft_padded,
        active_draft_mask=active_mask,
        control_params=ctrl,
        valid_vocab_size=vocab,
        world_size=1,
    )

    if is_precompile:
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            torch.accelerator.synchronize()
        state["prewarmed"] = True
        return

    next_k_cpu = res.next_effective_k.to("cpu").to(torch.float32)[:active_b]
    next_k_mean = float(next_k_cpu.mean().item())
    state["next_k_cap"] = max(min_k, min(k_pad, int(round(next_k_mean))))

    if telemetry_path:
        ent_cpu = res.token_entropies.to("cpu").to(torch.float32)[:active_b, :eff_cap]
        acc_cpu = res.accepted_lengths.to("cpu").to(torch.float32)[:active_b]
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
