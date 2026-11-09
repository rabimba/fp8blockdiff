"""Entropy-Adaptive Static-Tile Speculative Decoding Controller.

Couples parallel block-diffusion drafters (DFlash / DSpark) with our static-shape
Pallas verification kernel (`pallas_entropy_block_verify`) so that:
1. Effective speculative block length K_eff(B, H) dynamically shrinks when batch
   size B approaches the TPU v7 FP8 ridge point (625.2 FLOPs/Byte) or when local
   token Shannon entropy H spikes ("suffix decay").
2. Tensor shapes passed to both drafter and target verification remain strictly
   fixed at `K_pad` (e.g. 32), masked via `next_draft_mask`, eliminating XLA
   recompilations across fluctuating dynamic acceptance regimes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from kernels.pallas_block_verify import (
    BlockVerifyOutputs,
    pallas_entropy_block_verify,
    pytorch_reference_block_verify,
)


@dataclasses.dataclass
class AdaptiveSpecControllerConfig:
    """Configuration for the static-tile entropy-adaptive controller."""

    padded_block_capacity_k: int = 32
    min_block_k: int = 4
    entropy_threshold_nats: float = 1.65
    tpu_v7_ridge_token_budget: float = 512.0
    ema_smoothing: float = 0.7
    world_size: int = 1


class EntropyAdaptiveBlockController:
    """Stateful static-shape controller maintaining per-slot draft masks on TPU."""

    def __init__(
        self,
        cfg: AdaptiveSpecControllerConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.device = device

        # Start with ridge-aware initial effective horizon
        init_k = min(
            cfg.padded_block_capacity_k,
            max(cfg.min_block_k, int(cfg.tpu_v7_ridge_token_budget // max(batch_size, 1)) - 1),
        )
        arange_k = torch.arange(cfg.padded_block_capacity_k, device=device, dtype=torch.int32)
        self.active_draft_mask = (arange_k.unsqueeze(0) < init_k).to(torch.int32).expand(
            batch_size, cfg.padded_block_capacity_k
        ).contiguous()

        self.control_params = torch.tensor(
            [
                cfg.entropy_threshold_nats,
                float(batch_size),
                float(cfg.min_block_k),
                float(cfg.padded_block_capacity_k),
                cfg.tpu_v7_ridge_token_budget,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        self.step_count: int = 0
        self.total_accepted_tokens: int = 0
        self.total_emitted_tokens: int = 0

    def update_thresholds(
        self,
        entropy_threshold_nats: float | None = None,
        active_batch_size: int | None = None,
    ) -> None:
        """Update control tensor values in-place without changing tensor shapes (zero recompile)."""
        vals = self.control_params.to("cpu").tolist()
        if entropy_threshold_nats is not None:
            vals[0] = float(entropy_threshold_nats)
        if active_batch_size is not None:
            vals[1] = float(active_batch_size)
        self.control_params.copy_(
            torch.tensor(vals, dtype=torch.float32, device=self.device)
        )

    def verify_step(
        self,
        target_logits: torch.Tensor,
        draft_token_ids: torch.Tensor,
        valid_vocab_size: int | None = None,
        use_pallas_kernel: bool = True,
    ) -> BlockVerifyOutputs:
        """Run one static-shape verification & next-step mask update on TPU."""
        if use_pallas_kernel:
            outputs = pallas_entropy_block_verify(
                target_logits=target_logits,
                draft_token_ids=draft_token_ids,
                active_draft_mask=self.active_draft_mask,
                control_params=self.control_params,
                valid_vocab_size=valid_vocab_size,
                world_size=self.cfg.world_size,
            )
        else:
            outputs = pytorch_reference_block_verify(
                target_logits=target_logits,
                draft_token_ids=draft_token_ids,
                active_draft_mask=self.active_draft_mask,
                control_params=self.control_params,
                valid_vocab_size=valid_vocab_size,
            )

        self.active_draft_mask = outputs.next_draft_mask
        self.step_count += 1
        acc_sum = int(outputs.accepted_lengths.to("cpu").sum().item())
        self.total_accepted_tokens += acc_sum
        self.total_emitted_tokens += acc_sum + self.batch_size
        return outputs

    def summary_stats(self) -> dict[str, Any]:
        denom = max(self.step_count * self.batch_size, 1)
        return {
            "steps": self.step_count,
            "mean_accepted_draft_tokens": round(self.total_accepted_tokens / denom, 4),
            "mean_emitted_tokens_per_step": round(self.total_emitted_tokens / denom, 4),
        }
