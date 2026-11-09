"""Static-Shape Systolic-Tile-Aligned Block Verification & Entropy-Adaptive Masking Kernel.

Core systems & research contribution for Cloud TPU v7 (Ironwood / v7x):
1. Keeps all verification tensors strictly padded to TPU VPU/MXU tile multiples
   (rows aligned to 128, vocabulary tiles aligned to 1024 = 8 * 128) so XLA
   compiles each (B_pad, K_pad, V_pad) bucket ONCE and NEVER recompiles when
   dynamic acceptance lengths or entropy thresholds fluctuate at runtime.
2. Streams vocabulary logits through VMEM in tiles using a single-pass,
   shift-invariant online log-sum-exp + Shannon entropy + greedy argmax recurrence
   in FP32 vector registers, eliminating [B * (K + 1), V] probability HBM writebacks.
3. Wraps the Pallas TPU kernel natively for PyTorch & `torch.compile(backend="tpu")`
   using `torch.tpu.pallas.jax_op` with explicit `mesh` and `PartitionSpec` metadata
   inspected dynamically via `inspect.signature`.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Callable

import numpy as np
import torch


SYSTOLIC_ROW_TILE = 128
VOCAB_VMEM_TILE = 1024  # Exact multiple of 128 lanes


def align_up(value: int, multiple: int) -> int:
    """Round positive integer `value` up to the nearest multiple of `multiple`."""
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


@dataclasses.dataclass(frozen=True)
class BlockVerifyOutputs:
    """Structured outputs of static-shape block verification and entropy scheduling."""

    accepted_lengths: torch.Tensor      # [B] int32: accepted contiguous prefix length in [0, K_pad]
    bonus_token_ids: torch.Tensor       # [B] int32: lossless target token at position accepted_lengths[b]
    token_entropies: torch.Tensor       # [B, K_pad + 1] float32: exact Shannon entropy (nats)
    entropy_tail_mask: torch.Tensor     # [B, K_pad] int32: 1 where active & H <= threshold, else 0
    next_effective_k: torch.Tensor      # [B] int32: adaptive draft block horizon for next step
    next_draft_mask: torch.Tensor       # [B, K_pad] int32: static-shape binary mask for next step


def tiled_online_entropy_and_argmax_numpy(
    logits: np.ndarray,
    valid_vocab_size: int,
    vocab_tile: int = VOCAB_VMEM_TILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Host-verifiable NumPy implementation of our exact VMEM tiled online recurrence.

    Computes exact Shannon entropy H = log(l) - s / l and tie-consistent argmax
    across vocabulary tiles without materializing full-vocab softmax probabilities.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D [M, V_pad] logits, got {logits.shape}")
    m_rows, v_pad = logits.shape
    if v_pad % vocab_tile != 0:
        raise ValueError(f"v_pad ({v_pad}) must be divisible by vocab_tile ({vocab_tile})")

    m_acc = np.full((m_rows, 1), -np.inf, dtype=np.float32)
    l_acc = np.zeros((m_rows, 1), dtype=np.float32)
    s_acc = np.zeros((m_rows, 1), dtype=np.float32)
    best_val = np.full((m_rows, 1), -np.inf, dtype=np.float32)
    best_idx = np.zeros((m_rows, 1), dtype=np.int32)

    for v_start in range(0, v_pad, vocab_tile):
        tile = logits[:, v_start : v_start + vocab_tile].astype(np.float32)
        cols = np.arange(v_start, v_start + vocab_tile, dtype=np.int32)[None, :]
        valid = cols < valid_vocab_size
        masked_tile = np.where(valid, tile, -1e30).astype(np.float32)

        m_curr = np.max(masked_tile, axis=1, keepdims=True)
        shifted = masked_tile - m_curr
        exp_curr = np.where(valid, np.exp(shifted), 0.0).astype(np.float32)
        l_curr = np.sum(exp_curr, axis=1, keepdims=True)
        s_curr = np.sum(np.where(valid, exp_curr * shifted, 0.0), axis=1, keepdims=True)

        m_new = np.maximum(m_acc, m_curr)
        dm_old = np.where(np.isfinite(m_acc) & np.isfinite(m_new), m_acc - m_new, 0.0).astype(
            np.float32
        )
        dm_curr = np.where(np.isfinite(m_curr) & np.isfinite(m_new), m_curr - m_new, 0.0).astype(
            np.float32
        )
        alpha = np.where(np.isfinite(m_acc), np.exp(dm_old), 0.0).astype(np.float32)
        beta = np.where(l_curr > 0.0, np.exp(dm_curr), 0.0).astype(np.float32)

        s_old_term = np.where(
            l_acc > 0.0,
            alpha * s_acc + alpha * l_acc * dm_old,
            0.0,
        )
        s_curr_term = np.where(
            l_curr > 0.0,
            beta * s_curr + beta * l_curr * dm_curr,
            0.0,
        )
        l_acc = alpha * l_acc + beta * l_curr
        s_acc = s_old_term + s_curr_term
        m_acc = m_new

        tile_best_val = np.max(masked_tile, axis=1, keepdims=True)
        tile_best_idx = np.argmax(masked_tile, axis=1, keepdims=True).astype(np.int32) + v_start
        take_curr = tile_best_val > best_val
        best_idx = np.where(take_curr, tile_best_idx, best_idx)
        best_val = np.maximum(best_val, tile_best_val)

    entropy = np.log(np.maximum(l_acc, 1e-30)) - (s_acc / np.maximum(l_acc, 1e-30))
    return entropy.squeeze(1).astype(np.float32), best_idx.squeeze(1).astype(np.int32)


def pytorch_reference_block_verify(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    active_draft_mask: torch.Tensor,
    control_params: torch.Tensor,
    valid_vocab_size: int | None = None,
) -> BlockVerifyOutputs:
    """Pure PyTorch eager reference implementation (compatible with CPU and TPU).

    Args:
        target_logits: [B, K_pad + 1, V] float32 or bfloat16 logits from target model.
        draft_token_ids: [B, K_pad] int32 candidate tokens proposed by drafter.
        active_draft_mask: [B, K_pad] int32 mask (1 for active positions, 0 for padded tail).
        control_params: [8] float32 tensor:
            [0] entropy_threshold (nats)
            [1] active_batch_size (B)
            [2] min_k
            [3] max_k (<= K_pad)
            [4] ridge_token_budget (e.g. 512.0 on TPU v7 FP8)
        valid_vocab_size: Optional unpadded vocabulary count <= V.
    """
    bsz, k_plus_1, vocab_dim = target_logits.shape
    k_pad = draft_token_ids.shape[1]
    if k_plus_1 != k_pad + 1:
        raise ValueError(
            f"target_logits second dim ({k_plus_1}) must equal K_pad + 1 ({k_pad + 1})"
        )

    v_valid = valid_vocab_size if valid_vocab_size is not None else vocab_dim
    logits_f32 = target_logits[:, :, :v_valid].to(torch.float32)

    # Numerically stable Shannon entropy in nats: H = -sum(p * log_p)
    log_probs = torch.log_softmax(logits_f32, dim=-1)
    probs = torch.exp(log_probs)
    token_entropies = -torch.sum(probs * log_probs, dim=-1)  # [B, K_pad + 1]

    greedy_target = torch.argmax(logits_f32, dim=-1).to(torch.int32)  # [B, K_pad + 1]
    draft_i32 = draft_token_ids.to(torch.int32)
    mask_i32 = (active_draft_mask > 0).to(torch.int32)

    matches = ((draft_i32 == greedy_target[:, :k_pad]).to(torch.int32)) * mask_i32
    prefix_valid = torch.cumprod(matches, dim=1)
    accepted_lengths = torch.sum(prefix_valid, dim=1).to(torch.int32)  # [B]

    gather_idx = accepted_lengths.to(torch.int64).unsqueeze(1)
    bonus_token_ids = torch.gather(greedy_target, dim=1, index=gather_idx).squeeze(1).to(torch.int32)

    entropy_thresh = control_params[0].to(torch.float32)
    active_b = torch.clamp(control_params[1].to(torch.float32), min=1.0)
    min_k = control_params[2].to(torch.float32)
    max_k = torch.clamp(control_params[3].to(torch.float32), max=float(k_pad))
    ridge_budget = control_params[4].to(torch.float32)

    draft_entropies = token_entropies[:, :k_pad]
    entropy_tail_mask = ((draft_entropies <= entropy_thresh).to(torch.int32)) * mask_i32

    # Compute mean entropy across active positions (including bonus position 0 fallback)
    active_counts = torch.clamp(torch.sum(mask_i32, dim=1).to(torch.float32), min=1.0)
    mean_entropy = torch.sum(draft_entropies * mask_i32.to(torch.float32), dim=1) / active_counts

    # Smooth exponential horizon decay under high entropy + systolic ridge-point cap
    safe_thresh = torch.clamp(entropy_thresh, min=1e-3)
    excess_h = torch.clamp(mean_entropy - 0.5 * safe_thresh, min=0.0)
    k_entropy = torch.round(max_k * torch.exp(-excess_h / safe_thresh))
    k_entropy = torch.clamp(k_entropy, min=min_k, max=max_k)

    k_ridge = torch.clamp(torch.floor(ridge_budget / active_b) - 1.0, min=min_k, max=max_k)
    next_k_f32 = torch.minimum(k_entropy, k_ridge)
    next_effective_k = next_k_f32.to(torch.int32)

    arange_k = torch.arange(k_pad, device=target_logits.device, dtype=torch.int32).unsqueeze(0)
    next_draft_mask = (arange_k < next_effective_k.unsqueeze(1)).to(torch.int32)

    return BlockVerifyOutputs(
        accepted_lengths=accepted_lengths,
        bonus_token_ids=bonus_token_ids,
        token_entropies=token_entropies,
        entropy_tail_mask=entropy_tail_mask,
        next_effective_k=next_effective_k,
        next_draft_mask=next_draft_mask,
    )


def _select_vocab_tile_size(v_pad: int) -> int:
    """Query openxla/tokamax metadata if available while enforcing TPU 128-lane alignment."""
    preferred = VOCAB_VMEM_TILE
    try:
        import tokamax  # noqa: F401

        # Keep VMEM footprint well below TPU v7 scoped VMEM limit (~16-32 MiB):
        # 128 * 1024 * 4B = 512 KiB per tile buffer.
        if v_pad % 2048 == 0:
            preferred = 2048
    except ImportError:
        pass
    if v_pad % preferred != 0:
        return VOCAB_VMEM_TILE
    return preferred


@functools.lru_cache(maxsize=16)
def _build_cached_jax_op(
    m_pad: int,
    v_pad: int,
    valid_vocab_size: int,
    b_pad: int,
    k_pad: int,
    world_size: int,
) -> Callable[..., Any]:
    """Build and cache a static-shape Pallas + jax_op callable per padded tile bucket."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    vocab_tile = _select_vocab_tile_size(v_pad)
    num_m_tiles = m_pad // SYSTOLIC_ROW_TILE
    num_v_tiles = v_pad // vocab_tile

    def _pallas_online_entropy_argmax_kernel(
        logits_ref: Any,
        valid_v_ref: Any,
        out_entropy_ref: Any,
        out_argmax_ref: Any,
        m_scratch_ref: Any,
        l_scratch_ref: Any,
        s_scratch_ref: Any,
        bval_scratch_ref: Any,
        bidx_scratch_ref: Any,
    ) -> None:
        v_idx = pl.program_id(1)
        v_start = v_idx * vocab_tile
        valid_v = valid_v_ref[0, 0]

        @pl.when(v_idx == 0)
        def _init_accumulators() -> None:
            m_scratch_ref[...] = jnp.full((SYSTOLIC_ROW_TILE, 128), -jnp.inf, dtype=jnp.float32)
            l_scratch_ref[...] = jnp.zeros((SYSTOLIC_ROW_TILE, 128), dtype=jnp.float32)
            s_scratch_ref[...] = jnp.zeros((SYSTOLIC_ROW_TILE, 128), dtype=jnp.float32)
            bval_scratch_ref[...] = jnp.full((SYSTOLIC_ROW_TILE, 128), -jnp.inf, dtype=jnp.float32)
            bidx_scratch_ref[...] = jnp.zeros((SYSTOLIC_ROW_TILE, 128), dtype=jnp.int32)

        tile = logits_ref[...].astype(jnp.float32)
        col_ids = v_start + jnp.arange(vocab_tile, dtype=jnp.int32)[None, :]
        valid_cols = col_ids < valid_v
        masked_tile = jnp.where(valid_cols, tile, -1e30)

        m_curr_1d = jnp.max(masked_tile, axis=1, keepdims=True)
        shifted = masked_tile - m_curr_1d
        exp_curr = jnp.where(valid_cols, jnp.exp(shifted), 0.0)
        l_curr_1d = jnp.sum(exp_curr, axis=1, keepdims=True)
        s_curr_1d = jnp.sum(jnp.where(valid_cols, exp_curr * shifted, 0.0), axis=1, keepdims=True)

        # Broadcast [128, 1] reductions across [128, 128] VMEM register tile
        m_curr = jnp.broadcast_to(m_curr_1d, (SYSTOLIC_ROW_TILE, 128))
        l_curr = jnp.broadcast_to(l_curr_1d, (SYSTOLIC_ROW_TILE, 128))
        s_curr = jnp.broadcast_to(s_curr_1d, (SYSTOLIC_ROW_TILE, 128))

        m_old = m_scratch_ref[...]
        l_old = l_scratch_ref[...]
        s_old = s_scratch_ref[...]

        m_new = jnp.maximum(m_old, m_curr)
        dm_old = jnp.where(jnp.isfinite(m_old) & jnp.isfinite(m_new), m_old - m_new, 0.0)
        dm_curr = jnp.where(jnp.isfinite(m_curr) & jnp.isfinite(m_new), m_curr - m_new, 0.0)
        alpha = jnp.where(jnp.isfinite(m_old), jnp.exp(dm_old), 0.0)
        beta = jnp.where(l_curr > 0.0, jnp.exp(dm_curr), 0.0)

        s_old_term = jnp.where(l_old > 0.0, alpha * s_old + alpha * l_old * dm_old, 0.0)
        s_curr_term = jnp.where(
            l_curr > 0.0, beta * s_curr + beta * l_curr * dm_curr, 0.0
        )

        l_new = alpha * l_old + beta * l_curr
        s_new = s_old_term + s_curr_term

        m_scratch_ref[...] = m_new
        l_scratch_ref[...] = l_new
        s_scratch_ref[...] = s_new

        tile_bval_1d = jnp.max(masked_tile, axis=1, keepdims=True)
        tile_bidx_1d = jnp.argmax(masked_tile, axis=1).reshape((SYSTOLIC_ROW_TILE, 1)).astype(
            jnp.int32
        ) + v_start
        tile_bval = jnp.broadcast_to(tile_bval_1d, (SYSTOLIC_ROW_TILE, 128))
        tile_bidx = jnp.broadcast_to(tile_bidx_1d, (SYSTOLIC_ROW_TILE, 128))

        prev_bval = bval_scratch_ref[...]
        prev_bidx = bidx_scratch_ref[...]
        take_curr = tile_bval > prev_bval
        bidx_scratch_ref[...] = jnp.where(take_curr, tile_bidx, prev_bidx)
        bval_scratch_ref[...] = jnp.maximum(prev_bval, tile_bval)

        @pl.when(v_idx == num_v_tiles - 1)
        def _write_final_tile() -> None:
            safe_l = jnp.maximum(l_scratch_ref[...], 1e-30)
            out_entropy_ref[...] = jnp.log(safe_l) - (s_scratch_ref[...] / safe_l)
            out_argmax_ref[...] = bidx_scratch_ref[...]

    def _jax_static_block_verify_fn(
        padded_logits_2d: jax.Array,
        draft_ids_2d: jax.Array,
        active_mask_2d: jax.Array,
        control_vec_2d: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        valid_v_mat = jnp.full((8, 128), valid_vocab_size, dtype=jnp.int32)

        out_shapes = (
            jax.ShapeDtypeStruct((m_pad, 128), jnp.float32),
            jax.ShapeDtypeStruct((m_pad, 128), jnp.int32),
        )
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (SYSTOLIC_ROW_TILE, vocab_tile),
                    lambda m_i, v_i: (m_i, v_i),
                ),
                pl.BlockSpec((8, 128), lambda m_i, v_i: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((SYSTOLIC_ROW_TILE, 128), lambda m_i, v_i: (m_i, 0)),
                pl.BlockSpec((SYSTOLIC_ROW_TILE, 128), lambda m_i, v_i: (m_i, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((SYSTOLIC_ROW_TILE, 128), jnp.float32),
                pltpu.VMEM((SYSTOLIC_ROW_TILE, 128), jnp.float32),
                pltpu.VMEM((SYSTOLIC_ROW_TILE, 128), jnp.float32),
                pltpu.VMEM((SYSTOLIC_ROW_TILE, 128), jnp.float32),
                pltpu.VMEM((SYSTOLIC_ROW_TILE, 128), jnp.int32),
            ],
            grid=(num_m_tiles, num_v_tiles),
            dimension_semantics=("parallel", "arbitrary"),
        )

        entropy_2d, argmax_2d = pl.pallas_call(
            _pallas_online_entropy_argmax_kernel,
            out_shape=out_shapes,
            grid_spec=grid_spec,
        )(padded_logits_2d, valid_v_mat)

        # Slice unpadded batch * (k_pad + 1) rows from static tile buffer
        total_active_rows = b_pad * (k_pad + 1)
        token_entropies = entropy_2d[:total_active_rows, 0].reshape((b_pad, k_pad + 1))
        greedy_target = argmax_2d[:total_active_rows, 0].reshape((b_pad, k_pad + 1))

        mask_i32 = (active_mask_2d > 0).astype(jnp.int32)
        matches = ((draft_ids_2d == greedy_target[:, :k_pad]).astype(jnp.int32)) * mask_i32
        prefix_valid = jnp.cumprod(matches, axis=1)
        accepted_lengths = jnp.sum(prefix_valid, axis=1).astype(jnp.int32)

        batch_idx = jnp.arange(b_pad, dtype=jnp.int32)
        bonus_tokens = greedy_target[batch_idx, accepted_lengths].astype(jnp.int32)

        entropy_thresh = control_vec_2d[0, 0]
        active_b = jnp.maximum(control_vec_2d[0, 1], 1.0)
        min_k = control_vec_2d[0, 2]
        max_k = jnp.minimum(control_vec_2d[0, 3], float(k_pad))
        ridge_budget = control_vec_2d[0, 4]

        draft_entropies = token_entropies[:, :k_pad]
        entropy_tail_mask = ((draft_entropies <= entropy_thresh).astype(jnp.int32)) * mask_i32

        active_counts = jnp.maximum(jnp.sum(mask_i32, axis=1).astype(jnp.float32), 1.0)
        mean_entropy = jnp.sum(draft_entropies * mask_i32.astype(jnp.float32), axis=1) / active_counts

        safe_thresh = jnp.maximum(entropy_thresh, 1e-3)
        excess_h = jnp.maximum(mean_entropy - 0.5 * safe_thresh, 0.0)
        k_entropy = jnp.clip(jnp.round(max_k * jnp.exp(-excess_h / safe_thresh)), min_k, max_k)
        k_ridge = jnp.clip(jnp.floor(ridge_budget / active_b) - 1.0, min_k, max_k)
        next_effective_k = jnp.minimum(k_entropy, k_ridge).astype(jnp.int32)

        arange_k = jnp.arange(k_pad, dtype=jnp.int32)[None, :]
        next_draft_mask = (arange_k < next_effective_k[:, None]).astype(jnp.int32)

        return (
            accepted_lengths,
            bonus_tokens,
            token_entropies,
            entropy_tail_mask,
            next_effective_k,
            next_draft_mask,
        )

    return _bind_torch_tpu_jax_op(_jax_static_block_verify_fn, world_size=world_size)


def _bind_torch_tpu_jax_op(jax_fn: Callable[..., Any], world_size: int) -> Callable[..., Any]:
    """Bind `jax_fn` via `torch.tpu.pallas.jax_op` using live signature introspection."""
    import jax
    import torch_tpu  # noqa: F401

    pallas_bridge = getattr(getattr(torch, "tpu", None), "pallas", None)
    if pallas_bridge is None or not hasattr(pallas_bridge, "jax_op"):
        raise RuntimeError(
            "torch.tpu.pallas.jax_op is not available on this TorchTPU installation. "
            "Run scripts/phase0_discover.py and inspect docs/api_discovery.md."
        )

    P = jax.sharding.PartitionSpec
    if hasattr(pallas_bridge, "get_pallas_mesh"):
        mesh_sig = inspect.signature(pallas_bridge.get_pallas_mesh)
        if "axis_names" in mesh_sig.parameters and "mesh_shape" in mesh_sig.parameters:
            mesh = pallas_bridge.get_pallas_mesh(axis_names=("tp",), mesh_shape=(world_size,))
        else:
            mesh = jax.sharding.Mesh(np.array(jax.devices()[:world_size]), ("tp",))
    else:
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:world_size]), ("tp",))

    # Replicate verification inputs across single-device or post-gather TP mesh
    in_specs = [P(None, None), P(None, None), P(None, None), P(None, None)]
    out_specs = [P(None), P(None), P(None, None), P(None, None), P(None), P(None, None)]

    jax_op_fn = pallas_bridge.jax_op
    sig = inspect.signature(jax_op_fn)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "name" in params:
        kwargs["name"] = (
            f"fp8blockdiff::pallas_block_verify_{abs(hash(jax_fn)) % 10_000_000}"
        )
    if "mesh" in params:
        kwargs["mesh"] = mesh
    if "input_partition_specs" in params:
        kwargs["input_partition_specs"] = in_specs
    elif "in_specs" in params:
        kwargs["in_specs"] = in_specs
    if "output_partition_specs" in params:
        kwargs["output_partition_specs"] = out_specs
    elif "out_specs" in params:
        kwargs["out_specs"] = out_specs

    if "op" in params:
        kwargs["op"] = jax_fn
        return jax_op_fn(**kwargs)
    if "fn" in params:
        kwargs["fn"] = jax_fn
        return jax_op_fn(**kwargs)
    return jax_op_fn(jax_fn, **kwargs)


def pallas_entropy_block_verify(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    active_draft_mask: torch.Tensor,
    control_params: torch.Tensor,
    valid_vocab_size: int | None = None,
    world_size: int = 1,
) -> BlockVerifyOutputs:
    """Execute static-shape Pallas TPU block verification via `torch.tpu.pallas.jax_op`.

    Pads flattened batch-block token rows to multiples of 128 and vocabulary columns
    to multiples of 1024 so XLA never recompiles across dynamic acceptance lengths
    or runtime entropy threshold updates.
    """
    if target_logits.device.type != "tpu":
        raise RuntimeError(
            f"pallas_entropy_block_verify requires tensors on torch.device('tpu'), "
            f"got {target_logits.device}. Never fall back to CPU."
        )

    bsz, k_plus_1, vocab_dim = target_logits.shape
    k_pad = draft_token_ids.shape[1]
    if k_plus_1 != k_pad + 1:
        raise ValueError(f"Expected target_logits.shape[1] == {k_pad + 1}, got {k_plus_1}")

    v_valid = valid_vocab_size if valid_vocab_size is not None else vocab_dim
    b_pad = align_up(bsz, 8)
    m_active = b_pad * k_plus_1
    m_pad = align_up(m_active, SYSTOLIC_ROW_TILE)
    v_pad = align_up(vocab_dim, _select_vocab_tile_size(align_up(vocab_dim, VOCAB_VMEM_TILE)))

    # Pad batch -> b_pad, flattened rows -> m_pad, vocab -> v_pad with static shapes
    logits_flat = target_logits.reshape(bsz * k_plus_1, vocab_dim)
    row_pad = m_pad - (bsz * k_plus_1)
    col_pad = v_pad - vocab_dim
    if row_pad > 0 or col_pad > 0:
        padded_logits_2d = torch.nn.functional.pad(
            logits_flat, (0, col_pad, 0, row_pad), value=-1e30
        )
    else:
        padded_logits_2d = logits_flat

    b_extra = b_pad - bsz
    if b_extra > 0:
        draft_padded = torch.nn.functional.pad(draft_token_ids.to(torch.int32), (0, 0, 0, b_extra))
        mask_padded = torch.nn.functional.pad(
            active_draft_mask.to(torch.int32), (0, 0, 0, b_extra)
        )
    else:
        draft_padded = draft_token_ids.to(torch.int32)
        mask_padded = active_draft_mask.to(torch.int32)

    ctrl_flat = control_params.to(torch.float32).flatten()
    ctrl_2d = torch.nn.functional.pad(ctrl_flat, (0, 128 - ctrl_flat.numel())).unsqueeze(0)
    ctrl_2d = ctrl_2d.expand(8, 128).contiguous()

    custom_op = _build_cached_jax_op(
        m_pad=m_pad,
        v_pad=v_pad,
        valid_vocab_size=v_valid,
        b_pad=b_pad,
        k_pad=k_pad,
        world_size=world_size,
    )

    acc_len, bonus_tok, tok_ent, ent_mask, next_k, next_mask = custom_op(
        padded_logits_2d,
        draft_padded,
        mask_padded,
        ctrl_2d,
    )

    return BlockVerifyOutputs(
        accepted_lengths=acc_len[:bsz],
        bonus_token_ids=bonus_tok[:bsz],
        token_entropies=tok_ent[:bsz],
        entropy_tail_mask=ent_mask[:bsz],
        next_effective_k=next_k[:bsz],
        next_draft_mask=next_mask[:bsz],
    )
