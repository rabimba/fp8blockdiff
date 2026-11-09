#!/usr/bin/env python3
"""Verify and cache target FP8/BF16 models and speculative drafter checkpoints
(DFlash block-diffusion and EAGLE-3) from Hugging Face Hub.

Writes exact resolved checkpoint IDs, local cache paths, and architecture
metadata to artifacts/logs/model_manifest.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def inspect_remote_model_config(api: HfApi, repo_id: str) -> dict[str, Any] | None:
    """Verify repository exists on Hugging Face Hub and inspect its config.json."""
    try:
        api.model_info(repo_id)
        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        return json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[models] Candidate '{repo_id}' unavailable or missing config.json: {exc}")
        return None


def stage_snapshot(repo_id: str, metadata_only: bool = False) -> str:
    allow_patterns = ["config.json", "tokenizer*", "generation_config.json", "*.json"]
    if metadata_only:
        return snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)
    return snapshot_download(
        repo_id=repo_id,
        ignore_patterns=["*.msgpack", "*.h5", "original/*"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify & stage FP8 target + drafter models")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "models.yaml",
        help="Path to configs/models.yaml",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Override active_profile in models.yaml",
    )
    parser.add_argument(
        "--include-bf16-contrast",
        action="store_true",
        help="Also pre-stage the full BF16 target checkpoint for Phase 3 roofline contrast",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Verify remote Hub configs and tokenizers without downloading full safetensors",
    )
    args = parser.parse_args()

    cfg_raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    profile_name = args.profile or cfg_raw["active_profile"]
    profile = cfg_raw["profiles"][profile_name]
    expected_vocab = int(profile["expected_vocab_size"])

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    # 1. Verify & stage primary FP8 target model
    target_fp8_id = profile["target_fp8"]
    target_fp8_cfg = inspect_remote_model_config(api, target_fp8_id)
    if target_fp8_cfg is None:
        raise RuntimeError(f"Primary FP8 target model '{target_fp8_id}' could not be verified on HF Hub.")

    target_vocab = int(
        target_fp8_cfg.get("vocab_size")
        or target_fp8_cfg.get("text_config", {}).get("vocab_size", expected_vocab)
    )
    if target_vocab != expected_vocab:
        print(
            f"[models] Note: {target_fp8_id} vocab_size={target_vocab} "
            f"(expected {expected_vocab}); updating expected_vocab."
        )
        expected_vocab = target_vocab

    print(f"[models] Staging FP8 target checkpoint: {target_fp8_id} ...")
    target_fp8_local = stage_snapshot(target_fp8_id, metadata_only=args.metadata_only)

    # 2. Verify optional BF16 contrast checkpoint
    target_bf16_id = profile["target_bf16"]
    target_bf16_cfg = inspect_remote_model_config(api, target_bf16_id)
    target_bf16_local: str | None = None
    if target_bf16_cfg is not None and args.include_bf16_contrast:
        print(f"[models] Staging BF16 contrast checkpoint: {target_bf16_id} ...")
        target_bf16_local = stage_snapshot(target_bf16_id, metadata_only=args.metadata_only)

    # 3. Resolve & stage DFlash block-diffusion drafter checkpoint
    resolved_dflash: dict[str, Any] | None = None
    for cand in profile.get("dflash_candidates", []):
        cand_cfg = inspect_remote_model_config(api, cand)
        if cand_cfg is not None:
            local_dir = stage_snapshot(cand, metadata_only=args.metadata_only)
            resolved_dflash = {
                "repo_id": cand,
                "local_path": local_dir,
                "architectures": cand_cfg.get("architectures", []),
                "vocab_size": cand_cfg.get("vocab_size", expected_vocab),
                "block_size": cand_cfg.get("block_size") or cand_cfg.get("dflash_block_size", 16),
            }
            print(f"[models] Resolved & staged DFlash drafter: {cand} -> {local_dir}")
            break

    if resolved_dflash is None:
        raise RuntimeError(
            f"None of the DFlash drafter candidates {profile.get('dflash_candidates')} "
            "could be verified on Hugging Face Hub."
        )

    # 4. Resolve & stage EAGLE-3 drafter checkpoint (or record fallback method if absent)
    resolved_eagle3: dict[str, Any] | None = None
    for cand in profile.get("eagle3_candidates", []):
        cand_cfg = inspect_remote_model_config(api, cand)
        if cand_cfg is not None:
            local_dir = stage_snapshot(cand, metadata_only=args.metadata_only)
            resolved_eagle3 = {
                "repo_id": cand,
                "local_path": local_dir,
                "architectures": cand_cfg.get("architectures", []),
                "vocab_size": cand_cfg.get("vocab_size", expected_vocab),
            }
            print(f"[models] Resolved & staged EAGLE-3 drafter: {cand} -> {local_dir}")
            break

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "metadata_only": args.metadata_only,
        "target_fp8": {
            "repo_id": target_fp8_id,
            "local_path": target_fp8_local,
            "architectures": target_fp8_cfg.get("architectures", []),
            "vocab_size": expected_vocab,
            "quantization_config": target_fp8_cfg.get("quantization_config"),
        },
        "target_bf16": {
            "repo_id": target_bf16_id,
            "verified_remote": target_bf16_cfg is not None,
            "local_path": target_bf16_local,
        },
        "dflash_drafter": resolved_dflash,
        "eagle3_drafter": resolved_eagle3,
        "fallback_spec_method": profile.get("fallback_spec_method", "ngram"),
    }

    out_path = REPO_ROOT / "artifacts" / "logs" / "model_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[models] Wrote verified model manifest to {out_path}")


if __name__ == "__main__":
    main()
