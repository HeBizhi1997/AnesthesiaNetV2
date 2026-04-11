"""
train.py — Training entry point.

Usage:
    # Standard training (v1 config, 10-second sequences, no TBPTT)
    python scripts/train.py --config configs/pipeline_v1.yaml

    # TBPTT training (v2 config, 5-minute chunks, hidden-state propagation)
    python scripts/train.py --config configs/pipeline_v2.yaml --tbptt

    # Resume from checkpoint
    python scripts/train.py --config configs/pipeline_v2.yaml --tbptt \\
                            --resume outputs/checkpoints/best_model.pt

Other flags:
    --data            path to HDF5 dataset
    --checkpoint_dir  where to save checkpoints
    --no-amp          disable AMP/BF16
    --compile         force torch.compile (requires Triton, not on Windows by default)
"""

from __future__ import annotations
import argparse
import os
import sys
import datetime
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, tag: str = "INFO") -> None:
    print(f"[{_ts()}][{tag}] {msg}", flush=True)

import gc
import random
import yaml
import h5py
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import build_datasets, SequenceDataset
from src.models.anesthesia_net import AnesthesiaNet
from src.models.anesthesia_net_v2 import AnesthesiaNetV2
from src.training.trainer import Trainer
from src.training.trainer_v2 import TrainerV2
from src.training.tbptt_trainer import TBPTTTrainer, PatientStore


def _patient_split(h5_path: str, val_split: float, test_split: float,
                   seed: int):
    """Reproduce the exact patient-level split from build_datasets()."""
    with h5py.File(h5_path, "r") as f:
        all_cases = sorted(f.keys())
    rng = random.Random(seed)
    rng.shuffle(all_cases)
    n = len(all_cases)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    test_ids  = all_cases[:n_test]
    val_ids   = all_cases[n_test:n_test + n_val]
    train_ids = all_cases[n_test + n_val:]
    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="configs/pipeline_v1.yaml")
    parser.add_argument("--data",           default="outputs/preprocessed/dataset.h5")
    parser.add_argument("--checkpoint_dir", default="outputs/checkpoints")
    parser.add_argument("--resume",         default=None)
    parser.add_argument("--tbptt",          action="store_true",
                        help="Use TBPTT training (requires tbptt:true in config or this flag)")
    parser.add_argument("--no-compile",     action="store_true")
    parser.add_argument("--compile",        action="store_true",
                        help="Force torch.compile even on Windows (requires Triton)")
    parser.add_argument("--no-amp",         action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # If --checkpoint_dir not explicitly provided, fall back to cfg paths value
    if args.checkpoint_dir == "outputs/checkpoints":
        args.checkpoint_dir = cfg.get("paths", {}).get("checkpoints", args.checkpoint_dir)

    torch.manual_seed(cfg["training"]["seed"])

    use_tbptt = args.tbptt or cfg["training"].get("tbptt", False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_version = cfg["training"].get("model_version", "v1")
    if model_version == "v2":
        model = AnesthesiaNetV2.from_config(cfg)
    else:
        model = AnesthesiaNet.from_config(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model: AnesthesiaNet{model_version.upper()}  parameters: {n_params:,}  "
         f"config={args.config}", "MODEL")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # ── torch.compile ─────────────────────────────────────────────────────────
    import platform
    on_windows = platform.system() == "Windows"
    use_compile = (not args.no_compile
                   and (not on_windows or args.compile)
                   and hasattr(torch, "compile")
                   and torch.cuda.is_available())
    if on_windows and not args.compile and not args.no_compile:
        print("torch.compile disabled (Triton not supported on Windows)")
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile failed ({e}), falling back to eager mode")

    # ── TBPTT path ────────────────────────────────────────────────────────────
    if use_tbptt:
        print(f"\n{'='*60}")
        print("  Mode: TBPTT (hidden-state propagation across 5-min chunks)")
        print(f"  seq_len={cfg['training']['seq_len']}  "
              f"batch_patients={cfg['training']['batch_size']}")
        print(f"{'='*60}\n")

        train_ids, val_ids, _ = _patient_split(
            args.data,
            cfg["training"]["val_split"],
            cfg["training"]["test_split"],
            cfg["training"]["seed"],
        )
        print(f"Patient split — train: {len(train_ids)}, val: {len(val_ids)}")

        # Load training patients into RAM
        print("Loading training patients into RAM …")
        train_store = PatientStore(args.data, train_ids, verbose=True)

        # Validation: standard SequenceDataset (seq_len=10 for consistent metric)
        val_seq_len = min(cfg["training"]["seq_len"], 10)
        _, val_ds, _ = build_datasets(
            h5_path=args.data,
            val_split=cfg["training"]["val_split"],
            test_split=cfg["training"]["test_split"],
            seq_len=val_seq_len,
            seed=cfg["training"]["seed"],
            noise_std=0.0,
            cache_in_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"] * 8,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
        )
        print(f"Validation: {len(val_ds):,} sequences (seq_len={val_seq_len})")

        trainer = TBPTTTrainer(
            model=model,
            train_store=train_store,
            val_loader=val_loader,
            cfg=cfg,
            checkpoint_dir=args.checkpoint_dir,
            use_amp=(not args.no_amp),
        )

    # ── Standard path (v1 / v2 config) ──────────────────────────────────────
    else:
        min_seq_std       = cfg["training"].get("min_seq_std", 0.0)
        balance_bis       = cfg["training"].get("balance_bis", False)
        case_std_pct_low  = cfg["training"].get("case_std_pct_low",  0.0)
        case_std_pct_high = cfg["training"].get("case_std_pct_high", 100.0)

        induction_boost = cfg["training"].get("induction_boost", 0)

        train_ds, val_ds, test_ds = build_datasets(
            h5_path=args.data,
            val_split=cfg["training"]["val_split"],
            test_split=cfg["training"]["test_split"],
            seq_len=cfg["training"]["seq_len"],
            seq_stride=cfg["training"].get("seq_stride", None),
            seed=cfg["training"]["seed"],
            noise_std=cfg["training"].get("noise_std", 0.05),
            cache_in_memory=True,
            min_seq_std=min_seq_std,
            case_std_pct_low=case_std_pct_low,
            case_std_pct_high=case_std_pct_high,
            induction_boost=induction_boost,
        )

        # Free test_ds immediately — not used during training, save ~2 GB RAM
        if hasattr(test_ds, "_cache"):
            test_ds._cache.clear()
        del test_ds
        gc.collect()

        # Validation: use same seq_len as training (seq_len=300) for stable vMAE signal.
        # The previous seq_len=10 caused noisy val metrics (vMAE oscillated 6.97-8.60)
        # because 10-step context is insufficient for the LNN to build good predictions.
        # The AUROC OOM (124 GiB) from v7 is fixed: _auroc_numpy now uses rank-sum
        # O(n log n) instead of the O(n_pos × n_neg) matrix multiplication.
        # The original val_ds from build_datasets (seq_len=train_seq_len) is reused directly.
        val_seq_len = cfg["training"]["seq_len"]
        if False:  # disabled: val rebuild to shorter seq_len is no longer needed
            # CRITICAL: build val with EXACT same patient IDs from the first split.
            # Calling build_datasets again would produce a different patient pool
            # (seq_len=10 allows shorter recordings excluded in seq_len=300 split)
            # AND double-load 25+ GB of data, causing OOM.
            #
            # Fix: extract val_ids directly from the first split's cache, then
            # create a new SequenceDataset (209 patients only, ~2 GB, ~10s to load).
            val_seq_stride = cfg["training"].get("seq_stride", val_seq_len) or val_seq_len
            val_ids = sorted(val_ds._cache.keys())  # exact same patients as first split
            # Free the seq_len=300 val cache before loading seq_len=10 version
            val_ds._cache.clear()
            del val_ds
            gc.collect()
            val_ds = SequenceDataset(
                h5_path=args.data,
                case_ids=val_ids,
                seq_len=val_seq_len,
                seq_stride=val_seq_stride,
                augment=False,
                noise_std=0.0,
                cache_in_memory=True,
            )
            _log(f"Val dataset rebuilt  seq_len={val_seq_len}  stride={val_seq_stride}  "
                 f"val_seqs={len(val_ds):,}  "
                 f"(fast eval mode, same {len(val_ids)} patients)", "DATA")

        num_workers = 0   # in-memory cache → no workers needed on Windows

        # ── BIS-balanced sampler (文档: 训练分布 = 生产分布) ─────────────────
        if balance_bis:
            weights = train_ds.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            print(f"BIS-balanced WeightedRandomSampler enabled "
                  f"(n_bins=10, {len(weights):,} samples)")
            train_shuffle = False
        else:
            sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=train_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"] * 2,  # 64 (not 256): 256×300=76k windows/batch → 32GB VRAM pressure → 17min val
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
        )
        print(f"DataLoader workers: {num_workers} (cache in RAM)")

        if model_version == "v2":
            trainer = TrainerV2(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                checkpoint_dir=args.checkpoint_dir,
                use_amp=(not args.no_amp),
            )
        else:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                checkpoint_dir=args.checkpoint_dir,
                use_amp=(not args.no_amp),
            )

    trainer.fit()


if __name__ == "__main__":
    main()
