"""
Trainer — training and validation loop for AnesthesiaNet.

Performance optimisations:
  - AMP (BF16): RTX 5090 Tensor Cores give ~2-4× throughput on BF16
  - GradScaler: keeps AMP numerically stable (required even for BF16 on some ops)
  - Batch-level tqdm bar: loss, GPU mem, samples/sec visible in real time
  - Epoch summary: train/val loss, MAE, LR, monotonic loss breakdown

Data flow (P1 fix — seq_len is now used correctly):
  SequenceDataset returns batches with shape:
    wave      : (B, seq_len, n_ch, win_samp)
    features  : (B, seq_len, n_feat)
    sqi       : (B, seq_len, n_ch)
    label     : (B,)            — BIS of last window, normalised [0,1]
    label_seq : (B, seq_len)    — BIS for every window in sequence [0,1]
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
try:
    from tqdm import tqdm
    _TQDM = sys.stdout.isatty()  # disable tqdm when redirected to file
except ImportError:
    _TQDM = False

from .loss import AnesthesiaLoss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "outputs/checkpoints",
        use_amp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg["training"]
        self.epochs         = tcfg["epochs"]
        self.patience       = tcfg["patience"]
        # steps_per_epoch: if set, randomly samples this many batches per epoch.
        # Prevents 73-min epochs when the dataset has 13M+ sequences.
        # None = use all batches (full epoch).
        self.steps_per_epoch = tcfg.get("steps_per_epoch", None)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tcfg["lr"], weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        self.criterion = AnesthesiaLoss(
            lambda_monotonic=tcfg["lambda_monotonic"],
            lambda_physio=tcfg.get("lambda_physio", 0.0),
        )

        # ── Automatic Mixed Precision ────────────────────────────────────
        # BF16 preferred on Ampere/Hopper/Blackwell (RTX 3090/4090/5090).
        # Falls back to FP16 if BF16 is not available, or disables AMP on CPU.
        self.use_amp = use_amp and self.device.type == "cuda"
        if self.use_amp:
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=(self.amp_dtype == torch.float16)
            )
            print(f"AMP enabled: {self.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        self.model.to(self.device)
        self.best_val_loss = float("inf")
        self.no_improve = 0
        self.start_epoch = 1
        self.history: Dict[str, list] = {
            "train_loss": [], "train_mse": [], "train_mono": [],
            "val_loss": [], "val_mae": [], "lr": [],
        }

        # ── Resume from checkpoint if available ──────────────────────────
        resume_path = self.checkpoint_dir / "best_model.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.best_val_loss = ckpt["val_loss"]
            self.start_epoch   = ckpt["epoch"] + 1
            self.history       = ckpt.get("history", self.history)
            # advance scheduler to match resumed epoch
            # call optimizer.step() once first to suppress PyTorch warning
            self.optimizer.step()
            for _ in range(ckpt["epoch"]):
                self.scheduler.step()
            print(f"Resumed from epoch {ckpt['epoch']}  "
                  f"best_val_loss={self.best_val_loss:.4f}  "
                  f"val_MAE={ckpt['val_mae']:.2f} BIS")

    def _batch_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _forward_batch(
        self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wave      = batch["wave"]       # (B, T, n_ch, win_samp)
        feat      = batch["features"]   # (B, T, n_feat)
        sqi       = batch["sqi"]        # (B, T, n_ch)
        label     = batch["label"]      # (B,)
        label_seq = batch["label_seq"]  # (B, T)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                            enabled=self.use_amp):
            pred, pred_seq, _ = self.model(wave, feat, sqi)

        sqi_mean = sqi.mean(dim=(1, 2))
        return pred, pred_seq, label, label_seq, sqi_mean

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = total_base = total_mono = 0.0
        n = 0
        t_data = 0.0
        t_fwd  = 0.0
        t_bwd  = 0.0
        t0_epoch = time.perf_counter()

        n_batches = (self.steps_per_epoch
                     if self.steps_per_epoch is not None
                     else len(self.train_loader))
        bar = _make_bar(n_batches, desc=f"Epoch {epoch:03d} train", leave=False)

        t_start = time.perf_counter()
        steps_done = 0
        for batch in self.train_loader:
            if self.steps_per_epoch is not None and steps_done >= self.steps_per_epoch:
                break
            t_data += time.perf_counter() - t_start

            t_fwd_start = time.perf_counter()
            batch = self._batch_to_device(batch)
            pred, pred_seq, label, label_seq, sqi_mean = self._forward_batch(batch)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
                losses = self.criterion(pred, label, sqi_mean,
                                        pred_seq=pred_seq, label_seq=label_seq)
            loss = losses["loss"]
            t_fwd += time.perf_counter() - t_fwd_start

            t_bwd_start = time.perf_counter()
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            t_bwd += time.perf_counter() - t_bwd_start

            bs = label.size(0)
            total_loss += loss.item() * bs
            total_base += losses["base"] * bs
            total_mono += losses["monotonic"] * bs
            n += bs
            steps_done += 1

            # ── tqdm batch-level stats ───────────────────────────────────
            if bar is not None:
                gpu_mb = (torch.cuda.memory_allocated(self.device) / 1e6
                          if self.device.type == "cuda" else 0.0)
                sps = bs / max(time.perf_counter() - t_start, 1e-6)
                bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mono=f"{losses['monotonic']:.4f}",
                    gpu=f"{gpu_mb:.0f}MB",
                    sps=f"{sps:.0f}",
                )
                bar.update(1)

            t_start = time.perf_counter()

        if bar is not None:
            bar.close()

        epoch_time = time.perf_counter() - t0_epoch
        return {
            "train_loss": total_loss / n,
            "train_base": total_base / n,
            "train_mono": total_mono / n,
            "time_data_s":  t_data,
            "time_fwd_s":   t_fwd,
            "time_bwd_s":   t_bwd,
            "epoch_time_s": epoch_time,
            "throughput":   n / epoch_time,
        }

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_pred, all_label = [], []
        n = 0

        bar = _make_bar(len(self.val_loader), desc=f"Epoch {epoch:03d} val  ", leave=False)

        for batch in self.val_loader:
            batch = self._batch_to_device(batch)
            pred, pred_seq, label, label_seq, sqi_mean = self._forward_batch(batch)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
                losses = self.criterion(pred, label, sqi_mean,
                                        pred_seq=pred_seq, label_seq=label_seq)
            bs = label.size(0)
            total_loss += losses["loss"].item() * bs
            n += bs

            all_pred.append(pred.squeeze(-1).cpu() * 100.0)
            all_label.append(batch["label_raw"].cpu())

            if bar is not None:
                bar.update(1)

        if bar is not None:
            bar.close()

        all_pred  = torch.cat(all_pred)
        all_label = torch.cat(all_label)
        mae = float((all_pred - all_label).abs().mean())
        return {"val_loss": total_loss / n, "val_mae": mae}

    def fit(self) -> None:
        cur_lr = self.optimizer.param_groups[0]["lr"]
        n_total = len(self.train_loader)
        steps   = self.steps_per_epoch or n_total
        coverage = steps / n_total * 100
        print(f"\nTraining on {self.device}  |  AMP={self.use_amp} ({self.amp_dtype})")
        print(f"Epochs={self.epochs}  patience={self.patience}  "
              f"lr={cur_lr:.2e}  batch={self.cfg['training']['batch_size']}")
        print(f"steps/epoch={steps:,} / {n_total:,} total  "
              f"({coverage:.1f}% coverage per epoch, "
              f"full dataset seen every ~{100/coverage:.0f} epochs)")
        print(f"{'Epoch':>6}  {'trainLoss':>10}  {'baseLoss':>10}  "
              f"{'monoLoss':>10}  {'valLoss':>10}  {'valMAE':>8}  "
              f"{'LR':>9}  {'sps':>7}  {'time':>7}")
        print("-" * 100)

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m   = self.val_epoch(epoch)
            self.scheduler.step()

            cur_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_m["train_loss"])
            self.history["train_mse"].append(train_m["train_base"])
            self.history["train_mono"].append(train_m["train_mono"])
            self.history["val_loss"].append(val_m["val_loss"])
            self.history["val_mae"].append(val_m["val_mae"])
            self.history["lr"].append(cur_lr)

            # ── Epoch summary line ───────────────────────────────────────
            flag = "*" if val_m["val_loss"] < self.best_val_loss else " "
            print(
                f"{epoch:>6}  "
                f"{train_m['train_loss']:>10.4f}  "
                f"{train_m['train_base']:>10.4f}  "
                f"{train_m['train_mono']:>10.4f}  "
                f"{val_m['val_loss']:>10.4f}  "
                f"{val_m['val_mae']:>8.2f}  "
                f"{cur_lr:>9.2e}  "
                f"{train_m['throughput']:>7.0f}  "
                f"{train_m['epoch_time_s']:>6.1f}s"
                f"  {flag}"
            )

            # ── Timing breakdown every 10 epochs ────────────────────────
            if epoch % 10 == 0:
                print(f"         [timing] data={train_m['time_data_s']:.1f}s  "
                      f"fwd={train_m['time_fwd_s']:.1f}s  "
                      f"bwd={train_m['time_bwd_s']:.1f}s")

            if val_m["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_m["val_loss"]
                self.no_improve = 0
                self._save_checkpoint(epoch, val_m["val_mae"])
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}  "
                          f"(no improvement for {self.patience} epochs)")
                    break

        print("-" * 100)
        print(f"Training complete.  Best val_loss={self.best_val_loss:.4f}")

    def _save_checkpoint(self, epoch: int, val_mae: float) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_mae": val_mae,
                "val_loss": self.best_val_loss,
                "history": self.history,
                "cfg": self.cfg,
            },
            path,
        )
        print(f"           -> checkpoint saved  val_MAE={val_mae:.2f} BIS")


def _make_bar(total: int, desc: str, leave: bool = True):
    """Return a tqdm bar if available, else None."""
    if not _TQDM:
        return None
    return tqdm(total=total, desc=desc, leave=leave,
                bar_format="{l_bar}{bar:30}{r_bar}")
