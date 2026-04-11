"""
trainer_v2.py - Training loop for AnesthesiaNetV2 (multi-task).

修复的Bug：
  - val_epoch 中 val_phase_acc 从未计算（history 里有但值全为0）
  - 缺少刺激检测 AUROC/sensitivity 指标
  - 增加相位分类混淆矩阵输出（每N个epoch）
  - 增加吞吐量/ETA 显示

Tracks per-task losses and reports phase-level MAE to show improvement
on the rare induction/recovery phases.
"""

from __future__ import annotations
import sys
import time
import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    _TQDM = sys.stdout.isatty()
except ImportError:
    _TQDM = False


def _now() -> str:
    """Current local time as compact string: 2026-04-07 23:15:42"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, tag: str = "INFO") -> None:
    """Structured log line: [2026-04-07 23:15:42][INFO] msg"""
    print(f"[{_now()}][{tag}] {msg}", flush=True)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as H:MM:SS."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"

from .loss_v2 import MultiTaskLoss


def _auroc_numpy(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via rank-sum (Wilcoxon-Mann-Whitney) — O(n log n), no matrix alloc.

    Previous implementation used pos[:,None]>neg[None,:] which allocates a
    n_pos × n_neg boolean matrix — 124 GiB OOM with seq_len=300 val.
    Rank-based computation: AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    where R_pos is the sum of ranks of positive scores in the combined ranking.
    """
    n_pos = int((labels == 1).sum())
    n_neg = int(len(labels)) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank all scores (1-based), average ranks for ties
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Resolve ties by averaging
    scores_sorted = scores[order]
    i = 0
    while i < len(scores_sorted):
        j = i + 1
        while j < len(scores_sorted) and scores_sorted[j] == scores_sorted[i]:
            j += 1
        if j > i + 1:
            avg_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(ranks[labels == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


class TrainerV2:
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
        self.val_loader   = val_loader
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg["training"]
        self.epochs          = tcfg["epochs"]
        self.patience        = tcfg["patience"]
        self.steps_per_epoch = tcfg.get("steps_per_epoch", None)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tcfg["lr"], weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.criterion = MultiTaskLoss(
            lambda_bis=tcfg.get("lambda_bis", 1.0),
            lambda_phase=tcfg.get("lambda_phase", 0.5),
            lambda_stim=tcfg.get("lambda_stim", 0.3),
            lambda_mono=tcfg.get("lambda_mono", 0.3),
            focal_gamma=tcfg.get("focal_gamma", 2.0),
            focal_alpha=tcfg.get("focal_alpha", 0.99),
            stim_pos_weight=tcfg.get("stim_pos_weight", 99.0),  # BUG FIX: 默认99非1
            use_auto_weight=tcfg.get("use_auto_weight", False),
            auto_weight_temp=tcfg.get("auto_weight_temp", 0.5),
        )

        self.use_amp = use_amp and self.device.type == "cuda"
        if self.use_amp:
            self.amp_dtype = (torch.bfloat16
                              if torch.cuda.is_bf16_supported() else torch.float16)
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=(self.amp_dtype == torch.float16))
            print(f"AMP enabled: {self.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        self.model.to(self.device)
        self.best_val_mae  = float("inf")
        self.best_val_loss = float("inf")
        self.no_improve = 0
        self.start_epoch = 1
        self.history: Dict[str, list] = {
            k: [] for k in [
                "train_loss", "train_bis", "train_phase", "train_stim", "train_mono",
                "val_loss", "val_mae", "val_mae_induction", "val_mae_recovery",
                "val_mae_preop", "val_mae_maint",
                "val_phase_acc", "val_stim_auroc",   # 修复：确保这些真的被填充
                "lr",
            ]
        }

        # Resume
        resume_path = self.checkpoint_dir / "best_model_v2.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=self.device,
                              weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.best_val_mae  = ckpt.get("val_mae", float("inf"))
            self.best_val_loss = ckpt["val_loss"]
            self.start_epoch   = ckpt["epoch"] + 1
            self.history       = ckpt.get("history", self.history)
            # Step scheduler N times to restore LR state (must call optimizer.step first
            # to suppress the "step before optimizer" warning — use a dummy step)
            self.optimizer.step()   # zero-grad already; creates grad tensors for Adam
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(ckpt["epoch"]):
                self.scheduler.step()
            _log(f"Resumed from epoch {ckpt['epoch']}  "
                 f"best_val_loss={self.best_val_loss:.4f}  "
                 f"val_MAE={ckpt['val_mae']:.2f} BIS", "RESUME")

    def _to(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        n_total = len(self.train_loader)
        steps   = min(self.steps_per_epoch or n_total, n_total)

        accum = {k: 0.0 for k in
                 ["loss", "bis_loss", "phase_loss", "stim_loss", "mono_loss"]}
        n_batches = 0
        t0 = time.time()
        sps_acc = 0.0

        loader_iter = iter(self.train_loader)
        bar = (tqdm(total=steps, desc=f"Epoch {epoch:03d} train", leave=False)
               if _TQDM else None)

        for step in range(steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            batch = self._to(batch)
            wave      = batch["wave"]
            features  = batch["features"]
            sqi       = batch["sqi"]
            label_seq = batch["label_seq"]        # (B, T) normalised
            sqi_mean  = sqi.mean(-1)              # (B, T)

            # Multi-task labels (fall back to zeros if not in HDF5)
            phase_labels = batch.get("phases",
                torch.full(label_seq.shape, 2, dtype=torch.long,
                           device=self.device))  # default: maintenance
            stim_labels = batch.get("stim_events",
                torch.zeros_like(label_seq))

            t_step = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.use_amp):
                pred_bis, phase_logits, stim_logits, _, _ = self.model(
                    wave, features, sqi)
                losses = self.criterion(
                    pred_bis, phase_logits, stim_logits,
                    label_seq, phase_labels, stim_labels, sqi_mean,
                )

            if self.scaler:
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            bs = wave.shape[0] * wave.shape[1]
            dt = time.time() - t_step
            sps_acc += bs / max(dt, 1e-6)

            for k in accum:
                accum[k] += losses[k].item()
            n_batches += 1

            if bar is not None:
                bar.update(1)
                bar.set_postfix(loss=f"{losses['loss'].item():.4f}",
                                sps=f"{bs/max(dt,1e-6):.0f}")

            # Explicitly release computation graph — prevents graph accumulation
            # across steps when AMP is enabled (scaler keeps references otherwise)
            del losses, pred_bis, phase_logits, stim_logits

        if bar is not None:
            bar.close()

        n_batches = max(n_batches, 1)
        elapsed = time.time() - t0
        return {
            "train_loss":  accum["loss"]       / n_batches,
            "train_bis":   accum["bis_loss"]   / n_batches,
            "train_phase": accum["phase_loss"] / n_batches,
            "train_stim":  accum["stim_loss"]  / n_batches,
            "train_mono":  accum["mono_loss"]  / n_batches,
            "throughput":  sps_acc / n_batches,
            "epoch_time_s": elapsed,
        }

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        """
        修复：
          - 计算相位分类准确率 (val_phase_acc) —— 之前从未填充
          - 计算刺激检测 AUROC
          - 使用实际 phase_labels 计算分相位 MAE
        """
        self.model.eval()
        total_loss = 0.0
        n = 0

        # Memory-safe accumulation: convert to numpy immediately after each batch.
        # Storing CPU tensors in Python lists retains the full tensor + any
        # residual autograd metadata, causing ~50-100 MB leak per val epoch.
        all_pred_bis:  list[np.ndarray] = []
        all_label_bis: list[np.ndarray] = []
        all_phase:     list[np.ndarray] = []
        all_pred_ph:   list[np.ndarray] = []
        all_true_ph:   list[np.ndarray] = []
        all_pred_stim: list[np.ndarray] = []
        all_true_stim: list[np.ndarray] = []

        bar = (tqdm(total=len(self.val_loader),
                    desc=f"Epoch {epoch:03d} val", leave=False)
               if _TQDM else None)

        for batch in self.val_loader:
            batch = self._to(batch)
            wave      = batch["wave"]
            features  = batch["features"]
            sqi       = batch["sqi"]
            label_seq = batch["label_seq"]
            sqi_mean  = sqi.mean(-1)

            phase_labels = batch.get("phases",
                torch.full(label_seq.shape, 2, dtype=torch.long,
                           device=self.device))
            stim_labels = batch.get("stim_events",
                torch.zeros_like(label_seq))

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.use_amp):
                pred_bis, phase_logits, stim_logits, _, _ = self.model(
                    wave, features, sqi)
                losses = self.criterion(
                    pred_bis, phase_logits, stim_logits,
                    label_seq, phase_labels, stim_labels, sqi_mean,
                )

            bs = wave.shape[0]
            total_loss += losses["loss"].item() * bs
            n += bs

            # Convert to numpy immediately — never accumulate GPU/CPU tensors in lists
            all_pred_bis.append(
                pred_bis[:, -1, 0].detach().cpu().float().numpy() * 100.0)
            all_label_bis.append(
                batch["label_raw"].cpu().numpy())
            all_phase.append(
                phase_labels[:, -1].cpu().numpy())

            # ── 相位分类：最后一步 ──────────────────────────────────────────
            all_pred_ph.append(
                phase_logits[:, -1, :].argmax(-1).detach().cpu().numpy())
            all_true_ph.append(
                phase_labels[:, -1].cpu().numpy())

            # ── 刺激检测：全序列 ────────────────────────────────────────────
            all_pred_stim.append(
                torch.sigmoid(stim_logits[:, :, 0]).detach().cpu().float().numpy().ravel())
            all_true_stim.append(
                stim_labels.cpu().float().numpy().ravel())

            if bar is not None:
                bar.update(1)

        if bar is not None:
            bar.close()

        pred_arr  = np.concatenate(all_pred_bis)
        label_arr = np.concatenate(all_label_bis)
        phase_arr = np.concatenate(all_phase)
        pred_ph   = np.concatenate(all_pred_ph)
        true_ph   = np.concatenate(all_true_ph)

        mae_overall = float(np.abs(pred_arr - label_arr).mean())

        # 分相位 BIS MAE
        def phase_mae(ph_id):
            m = phase_arr == ph_id
            if m.sum() < 5:
                return float("nan")
            return float(np.abs(pred_arr[m] - label_arr[m]).mean())

        # 相位分类准确率
        phase_acc = float((pred_ph == true_ph).mean())

        # 刺激检测 AUROC
        st_scores = np.concatenate(all_pred_stim)
        st_labels = np.concatenate(all_true_stim)
        stim_auroc = _auroc_numpy(st_scores, st_labels)

        return {
            "val_loss":          total_loss / max(n, 1),
            "val_mae":           mae_overall,
            "val_mae_induction": phase_mae(1),
            "val_mae_recovery":  phase_mae(3),
            "val_mae_preop":     phase_mae(0),
            "val_mae_maint":     phase_mae(2),
            "val_phase_acc":     phase_acc,           # 修复
            "val_stim_auroc":    stim_auroc,           # 修复
        }

    def fit(self) -> None:
        cur_lr = self.optimizer.param_groups[0]["lr"]
        n_total = len(self.train_loader)
        steps   = self.steps_per_epoch or n_total

        # 打印模型和训练配置摘要
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        sep = "=" * 110
        print(f"\n{sep}")
        _log(f"AnesthesiaNetV2 Training Start", "START")
        _log(f"device={self.device}  AMP={self.use_amp}({self.amp_dtype})", "CONFIG")
        _log(f"params={n_params:,}  epochs={self.epochs}  patience={self.patience}  "
             f"lr={cur_lr:.2e}  batch={self.cfg['training']['batch_size']}", "CONFIG")
        _log(f"steps/epoch={steps:,}  train_seqs={len(self.train_loader.dataset):,}  "
             f"val_seqs={len(self.val_loader.dataset):,}", "CONFIG")
        print(sep)

        # 表头
        hdr = (f"{'Timestamp':<19}  {'Ep':>4}  "
               f"{'TotLoss':>8}  {'BIS':>7}  {'Phase':>7}  {'Stim':>7}  {'Mono':>7}  "
               f"{'vMAE':>6}  {'vInd':>6}  {'vRec':>6}  "
               f"{'PhAcc':>6}  {'StAUC':>6}  "
               f"{'LR':>9}  {'SPS':>6}  {'train_t':>7}  {'val_t':>5}  {'flag'}")
        print(hdr)
        print("-" * len(hdr))

        train_start = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            _log(f"Epoch {epoch}/{self.epochs} train start  "
                 f"(elapsed={_fmt_elapsed(time.time()-train_start)})", "EPOCH")

            t_train = time.time()
            train_m = self.train_epoch(epoch)
            train_sec = time.time() - t_train

            _log(f"Epoch {epoch} train done  {train_sec:.0f}s  "
                 f"loss={train_m['train_loss']:.4f}  bis={train_m['train_bis']:.4f}  "
                 f"phase={train_m['train_phase']:.4f}  stim={train_m['train_stim']:.4f}  "
                 f"mono={train_m['train_mono']:.4f}  sps={train_m.get('throughput',0):.0f}",
                 "TRAIN")

            _log(f"Epoch {epoch}/{self.epochs} val start", "EPOCH")
            t_val = time.time()
            val_m = self.val_epoch(epoch)
            val_sec = time.time() - t_val

            # Single empty_cache per epoch after both train and val complete.
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            self.scheduler.step()
            cur_lr = self.optimizer.param_groups[0]["lr"]

            # 更新 history
            for k, v in [
                ("train_loss",  train_m["train_loss"]),
                ("train_bis",   train_m["train_bis"]),
                ("train_phase", train_m["train_phase"]),
                ("train_stim",  train_m["train_stim"]),
                ("train_mono",  train_m["train_mono"]),
                ("val_loss",    val_m["val_loss"]),
                ("val_mae",     val_m["val_mae"]),
                ("val_mae_induction", val_m["val_mae_induction"]),
                ("val_mae_recovery",  val_m["val_mae_recovery"]),
                ("val_mae_preop",     val_m["val_mae_preop"]),
                ("val_mae_maint",     val_m["val_mae_maint"]),
                ("val_phase_acc",     val_m["val_phase_acc"]),
                ("val_stim_auroc",    val_m["val_stim_auroc"]),
                ("lr",          cur_lr),
            ]:
                if k in self.history:
                    self.history[k].append(v)

            improved = val_m["val_mae"] < self.best_val_mae
            flag = "*** BEST" if improved else ""

            def _f(v, fmt=".2f"):
                return f"{v:{fmt}}" if (v == v) else "  nan"

            ind_s  = _f(val_m["val_mae_induction"])
            rec_s  = _f(val_m["val_mae_recovery"])
            ph_s   = _f(val_m["val_phase_acc"] * 100, ".1f") + "%"
            auc_s  = _f(val_m["val_stim_auroc"], ".3f")
            sps    = train_m.get("throughput", 0)
            elapsed_total = _fmt_elapsed(time.time() - train_start)

            # Summary row (timestamped)
            print(
                f"{_now()}  "
                f"{epoch:>4}  "
                f"{train_m['train_loss']:>8.4f}  "
                f"{train_m['train_bis']:>7.4f}  "
                f"{train_m['train_phase']:>7.4f}  "
                f"{train_m['train_stim']:>7.4f}  "
                f"{train_m['train_mono']:>7.4f}  "
                f"{val_m['val_mae']:>6.2f}  "
                f"{ind_s:>6}  "
                f"{rec_s:>6}  "
                f"{ph_s:>6}  "
                f"{auc_s:>6}  "
                f"{cur_lr:>9.2e}  "
                f"{sps:>6.0f}  "
                f"{train_sec:>6.0f}s  "
                f"{val_sec:>4.0f}s  "
                f"{flag}",
                flush=True,
            )

            # Detailed val log
            _log(f"Epoch {epoch} val done  {val_sec:.0f}s  "
                 f"vMAE={val_m['val_mae']:.2f}  vInd={ind_s}  vRec={rec_s}  "
                 f"PhAcc={ph_s}  StAUC={auc_s}  "
                 f"total_elapsed={elapsed_total}  "
                 f"eta={_fmt_elapsed((time.time()-train_start)/(epoch-self.start_epoch+1)*(self.epochs-epoch))}",
                 "VAL")

            if improved:
                self.best_val_mae  = val_m["val_mae"]
                self.best_val_loss = val_m["val_loss"]
                self.no_improve = 0
                self._save_checkpoint(epoch, val_m["val_mae"])
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    _log(f"Early stopping at epoch {epoch} (patience={self.patience}  "
                         f"no_improve={self.no_improve})", "STOP")
                    break

        print("-" * len(hdr))
        _log(f"Training complete.  Best val_MAE={self.best_val_mae:.2f} BIS", "DONE")

    def _save_checkpoint(self, epoch: int, val_mae: float) -> None:
        path = self.checkpoint_dir / "best_model_v2.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_mae":  val_mae,
            "val_loss": self.best_val_loss,
            "history":  self.history,
            "cfg":      self.cfg,
        }, path)
        _log(f"Checkpoint saved  epoch={epoch}  val_MAE={val_mae:.2f} BIS  "
             f"path={path}", "CKPT")
