"""
trainer_v3.py — MERIDIAN (AnesthesiaNetV3) 三阶段课程学习训练器

课程阶段控制：
  Phase 1 (ep 1 ~ phase2_start-1):
    — 只传入 EEG；drug_ce/vitals=None；激活 L_bis + L_phase
    — 目标：建立稳定的 BIS 回归和相位分类基础

  Phase 2 (phase2_start ~ phase3_start-1):
    — 继续 EEG only；激活 + L_stim（CV 标签）
    — 目标：在稳定 BIS 表征基础上加入刺激检测

  Phase 3 (phase3_start ~ end):
    — 传入 drug_ce + vitals；激活所有损失项
    — 目标：跨模态蒸馏提升 EEG 医学可解释性

指标追踪：
  vMAE / vInd / vRec / PhAcc / StAUC（与 v2 兼容）
  + DistPK / DistVit / PKD（Phase 3 新增）
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

from .loss_v3 import MultiTaskLossV3


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, tag: str = "INFO") -> None:
    print(f"[{_now()}][{tag}] {msg}", flush=True)


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def _auroc_numpy(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via rank-sum (Wilcoxon-Mann-Whitney) — O(n log n)。"""
    n_pos = int((labels == 1).sum())
    n_neg = int(len(labels)) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
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


# ─────────────────────────────────────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────────────────────────────────────

class TrainerV3:
    """
    MERIDIAN 三阶段课程学习训练器。

    兼容 DatasetV3（含 drug_ce/vitals/mask_drug/mask_vital/stim_cv）。
    Phase 1/2 不传入多模态输入，Phase 3 自动传入。
    """

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
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device or torch.device(
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

        self.criterion = MultiTaskLossV3(
            lambda_bis=tcfg.get("lambda_bis", 1.0),
            lambda_phase=tcfg.get("lambda_phase", 0.5),
            lambda_stim=tcfg.get("lambda_stim", 0.3),
            lambda_pkd=tcfg.get("lambda_pkd", 0.3),
            lambda_distill_pk=tcfg.get("lambda_distill_pk", 0.5),
            lambda_distill_vital=tcfg.get("lambda_distill_vital", 0.3),
            lambda_trans=tcfg.get("lambda_trans", 0.2),
            transition_boost=tcfg.get("transition_boost", 2.0),
            vel_threshold=tcfg.get("vel_threshold", 0.2),
            huber_delta=tcfg.get("huber_delta", 0.10),
            focal_gamma=tcfg.get("focal_gamma", 2.0),
            focal_alpha=tcfg.get("focal_alpha", 0.99),
            stim_pos_weight=tcfg.get("stim_pos_weight", 99.0),
            phase2_start_epoch=tcfg.get("phase2_start_epoch", 31),
            phase3_start_epoch=tcfg.get("phase3_start_epoch", 61),
            use_auto_weight=tcfg.get("use_auto_weight", False),
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
        self.no_improve    = 0
        self.start_epoch   = 1

        self.history: Dict[str, list] = {k: [] for k in [
            "train_loss", "train_bis", "train_phase", "train_stim",
            "train_pkd", "train_distill_pk", "train_distill_vital", "train_trans",
            "val_loss", "val_mae", "val_mae_induction", "val_mae_recovery",
            "val_mae_preop", "val_mae_maint",
            "val_phase_acc", "val_stim_auroc",
            "curriculum_phase", "lr",
        ]}

        # 自动恢复
        resume_path = self.checkpoint_dir / "best_model_v3.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=self.device,
                              weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.best_val_mae  = ckpt.get("val_mae", float("inf"))
            self.best_val_loss = ckpt["val_loss"]
            self.start_epoch   = ckpt["epoch"] + 1
            self.history       = ckpt.get("history", self.history)
            # 恢复 LR scheduler 状态（需要先 step optimizer）
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(ckpt["epoch"]):
                self.scheduler.step()
            _log(f"Resumed from epoch {ckpt['epoch']}  "
                 f"best_val_MAE={self.best_val_mae:.2f}  "
                 f"val_loss={self.best_val_loss:.4f}", "RESUME")

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _to(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _cur_phase(self, epoch: int) -> int:
        return self.criterion.get_curriculum_phase(epoch)

    # ── 训练 epoch ────────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        cur_phase = self._cur_phase(epoch)
        n_total = len(self.train_loader)
        steps   = min(self.steps_per_epoch or n_total, n_total)

        accum = {k: 0.0 for k in [
            "loss", "bis_loss", "phase_loss", "stim_loss",
            "pkd_loss", "distill_pk", "distill_vital", "trans_loss",
        ]}
        n_batches = 0
        t0 = time.time()
        sps_acc = 0.0

        loader_iter = iter(self.train_loader)
        bar = (tqdm(total=steps,
                    desc=f"Ep{epoch:03d}[Ph{cur_phase}] train",
                    leave=False)
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
            label_seq = batch["label_seq"]    # (B, T) 归一化 [0,1]
            sqi_mean  = sqi.mean(-1)

            # 分类标签（有则用，无则默认）
            phase_labels = batch.get("phases",
                torch.full(label_seq.shape, 2, dtype=torch.long,
                           device=self.device))
            # v3 使用 stim_cv；回退到 v2 的 stim_events
            stim_labels = batch.get("stim_cv",
                batch.get("stim_events",
                    torch.zeros_like(label_seq)))

            # 多模态输入（Phase 3 才传入）
            drug_ce    = batch.get("drug_ce",    None) if cur_phase >= 3 else None
            vitals     = batch.get("vitals",     None) if cur_phase >= 3 else None
            mask_drug  = batch.get("mask_drug",  None) if cur_phase >= 3 else None
            mask_vital = batch.get("mask_vital", None) if cur_phase >= 3 else None
            ce_velocity = batch.get("ce_velocity", None) if cur_phase >= 3 else None

            t_step = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(
                    wave, features, sqi,
                    drug_ce=drug_ce, vitals=vitals,
                    mask_drug=mask_drug, mask_vital=mask_vital,
                )
                losses = self.criterion(
                    pred_bis     = out["pred_bis"],
                    phase_logits = out["phase_logits"],
                    stim_logits  = out["stim_logits"],
                    label_bis    = label_seq,
                    phase_labels = phase_labels,
                    stim_labels  = stim_labels,
                    sqi_mean     = sqi_mean,
                    epoch        = epoch,
                    # Phase 3 多模态项
                    bis_pkd             = out.get("bis_pkd"),
                    loss_distill_pk     = out.get("loss_distill_pk"),
                    loss_distill_vital  = out.get("loss_distill_vital"),
                    drug_ce             = drug_ce,
                    mask_drug           = mask_drug,
                    ce_velocity         = ce_velocity,
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

            del losses, out

        if bar is not None:
            bar.close()

        n_batches = max(n_batches, 1)
        elapsed = time.time() - t0
        return {
            "train_loss":         accum["loss"]           / n_batches,
            "train_bis":          accum["bis_loss"]        / n_batches,
            "train_phase":        accum["phase_loss"]      / n_batches,
            "train_stim":         accum["stim_loss"]       / n_batches,
            "train_pkd":          accum["pkd_loss"]        / n_batches,
            "train_distill_pk":   accum["distill_pk"]      / n_batches,
            "train_distill_vital":accum["distill_vital"]   / n_batches,
            "train_trans":        accum["trans_loss"]      / n_batches,
            "throughput":         sps_acc / n_batches,
            "epoch_time_s":       elapsed,
            "curriculum_phase":   cur_phase,
        }

    # ── 验证 epoch ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        cur_phase = self._cur_phase(epoch)
        total_loss = 0.0
        n = 0

        all_pred_bis:  list[np.ndarray] = []
        all_label_bis: list[np.ndarray] = []
        all_phase:     list[np.ndarray] = []
        all_pred_ph:   list[np.ndarray] = []
        all_true_ph:   list[np.ndarray] = []
        all_pred_stim: list[np.ndarray] = []
        all_true_stim: list[np.ndarray] = []

        bar = (tqdm(total=len(self.val_loader),
                    desc=f"Ep{epoch:03d}[Ph{cur_phase}] val",
                    leave=False)
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
            stim_labels = batch.get("stim_cv",
                batch.get("stim_events", torch.zeros_like(label_seq)))

            # 验证时不传入多模态（评估 EEG only 推理性能）
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(wave, features, sqi)
                losses = self.criterion(
                    pred_bis     = out["pred_bis"],
                    phase_logits = out["phase_logits"],
                    stim_logits  = out["stim_logits"],
                    label_bis    = label_seq,
                    phase_labels = phase_labels,
                    stim_labels  = stim_labels,
                    sqi_mean     = sqi_mean,
                    epoch        = epoch,
                )

            bs = wave.shape[0]
            total_loss += losses["loss"].item() * bs
            n += bs

            all_pred_bis.append(
                out["pred_bis"][:, -1, 0].detach().cpu().float().numpy() * 100.0)
            all_label_bis.append(batch["label_raw"].cpu().numpy())
            all_phase.append(phase_labels[:, -1].cpu().numpy())

            all_pred_ph.append(
                out["phase_logits"][:, -1, :].argmax(-1).detach().cpu().numpy())
            all_true_ph.append(phase_labels[:, -1].cpu().numpy())

            all_pred_stim.append(
                torch.sigmoid(out["stim_logits"][:, :, 0]).detach().cpu().float().numpy().ravel())
            all_true_stim.append(stim_labels.cpu().float().numpy().ravel())

            if bar is not None:
                bar.update(1)

            del losses, out

        if bar is not None:
            bar.close()

        pred_arr  = np.concatenate(all_pred_bis)
        label_arr = np.concatenate(all_label_bis)
        phase_arr = np.concatenate(all_phase)
        pred_ph   = np.concatenate(all_pred_ph)
        true_ph   = np.concatenate(all_true_ph)

        mae_overall = float(np.abs(pred_arr - label_arr).mean())

        def phase_mae(ph_id):
            m = phase_arr == ph_id
            return float(np.abs(pred_arr[m] - label_arr[m]).mean()) if m.sum() >= 5 else float("nan")

        phase_acc  = float((pred_ph == true_ph).mean())
        st_scores  = np.concatenate(all_pred_stim)
        st_labels  = np.concatenate(all_true_stim)
        stim_auroc = _auroc_numpy(st_scores, st_labels)

        return {
            "val_loss":          total_loss / max(n, 1),
            "val_mae":           mae_overall,
            "val_mae_induction": phase_mae(1),
            "val_mae_recovery":  phase_mae(3),
            "val_mae_preop":     phase_mae(0),
            "val_mae_maint":     phase_mae(2),
            "val_phase_acc":     phase_acc,
            "val_stim_auroc":    stim_auroc,
        }

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    def fit(self) -> None:
        cur_lr  = self.optimizer.param_groups[0]["lr"]
        n_total = len(self.train_loader)
        steps   = self.steps_per_epoch or n_total

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        sep = "=" * 130
        print(f"\n{sep}")
        _log("MERIDIAN (AnesthesiaNetV3) Training Start", "START")
        _log(f"device={self.device}  AMP={self.use_amp}({self.amp_dtype})", "CONFIG")
        _log(f"params={n_params:,}  epochs={self.epochs}  patience={self.patience}  "
             f"lr={cur_lr:.2e}  batch={self.cfg['training']['batch_size']}", "CONFIG")
        _log(f"Phase1: ep1-{self.criterion.phase2_start_epoch-1}  "
             f"Phase2: ep{self.criterion.phase2_start_epoch}-{self.criterion.phase3_start_epoch-1}  "
             f"Phase3: ep{self.criterion.phase3_start_epoch}+", "CURRICULUM")
        _log(f"steps/epoch={steps:,}  train_seqs={len(self.train_loader.dataset):,}  "
             f"val_seqs={len(self.val_loader.dataset):,}", "CONFIG")
        print(sep)

        hdr = (
            f"{'Timestamp':<19}  {'Ep':>4}  {'Ph':>3}  "
            f"{'TotLoss':>8}  {'BIS':>7}  {'Phase':>7}  {'Stim':>7}  "
            f"{'PKD':>7}  {'DistPK':>7}  {'DistVit':>7}  {'Trans':>7}  "
            f"{'vMAE':>6}  {'vInd':>6}  {'vRec':>6}  "
            f"{'PhAcc':>6}  {'StAUC':>6}  "
            f"{'LR':>9}  {'SPS':>6}  {'train_t':>7}  {'val_t':>5}  {'flag'}"
        )
        print(hdr)
        print("-" * len(hdr))

        train_start = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            cur_phase = self._cur_phase(epoch)

            # Phase 切换提示
            if epoch in (self.criterion.phase2_start_epoch,
                         self.criterion.phase3_start_epoch):
                _log(f">>> Entering Curriculum Phase {cur_phase} at epoch {epoch} <<<",
                     "CURRICULUM")

            _log(f"Epoch {epoch}/{self.epochs} train start  "
                 f"phase={cur_phase}  "
                 f"(elapsed={_fmt_elapsed(time.time()-train_start)})", "EPOCH")

            t_train = time.time()
            train_m = self.train_epoch(epoch)
            train_sec = time.time() - t_train

            _log(f"Epoch {epoch} train done  {train_sec:.0f}s  "
                 f"loss={train_m['train_loss']:.4f}  "
                 f"bis={train_m['train_bis']:.4f}  "
                 f"phase={train_m['train_phase']:.4f}  "
                 f"stim={train_m['train_stim']:.4f}  "
                 f"pkd={train_m['train_pkd']:.4f}  "
                 f"dp={train_m['train_distill_pk']:.4f}  "
                 f"dv={train_m['train_distill_vital']:.4f}  "
                 f"trans={train_m['train_trans']:.4f}  "
                 f"sps={train_m.get('throughput',0):.0f}", "TRAIN")

            _log(f"Epoch {epoch}/{self.epochs} val start", "EPOCH")
            t_val = time.time()
            val_m = self.val_epoch(epoch)
            val_sec = time.time() - t_val

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            self.scheduler.step()
            cur_lr = self.optimizer.param_groups[0]["lr"]

            # 更新 history
            for k, v in [
                ("train_loss",          train_m["train_loss"]),
                ("train_bis",           train_m["train_bis"]),
                ("train_phase",         train_m["train_phase"]),
                ("train_stim",          train_m["train_stim"]),
                ("train_pkd",           train_m["train_pkd"]),
                ("train_distill_pk",    train_m["train_distill_pk"]),
                ("train_distill_vital", train_m["train_distill_vital"]),
                ("train_trans",         train_m["train_trans"]),
                ("val_loss",            val_m["val_loss"]),
                ("val_mae",             val_m["val_mae"]),
                ("val_mae_induction",   val_m["val_mae_induction"]),
                ("val_mae_recovery",    val_m["val_mae_recovery"]),
                ("val_mae_preop",       val_m["val_mae_preop"]),
                ("val_mae_maint",       val_m["val_mae_maint"]),
                ("val_phase_acc",       val_m["val_phase_acc"]),
                ("val_stim_auroc",      val_m["val_stim_auroc"]),
                ("curriculum_phase",    cur_phase),
                ("lr",                  cur_lr),
            ]:
                if k in self.history:
                    self.history[k].append(v)

            improved = val_m["val_mae"] < self.best_val_mae
            flag = "*** BEST" if improved else ""

            def _f(v, fmt=".2f"):
                return f"{v:{fmt}}" if (v == v) else "  nan"

            print(
                f"{_now()}  "
                f"{epoch:>4}  "
                f"Ph{cur_phase:>1}  "
                f"{train_m['train_loss']:>8.4f}  "
                f"{train_m['train_bis']:>7.4f}  "
                f"{train_m['train_phase']:>7.4f}  "
                f"{train_m['train_stim']:>7.4f}  "
                f"{train_m['train_pkd']:>7.4f}  "
                f"{train_m['train_distill_pk']:>7.4f}  "
                f"{train_m['train_distill_vital']:>7.4f}  "
                f"{train_m['train_trans']:>7.4f}  "
                f"{val_m['val_mae']:>6.2f}  "
                f"{_f(val_m['val_mae_induction']):>6}  "
                f"{_f(val_m['val_mae_recovery']):>6}  "
                f"{_f(val_m['val_phase_acc']*100,'.1f')+chr(37):>6}  "
                f"{_f(val_m['val_stim_auroc'],'.3f'):>6}  "
                f"{cur_lr:>9.2e}  "
                f"{train_m.get('throughput',0):>6.0f}  "
                f"{train_sec:>6.0f}s  "
                f"{val_sec:>4.0f}s  "
                f"{flag}",
                flush=True,
            )

            elapsed_total = _fmt_elapsed(time.time() - train_start)
            epochs_done   = epoch - self.start_epoch + 1
            eta_s = (time.time() - train_start) / epochs_done * (self.epochs - epoch)
            _log(f"Epoch {epoch} val done  {val_sec:.0f}s  "
                 f"vMAE={val_m['val_mae']:.2f}  "
                 f"vInd={_f(val_m['val_mae_induction'])}  "
                 f"vRec={_f(val_m['val_mae_recovery'])}  "
                 f"PhAcc={_f(val_m['val_phase_acc']*100,'.1f')}%  "
                 f"StAUC={_f(val_m['val_stim_auroc'],'.3f')}  "
                 f"elapsed={elapsed_total}  eta={_fmt_elapsed(eta_s)}", "VAL")

            if improved:
                self.best_val_mae  = val_m["val_mae"]
                self.best_val_loss = val_m["val_loss"]
                self.no_improve = 0
                self._save_checkpoint(epoch, val_m["val_mae"])
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    _log(f"Early stopping at epoch {epoch} "
                         f"(patience={self.patience}  no_improve={self.no_improve})", "STOP")
                    break

        print("-" * len(hdr))
        _log(f"Training complete.  Best val_MAE={self.best_val_mae:.2f} BIS", "DONE")

    def _save_checkpoint(self, epoch: int, val_mae: float) -> None:
        path = self.checkpoint_dir / "best_model_v3.pt"
        torch.save({
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "val_mae":             val_mae,
            "val_loss":            self.best_val_loss,
            "history":             self.history,
            "cfg":                 self.cfg,
        }, path)
        _log(f"Checkpoint saved  epoch={epoch}  val_MAE={val_mae:.2f} BIS  "
             f"path={path}", "CKPT")
