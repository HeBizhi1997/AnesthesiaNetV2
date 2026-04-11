"""
TBPTTTrainer — Truncated Back-Propagation Through Time for AnesthesiaNet.

修复的Bug（V2 兼容性）：
  - PatientStore 现在加载 phases 和 stim_events（之前只加载 waves/features/sqi/labels）
  - train_epoch 正确处理 AnesthesiaNetV2 的 5 个返回值（之前解包为3个值会崩溃）
  - 根据 model_version 自动选择 AnesthesiaLoss (V1) 或 MultiTaskLoss (V2)
  - V2 模式下 Loss 计算使用 MultiTaskLoss 接口

Core idea
─────────
The standard Trainer shuffles 10-second snippets from random positions in
random surgeries, always resetting h=None.  The LNN never sees its own
hidden state used as input beyond 10 steps → degenerates to a stateless MLP.

TBPTTTrainer instead:
  1. Loads all training cases into RAM (per-patient arrays).
  2. Each epoch, shuffles the patient list and groups them into batches of
     `batch_size` patients.
  3. For each group, processes ALL windows in sequential 5-minute (seq_len=300)
     chunks — propagating h across chunks, detaching gradients at each boundary
     (standard TBPTT).
  4. Loss computed on every valid timestep in each chunk; padded positions
     (for patients shorter than max chunk in the group) are masked out.

Usage
─────
  python scripts/train.py --config configs/pipeline_v2.yaml --tbptt
"""

from __future__ import annotations
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import h5py

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from .loss import AnesthesiaLoss
from .loss_v2 import MultiTaskLoss
from .trainer import Trainer, _make_bar


# ─────────────────────────────────────────────────────────────────────────────
# In-memory patient store
# ─────────────────────────────────────────────────────────────────────────────

class PatientStore:
    """
    Loads all patient arrays into RAM at init.

    修复：现在同时加载 phases 和 stim_events（多任务标签），
    供 TBPTTTrainer 在 V2 模式下使用。

    Each case stored as dict:
      waves       : (N, n_ch, W)
      features    : (N, F)
      sqi         : (N, n_ch)
      labels      : (N,)
      times       : (N,)
      phases      : (N,)  int64  — phase label per window (if available)
      stim_events : (N,)  float32 — stimulation event flag (if available)
    """

    def __init__(self, h5_path: str, case_ids: List[str],
                 verbose: bool = True):
        self._store: Dict[str, Dict[str, np.ndarray]] = {}
        self._valid_ids: List[str] = []
        self._has_multitask = False   # True if any case has phases/stim_events

        bar = (tqdm(case_ids, desc="  Loading patients", leave=False)
               if _TQDM and verbose else case_ids)
        with h5py.File(h5_path, "r") as f:
            for cid in bar:
                if cid not in f:
                    continue
                grp = f[cid]
                entry = {
                    "waves":    grp["waves"][:].astype(np.float32),
                    "features": grp["features"][:].astype(np.float32),
                    "sqi":      grp["sqi"][:].astype(np.float32),
                    "labels":   grp["labels"][:].astype(np.float32),
                }
                # 修复：加载多任务标签（如果存在）
                if "phases" in grp:
                    entry["phases"]      = grp["phases"][:].astype(np.int64)
                    entry["stim_events"] = grp["stim_events"][:].astype(np.float32)
                    self._has_multitask  = True
                if "times" in grp:
                    entry["times"] = grp["times"][:]
                self._store[cid] = entry
                self._valid_ids.append(cid)

        total_w = sum(len(v["labels"]) for v in self._store.values())
        if verbose:
            mt_str = "✓ phases+stim" if self._has_multitask else "✗ no multitask labels"
            print(f"  PatientStore: {len(self._valid_ids)} patients, "
                  f"{total_w:,} windows total in RAM  [{mt_str}]")

    def ids(self) -> List[str]:
        return list(self._valid_ids)

    def get(self, case_id: str) -> Dict[str, np.ndarray]:
        return self._store[case_id]

    def total_windows(self) -> int:
        return sum(len(v["labels"]) for v in self._store.values())

    @property
    def has_multitask(self) -> bool:
        return self._has_multitask


# ─────────────────────────────────────────────────────────────────────────────
# TBPTT Trainer
# ─────────────────────────────────────────────────────────────────────────────

class TBPTTTrainer(Trainer):
    """
    Extends Trainer with a TBPTT train_epoch.

    修复：根据 model_version 自动选择 V1 或 V2 损失函数。
    修复：正确处理 AnesthesiaNetV2 的 5 个返回值。
    """

    def __init__(
        self,
        model: nn.Module,
        train_store: PatientStore,
        val_loader,            # standard DataLoader for validation
        cfg: dict,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "outputs/checkpoints",
        use_amp: bool = True,
    ):
        # ── Reuse Trainer.__init__ for common setup (optimizer, AMP, etc.) ──
        super().__init__(
            model=model,
            train_loader=val_loader,   # placeholder — train_epoch is overridden
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_amp=use_amp,
        )
        self.train_store = train_store
        tcfg = cfg["training"]
        self.seq_len        = tcfg["seq_len"]          # 300
        self.batch_patients = tcfg["batch_size"]       # patients per TBPTT group
        self.noise_std      = tcfg.get("noise_std", 0.05)
        self.model_version  = tcfg.get("model_version", "v1")

        # 修复：根据模型版本选择损失函数
        if self.model_version == "v2":
            # 替换 Trainer.__init__ 中创建的 AnesthesiaLoss（V1）
            self.criterion = MultiTaskLoss(
                lambda_bis=tcfg.get("lambda_bis", 1.0),
                lambda_phase=tcfg.get("lambda_phase", 0.5),
                lambda_stim=tcfg.get("lambda_stim", 0.3),
                lambda_mono=tcfg.get("lambda_mono", 0.3),
                focal_gamma=tcfg.get("focal_gamma", 2.0),
                focal_alpha=tcfg.get("focal_alpha", 0.99),
                stim_pos_weight=tcfg.get("stim_pos_weight", 99.0),
                use_auto_weight=tcfg.get("use_auto_weight", False),
                auto_weight_temp=tcfg.get("auto_weight_temp", 0.5),
            )
            self.criterion.to(self.device)
            print(f"TBPTTTrainer: using MultiTaskLoss (V2 mode)")
        else:
            print(f"TBPTTTrainer: using AnesthesiaLoss (V1 mode)")

    # ── TBPTT training epoch ──────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        all_ids = self.train_store.ids()
        random.shuffle(all_ids)

        total_loss = total_base = total_mono = 0.0
        n_windows = 0
        t0 = time.perf_counter()

        n_groups  = max(1, len(all_ids) // self.batch_patients)
        bar = _make_bar(n_groups, desc=f"Epoch {epoch:03d} train", leave=False)

        has_mt = (self.model_version == "v2" and self.train_store.has_multitask)

        for g in range(n_groups):
            grp_ids = all_ids[g * self.batch_patients: (g + 1) * self.batch_patients]
            if not grp_ids:
                continue

            gdata = [self.train_store.get(pid) for pid in grp_ids]
            B = len(gdata)
            n_ch   = gdata[0]["waves"].shape[1]
            win_s  = gdata[0]["waves"].shape[2]
            n_feat = gdata[0]["features"].shape[1]

            chunk_counts = [len(d["labels"]) // self.seq_len for d in gdata]
            max_chunks   = max(chunk_counts) if chunk_counts else 0
            if max_chunks == 0:
                continue

            h: Optional[torch.Tensor] = None  # reset for each patient group

            for ci in range(max_chunks):
                waves_np  = np.zeros((B, self.seq_len, n_ch, win_s), dtype=np.float32)
                feat_np   = np.zeros((B, self.seq_len, n_feat),       dtype=np.float32)
                sqi_np    = np.zeros((B, self.seq_len, n_ch),         dtype=np.float32)
                lab_np    = np.zeros((B, self.seq_len),               dtype=np.float32)
                phase_np  = np.full((B, self.seq_len), 2,             dtype=np.int64)    # default: maintenance
                stim_np   = np.zeros((B, self.seq_len),               dtype=np.float32)
                valid     = np.zeros(B, dtype=bool)

                for b, d in enumerate(gdata):
                    start = ci * self.seq_len
                    end   = start + self.seq_len
                    nw    = len(d["labels"])
                    if start >= nw:
                        continue
                    actual_end = min(end, nw)
                    sl = actual_end - start

                    w  = d["waves"][start:actual_end].copy()
                    fe = d["features"][start:actual_end].copy()
                    s  = d["sqi"][start:actual_end].copy()
                    lb = d["labels"][start:actual_end].copy()

                    # Augmentation
                    if self.noise_std > 0:
                        noise = np.random.randn(*w.shape).astype(np.float32) * self.noise_std
                        scale = np.float32(np.random.uniform(0.85, 1.15))
                        w = w * scale + noise
                    if n_ch == 2 and np.random.rand() < 0.5:
                        w = w[:, [1, 0], :]

                    waves_np[b, :sl] = w
                    feat_np[b,  :sl] = fe
                    sqi_np[b,   :sl] = s
                    lab_np[b,   :sl] = lb / 100.0   # normalise [0,1]

                    # 修复：加载多任务标签
                    if has_mt and "phases" in d:
                        phase_np[b, :sl] = d["phases"][start:actual_end]
                        stim_np[b,  :sl] = d["stim_events"][start:actual_end]

                    valid[b] = True

                if not valid.any():
                    continue

                wave_t  = torch.from_numpy(waves_np).to(self.device, non_blocking=True)
                feat_t  = torch.from_numpy(feat_np).to(self.device,  non_blocking=True)
                sqi_t   = torch.from_numpy(sqi_np).to(self.device,   non_blocking=True)
                lab_t   = torch.from_numpy(lab_np).to(self.device,   non_blocking=True)
                phase_t = torch.from_numpy(phase_np).to(self.device, non_blocking=True)
                stim_t  = torch.from_numpy(stim_np).to(self.device,  non_blocking=True)

                # ── Forward（修复：正确处理V1和V2的返回值数量）──────────────
                with torch.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.use_amp):
                    if self.model_version == "v2":
                        # V2 returns: pred_bis, phase_logits, stim_logits, correction, h_new
                        pred_bis, phase_logits, stim_logits, _corr, h_new = \
                            self.model(wave_t, feat_t, sqi_t, hx=h)
                    else:
                        # V1 returns: pred, pred_seq, h_new
                        pred, pred_seq, h_new = self.model(wave_t, feat_t, sqi_t, hx=h)

                # Keep hidden state only for valid (non-exhausted) patients
                # 使用 LNNCore.mask_state() — 统一处理 GRU/CfC 形状差异
                h = h_new.detach()
                if not valid.all():
                    valid_mask = torch.from_numpy(valid).to(self.device)
                    h = self.model.temporal.mask_state(h, valid_mask)

                # ── Loss ─────────────────────────────────────────────────────
                valid_t  = torch.from_numpy(valid).to(self.device)
                sqi_mean = sqi_t[valid_t].mean(dim=-1)               # (Bv, T)

                with torch.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.use_amp):
                    if self.model_version == "v2":
                        # MultiTaskLoss 接口
                        label_seq_v = lab_t[valid_t]           # (Bv, T)
                        losses = self.criterion(
                            pred_bis[valid_t],
                            phase_logits[valid_t],
                            stim_logits[valid_t],
                            label_seq_v,
                            phase_t[valid_t],
                            stim_t[valid_t],
                            sqi_mean,
                        )
                        loss = losses["loss"]
                        base_val = losses["bis_loss"].item()
                        mono_val = losses["mono_loss"].item()
                    else:
                        # AnesthesiaLoss (V1) 接口
                        last_lab   = lab_t[valid_t, -1]
                        last_pred  = pred[valid_t]
                        label_seq_v = lab_t[valid_t]
                        pred_seq_v  = pred_seq[valid_t]
                        losses = self.criterion(
                            last_pred, last_lab, sqi_mean.mean(-1),
                            pred_seq=pred_seq_v, label_seq=label_seq_v)
                        loss = losses["loss"]
                        base_val = losses.get("base", 0.0)
                        mono_val = losses.get("monotonic", 0.0)
                        if isinstance(base_val, torch.Tensor):
                            base_val = base_val.item()
                        if isinstance(mono_val, torch.Tensor):
                            mono_val = mono_val.item()

                # ── Backward ─────────────────────────────────────────────────
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

                bv = int(valid.sum())
                total_loss += loss.item() * bv
                total_base += base_val * bv
                total_mono += mono_val * bv
                n_windows  += bv

            if bar is not None:
                gpu_mb = (torch.cuda.memory_allocated(self.device) / 1e6
                          if self.device.type == "cuda" else 0.0)
                bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mono=f"{mono_val:.4f}",
                    gpu=f"{gpu_mb:.0f}MB",
                )
                bar.update(1)

        if bar is not None:
            bar.close()

        epoch_time = time.perf_counter() - t0
        n_windows = max(n_windows, 1)

        return {
            "train_loss":   total_loss / n_windows,
            "train_base":   total_base / n_windows,
            "train_mono":   total_mono / n_windows,
            "time_data_s":  0.0,
            "time_fwd_s":   epoch_time,
            "time_bwd_s":   0.0,
            "epoch_time_s": epoch_time,
            "throughput":   n_windows / epoch_time,
        }
