"""
anesthesia_net_v3.py — MERIDIAN（AnesthesiaNetV3）

Multi-modal EEG Research Integration for Depth Interpretation in ANesthesia

训练时：EEG + Drug CE + Vitals → 跨模态蒸馏 + 多任务预测
推理时：EEG only → BIS + 相位 + 刺激（与 v2 接口完全兼容）

架构图：
  ┌──────────────────────────────────────────────────────────────┐
  │ TRAINING                                                      │
  │                                                               │
  │  EEG(B,T,n_ch,W)─→WaveEncoder─→FeatFusion─→GRU─→h_eeg ──┐  │
  │                                                            │  │
  │  drug_ce(B,T,6)─→PKEncoder──→h_pk──┐   ┌──proj_t_pk      │  │
  │                              stop-grad  │               (蒸馏)│
  │  vitals(B,T,5)──→VitalEncoder→h_vit─┘   └──proj_t_vital   │  │
  │                                                            │  │
  │  h_eeg──→proj_s_pk────→L_distill_pk                       │  │
  │  h_eeg──→proj_s_vital─→L_distill_vital                    │  │
  │                                                            │  │
  │  h_pk──→PKD_Head──→bis_pkd──→L_pkd(vs true BIS)          │  │
  │                                                            │  │
  │  h_eeg──→PhaseHead──→phase                                │  │
  │         ──→StimHead──→stim  (CV 标签监督)                  │  │
  │         ──→BISHead───→bis   (phase-gated)                 │  │
  └──────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────┐
  │ INFERENCE（仅 EEG）                  │
  │                                     │
  │ EEG─→WaveEncoder─→FeatFusion─→GRU  │
  │       ─→Phase/Stim/BIS heads        │
  └─────────────────────────────────────┘
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import WaveformEncoder
from .lnn_core import LNNCore
from .pk_encoder import PKEncoder
from .vital_encoder import VitalEncoder
from .distillation import CrossModalDistillation


# ─────────────────────────────────────────────────────────────────────────────
# 任务输出头（复用 v2 设计）
# ─────────────────────────────────────────────────────────────────────────────

class PhaseGatedBISHead(nn.Module):
    """相位条件 BIS 回归头（相位软混合）。"""

    def __init__(self, in_dim: int, n_phases: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Linear(in_dim // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_phases)
        ])

    def forward(self, h: torch.Tensor, phase_probs: torch.Tensor) -> torch.Tensor:
        """h:(B,T,D) phase_probs:(B,T,4) → pred:(B,T,1)"""
        preds = torch.stack([head(h).squeeze(-1) for head in self.heads], -1)
        return (preds * phase_probs).sum(-1, keepdim=True)


class PKDHead(nn.Module):
    """
    PK/PD 辅助回归头（训练时教师侧）。

    从 h_pk（药物编码）直接预测 BIS，学习个体化 CE50。
    不使用 Hill 方程固定参数（见理论文档 §9.1 修正）。
    """

    def __init__(self, d_pk: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_pk, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_pk: torch.Tensor) -> torch.Tensor:
        """h_pk:(B,T,d_pk) → bis_pkd:(B,T,1) ∈ [0,1]"""
        return self.net(h_pk)


class VitalDHead(nn.Module):
    """
    Vitals 辅助回归头（训练时教师侧，与 PKDHead 对称设计）。

    从 h_vital（生命体征编码）直接预测 BIS，赋予 VitalEncoder 梯度信号。
    临床依据：MAP/HR 与麻醉深度存在非线性相关（低 BIS → 低 MAP/HR）。

    修复 v9 的设计缺陷：VitalEncoder 在 v9 中无任何损失直接督导，
    导致 h_vital 保持随机初始化，wDV 占 Phase 3 损失 69% 为噪声。
    """

    def __init__(self, d_vital: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_vital, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_vital: torch.Tensor) -> torch.Tensor:
        """h_vital:(B,T,d_vital) → bis_vitald:(B,T,1) ∈ [0,1]"""
        return self.net(h_vital)


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class AnesthesiaNetV3(nn.Module):
    """
    MERIDIAN — 多模态麻醉深度监测模型（训练+推理分离设计）。

    Tasks:
      1. Phase classification  (4 classes)
      2. Stimulation detection (CV-label supervised)
      3. BIS regression        (phase-gated)
      4. PK-supervised BIS     (auxiliary, training only)
      5. Cross-modal distill   (EEG ↔ PK, EEG ↔ Vitals, training only)
    """

    N_PHASES = 4

    def __init__(
        self,
        # EEG encoder 参数（与 v2 一致）
        n_eeg_channels: int    = 2,
        cnn_channels: list     = None,
        cnn_kernel: int        = 7,
        window_samples: int    = 512,
        feature_dim: int       = 24,
        d_model: int           = 128,
        gru_layers: int        = 2,
        gru_dropout: float     = 0.15,
        lnn_backend: str       = "gru",
        sqi_inertia_threshold: float = 0.5,
        bsr_layer: bool        = True,
        grad_checkpoint: bool  = True,
        # v3 新增：多模态教师网络参数
        pk_hidden: int         = 64,
        vital_hidden: int      = 64,
        d_proj: int            = 64,   # 蒸馏投影维度
    ):
        super().__init__()
        cnn_channels = cnn_channels or [32, 64, 128]

        # ── EEG 编码器（复用 v2 WaveformEncoder）────────────────────────────
        self.wave_enc = WaveformEncoder(
            n_channels=n_eeg_channels,
            window_samples=window_samples,
            conv_channels=cnn_channels,
            kernel_size=cnn_kernel,
            global_pool=True,
            bsr_layer=bsr_layer,
            use_grad_checkpoint=grad_checkpoint,
        )
        cnn_out = self.wave_enc.out_dim   # 128

        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim + n_eeg_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ── 时序模型（GRU）────────────────────────────────────────────────
        self.temporal = LNNCore(
            input_dim=d_model,
            units=d_model,
            return_sequences=True,
            backend=lnn_backend,
            num_layers=gru_layers,
            dropout=gru_dropout,
        )
        self.sqi_inertia_threshold = sqi_inertia_threshold

        # ── 任务输出头 ────────────────────────────────────────────────────
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, self.N_PHASES),
        )
        self.stim_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 1),
        )
        self.bis_head = PhaseGatedBISHead(d_model, self.N_PHASES)

        # ── v3 新增：多模态教师网络（训练时使用，推理时忽略）──────────────
        self.pk_enc     = PKEncoder(in_dim=6, hidden=pk_hidden, d_pk=pk_hidden)
        self.vital_enc  = VitalEncoder(in_dim=5, hidden=vital_hidden, d_v=vital_hidden)
        self.pkd_head   = PKDHead(d_pk=pk_hidden)
        self.vitald_head = VitalDHead(d_vital=vital_hidden)   # v3 fix: gives VitalEncoder gradient

        self.distill = CrossModalDistillation(
            d_student=d_model,
            d_pk=pk_hidden,
            d_vital=vital_hidden,
            d_proj=d_proj,
        )

        self.d_model = d_model

    # ── Forward（训练+推理统一接口）──────────────────────────────────────────

    def forward(
        self,
        wave: torch.Tensor,           # (B, T, n_ch, win_samp)
        features: torch.Tensor,       # (B, T, n_feat)
        sqi: torch.Tensor,            # (B, T, n_ch)
        hx: Optional[torch.Tensor] = None,
        # v3 新增（仅训练时提供，推理时传 None）
        drug_ce: Optional[torch.Tensor]  = None,   # (B, T, 6)
        vitals: Optional[torch.Tensor]   = None,   # (B, T, 5)
        mask_drug: Optional[torch.Tensor]  = None, # (B, T)
        mask_vital: Optional[torch.Tensor] = None, # (B, T)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict 包含：
          pred_bis       : (B, T, 1)   推理输出
          phase_logits   : (B, T, 4)
          stim_logits    : (B, T, 1)
          h              : 隐状态（用于流式推理）
          --- 以下仅在训练时（drug_ce 非 None）返回 ---
          bis_pkd        : (B, T, 1)   PK 辅助 BIS 预测
          loss_distill_pk    : 标量
          loss_distill_vital : 标量
        """
        B, T, n_ch, W = wave.shape

        # ── EEG 编码 ──────────────────────────────────────────────────────
        wave_flat = wave.view(B * T, n_ch, W)
        wave_emb  = self.wave_enc(wave_flat)                    # (B*T, 128)

        feat_flat = features.view(B * T, -1)
        sqi_flat  = sqi.view(B * T, -1)
        feat_emb  = self.feat_proj(torch.cat([feat_flat, sqi_flat], -1))

        fused = self.fusion(torch.cat([wave_emb, feat_emb], -1))
        seq   = fused.view(B, T, -1)                            # (B, T, d_model)

        # ── 时序模型 ──────────────────────────────────────────────────────
        h_seq, h = self.temporal(seq, hx)                       # (B, T, d_model)

        # SQI 惰性模式 — 向量化实现（v10 优化，消除 T-1 次 Python→CUDA 同步）
        # 原理：last_src[b,t] = 最近一个 sqi_ok 为 True 的时步索引
        # 等价于对 pos_masked = (t if sqi_ok[b,t] else -1) 做 cummax
        if self.sqi_inertia_threshold > 0.0 and T > 1:
            sqi_ok = (sqi.mean(-1) >= self.sqi_inertia_threshold)   # (B, T)
            with torch.no_grad():
                pos = torch.arange(T, device=h_seq.device).unsqueeze(0).expand(B, -1)  # (B, T)
                pos_masked = torch.where(sqi_ok, pos, pos.new_full((), -1))             # (B, T)
                last_src = torch.cummax(pos_masked, dim=1).values.clamp(min=0)          # (B, T)
                idx = last_src.unsqueeze(-1).expand(-1, -1, h_seq.shape[-1])
            h_seq = h_seq.gather(1, idx)

        # ── 任务输出头 ────────────────────────────────────────────────────
        phase_logits = self.phase_head(h_seq)                   # (B, T, 4)
        stim_logits  = self.stim_head(h_seq)                    # (B, T, 1)
        phase_probs  = F.softmax(phase_logits, dim=-1)
        pred_bis     = self.bis_head(h_seq, phase_probs)        # (B, T, 1)

        out = {
            "pred_bis":     pred_bis,
            "phase_logits": phase_logits,
            "stim_logits":  stim_logits,
            "h":            h,
        }

        # ── 多模态训练分支（推理时跳过）─────────────────────────────────────
        if drug_ce is not None:
            h_pk    = self.pk_enc(drug_ce)                      # (B, T, d_pk)
            bis_pkd = self.pkd_head(h_pk)                       # (B, T, 1)
            out["bis_pkd"] = bis_pkd

            if vitals is not None:
                h_vital    = self.vital_enc(vitals)             # (B, T, d_vital)
                bis_vitald = self.vitald_head(h_vital)          # (B, T, 1)  ← v3 fix
                out["bis_vitald"] = bis_vitald

                mask_d  = mask_drug  if mask_drug  is not None else torch.ones(B, T, device=h_seq.device)
                mask_v  = mask_vital if mask_vital is not None else torch.ones(B, T, device=h_seq.device)

                loss_pk, loss_vital = self.distill(
                    h_seq, h_pk, h_vital, mask_d, mask_v
                )
                out["loss_distill_pk"]    = loss_pk
                out["loss_distill_vital"] = loss_vital

        return out

    @classmethod
    def from_config(cls, cfg: dict) -> "AnesthesiaNetV3":
        m   = cfg["model"]
        t   = cfg["training"]
        eeg = cfg["eeg"]
        w   = cfg["windowing"]
        win_samp = int(w["window_sec"] * eeg["srate"])
        return cls(
            n_eeg_channels=len(eeg["channels"]),
            cnn_channels=m.get("cnn_channels", [32, 64, 128]),
            cnn_kernel=m.get("cnn_kernel", 7),
            window_samples=win_samp,
            feature_dim=m.get("feature_dim", 24),
            d_model=m.get("d_model", 128),
            gru_layers=m.get("gru_layers", 2),
            gru_dropout=m.get("gru_dropout", 0.15),
            lnn_backend=m.get("lnn_backend", "gru"),
            sqi_inertia_threshold=m.get("sqi_inertia_threshold", 0.5),
            bsr_layer=m.get("bsr_layer", True),
            grad_checkpoint=m.get("grad_checkpoint", True),
            pk_hidden=m.get("pk_hidden", 64),
            vital_hidden=m.get("vital_hidden", 64),
            d_proj=m.get("d_proj", 64),
        )
