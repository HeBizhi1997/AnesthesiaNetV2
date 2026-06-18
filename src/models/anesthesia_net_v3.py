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


class SharedBISHead(nn.Module):
    """
    单一共享 BIS 回归头 + 软相位条件（v14）。

    相比 PhaseGatedBISHead（4 个硬专家 × softmax 门控）的优势：
      - 所有数据训练同一个头，不会让 induction(0.3%)/recovery(1.5%) 专家饿死。
      - phase 仅作为软特征拼入输入，对过渡区做轻度调制而非硬切换。
      v13 诊断：硬门控头在过渡区门控模糊 → 诱导/复苏 MAE ~9-10（整体 2×）。
    """

    def __init__(self, in_dim: int, n_phases: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + n_phases, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, phase_probs: torch.Tensor) -> torch.Tensor:
        """h:(B,T,D) phase_probs:(B,T,4) → pred:(B,T,1)"""
        return self.net(torch.cat([h, phase_probs], dim=-1))


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
        # v14 新增
        bis_head: str          = "gated",   # "gated"(v13 硬专家) | "shared"(v14 软条件)
        norm_type: str         = "batch",   # "batch" | "group"(消除 eval running-stat 噪声)
        enable_multimodal: bool = True,     # False=不构建 PK/Vital/蒸馏（EEG-only 训练）
        # v3 新增：多模态教师网络参数
        pk_hidden: int         = 64,
        vital_hidden: int      = 64,
        d_proj: int            = 64,   # 蒸馏投影维度
        # v16 新增
        distill_mode: str      = "byol",    # "byol"(特征对齐,已验证无效) | "response"(响应蒸馏)
        bis_uncertainty: bool  = False,     # True=输出异方差不确定性(Laplace 尺度 log b)
        # v17 新增
        uncertainty_phase_cond: bool = False,  # logvar 头额外接相位软概率（修 recovery 过度自信）
    ):
        super().__init__()
        cnn_channels = cnn_channels or [32, 64, 128]
        self.enable_multimodal = enable_multimodal
        self.distill_mode      = distill_mode
        self.bis_uncertainty   = bis_uncertainty
        self.uncertainty_phase_cond = uncertainty_phase_cond

        # ── EEG 编码器（复用 v2 WaveformEncoder）────────────────────────────
        self.wave_enc = WaveformEncoder(
            n_channels=n_eeg_channels,
            window_samples=window_samples,
            conv_channels=cnn_channels,
            kernel_size=cnn_kernel,
            global_pool=True,
            bsr_layer=bsr_layer,
            use_grad_checkpoint=grad_checkpoint,
            norm_type=norm_type,
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
        if bis_head == "shared":
            self.bis_head = SharedBISHead(d_model, self.N_PHASES)
        else:
            self.bis_head = PhaseGatedBISHead(d_model, self.N_PHASES)

        # ── v16：异方差不确定性头（Laplace 尺度 log b）────────────────────────
        # 与 bis_head 解耦的独立 MLP，从 h_seq 直接回归 log b（对数尺度参数）。
        # 临床意义：诱导/复苏等过渡区天然高不确定性，预测带可信区间比单点更有价值；
        # 训练上 Laplace NLL 自动按不确定度调节各时步权重，对准高误差过渡段。
        if bis_uncertainty:
            # v17：可选接入相位软概率，让不确定度头能学到 recovery/induction 更难
            logvar_in = d_model + (self.N_PHASES if uncertainty_phase_cond else 0)
            self.logvar_head = nn.Sequential(
                nn.Linear(logvar_in, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.logvar_head = None

        # ── v3 多模态教师网络（仅 enable_multimodal=True 时构建）────────────
        # v14 诊断：Phase-3 BYOL 蒸馏对 BIS MAE 无增益且拖垮 StAUC（见 training_history）。
        # v16：distill_mode="response" 改为响应蒸馏——教师头直接预测 BIS（强监督），
        #      学生 pred_bis 被拉向教师的（detach）软预测，替代失效的特征对齐。
        # EEG-only 训练时置 None，节省参数且 forward 安全跳过。
        if enable_multimodal:
            self.pk_enc     = PKEncoder(in_dim=6, hidden=pk_hidden, d_pk=pk_hidden)
            self.vital_enc  = VitalEncoder(in_dim=5, hidden=vital_hidden, d_v=vital_hidden)
            self.pkd_head   = PKDHead(d_pk=pk_hidden)
            self.vitald_head = VitalDHead(d_vital=vital_hidden)  # v3 fix: gives VitalEncoder gradient
            if distill_mode == "byol":
                self.distill = CrossModalDistillation(
                    d_student=d_model,
                    d_pk=pk_hidden,
                    d_vital=vital_hidden,
                    d_proj=d_proj,
                )
            else:
                # response 模式无投影头；教师融合为算术加权，无额外参数
                self.distill = None
        else:
            self.pk_enc = self.vital_enc = self.pkd_head = None
            self.vitald_head = self.distill = None

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
        use_vital: bool = False,  # skip VitalEncoder when vitals loss λ=0
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict 包含：
          pred_bis       : (B, T, 1)   推理输出
          phase_logits   : (B, T, 4)
          stim_logits    : (B, T, 1)
          h              : 隐状态（用于流式推理）
          --- 以下仅在训练时（drug_ce 非 None）返回 ---
          bis_pkd        : (B, T, 1)   PK 辅助 BIS 预测
          bis_vitald     : (B, T, 1)   Vital 辅助 BIS 预测（v3 fix）
          loss_distill_pk    : 标量
          loss_distill_vital : 标量
          reg_distill_pk     : 标量（教师方差正则化）
          reg_distill_vital  : 标量（教师方差正则化）
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

        # SQI 惰性模式 — 输出级保持（非递归状态冻结）
        # 注意：GRU 的递归隐状态仍会处理坏 SQI 窗口；此处只在「输出」层面，
        #   将坏 SQI 时步的 h_seq 替换为最近一个 sqi_ok=True 时步的输出，
        #   从而冻结下游 BIS/相位预测，避免噪声窗口污染显示值。
        # 向量化实现（v10）：last_src[b,t] = 最近一个 sqi_ok 为 True 的时步索引
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

        # v16：异方差不确定性（Laplace log b，clamp 防数值爆炸）
        # v17：可选把相位软概率拼入输入，使 recovery/induction 能学到更高不确定度
        if self.logvar_head is not None:
            logvar_in = (torch.cat([h_seq, phase_probs], dim=-1)
                         if self.uncertainty_phase_cond else h_seq)
            pred_logb = self.logvar_head(logvar_in).clamp(-6.0, 2.0)  # (B, T, 1)
            out["pred_logvar"] = pred_logb

        # ── 多模态训练分支（推理时 / enable_multimodal=False 时跳过）────────
        if drug_ce is not None and self.pk_enc is not None:
            h_pk    = self.pk_enc(drug_ce)                      # (B, T, d_pk)
            bis_pkd = self.pkd_head(h_pk)                       # (B, T, 1)
            out["bis_pkd"] = bis_pkd

            mask_d = mask_drug if mask_drug is not None else torch.ones(B, T, device=h_seq.device)

            # VitalEncoder — 仅在 use_vital=True 时运行
            if vitals is not None and use_vital:
                h_vital    = self.vital_enc(vitals)             # (B, T, d_vital)
                bis_vitald = self.vitald_head(h_vital)          # (B, T, 1)
                out["bis_vitald"] = bis_vitald
                mask_v = mask_vital if mask_vital is not None else torch.ones(B, T, device=h_seq.device)
            else:
                h_vital = None
                mask_v = None

            if self.distill_mode == "response":
                # ── v16 响应蒸馏：融合教师 BIS 预测（逐时步、按可用性掩码加权）──
                # 教师头(pkd/vitald)由 L_pkd/L_vitald 用真实 BIS 直接监督 → 是强教师；
                # 学生 pred_bis 在 loss 中被拉向 bis_teacher.detach()（梯度只入 EEG 学生）。
                w_d = mask_d.unsqueeze(-1)                      # (B, T, 1)
                t_num = bis_pkd * w_d
                t_den = w_d.clone()
                if h_vital is not None:
                    w_v = mask_v.unsqueeze(-1)
                    t_num = t_num + bis_vitald * w_v
                    t_den = t_den + w_v
                bis_teacher = t_num / t_den.clamp(min=1e-6)     # (B, T, 1)
                out["bis_teacher"] = bis_teacher
                out["mask_resp"]   = (t_den.squeeze(-1) > 1e-6).float()  # (B, T)
            else:
                # ── BYOL 特征对齐（保留兼容；已验证无效，v16 默认不用）──────
                loss_pk, loss_vital, reg_pk, reg_vital = self.distill(
                    h_seq, h_pk, h_vital, mask_d, mask_v
                )
                out["loss_distill_pk"]    = loss_pk
                if loss_vital is not None:
                    out["loss_distill_vital"] = loss_vital
                    out["reg_distill_vital"]  = reg_vital
                out["reg_distill_pk"]     = reg_pk

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
            bis_head=m.get("bis_head", "gated"),
            norm_type=m.get("norm_type", "batch"),
            enable_multimodal=m.get("enable_multimodal", True),
            pk_hidden=m.get("pk_hidden", 64),
            vital_hidden=m.get("vital_hidden", 64),
            d_proj=m.get("d_proj", 64),
            distill_mode=m.get("distill_mode", "byol"),
            bis_uncertainty=m.get("bis_uncertainty", False),
            uncertainty_phase_cond=m.get("uncertainty_phase_cond", False),
        )
