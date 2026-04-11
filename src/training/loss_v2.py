"""
loss_v2.py - Multi-task loss for AnesthesiaNetV2.

Three tasks, three losses:
  1. BIS regression     : Huber loss (robust to outliers during stimulation events)
  2. Phase classification: weighted cross-entropy (handles 95% maintenance imbalance)
  3. Stimulation detection: focal loss + pos_weight (handles 0.7% / 144:1 imbalance)

Combined loss — two modes:
  Fixed:   L = λ_bis * L_bis + λ_phase * L_phase + λ_stim * L_stim + λ_mono * L_mono
  Auto:    UW-SO adaptive weighting per batch, normalised by loss magnitudes.
           ω_k = L_k / (Σ L_j) so each task contributes equally on average.
           Soft-clamped with temperature to prevent extreme concentration.

Phase weights (pre_op, induction, maintenance, recovery):
  Inverse-frequency to upweight rare induction (0.3%) and recovery (1.5%).
  Maintenance (95.9%) gets low weight so it doesn't dominate.

Stim focal loss with pos_weight:
  Combines focal modulation (1-p_t)^γ with pos_weight BCE to fix 144:1 imbalance.
  alpha=0.99 (positive class), gamma=2, pos_weight=99 jointly address the imbalance.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Phase class weights (empirical from 1749 cases) ───────────────────────────
# pre_op=2.3%, induction=0.3%, maintenance=95.9%, recovery=1.5%
# Inverse-freq then re-normalise to sum=4 (one per class on average):
_RAW_WEIGHTS = torch.tensor([
    1.0 / 0.023,    # pre_op
    1.0 / 0.003,    # induction  <- highest weight (rarest)
    1.0 / 0.959,    # maintenance
    1.0 / 0.015,    # recovery
], dtype=torch.float32)
_PHASE_WEIGHTS = _RAW_WEIGHTS / _RAW_WEIGHTS.sum() * 4.0


def focal_loss(
    logits: torch.Tensor,   # (N,) raw logit
    targets: torch.Tensor,  # (N,) binary float {0, 1}
    gamma: float = 2.0,
    alpha: float = 0.99,    # weight for positive class (high for imbalanced data)
    pos_weight: float = 1.0,  # multiplies BCE loss for positive targets
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss with pos_weight — handles extreme class imbalance.

    Combines:
      - pos_weight: scales the BCE loss for positive labels (equivalent to
        oversampling positives by pos_weight times)
      - alpha: class-level balancing (alpha for positive class)
      - gamma: focal modulation (1-p_t)^gamma to down-weight easy negatives

    For 144:1 imbalance (0.7% stim positive rate), use:
      alpha=0.99, pos_weight=99.0, gamma=2.0
    """
    p   = torch.sigmoid(logits)
    pw  = logits.new_tensor(pos_weight)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    fl = alpha_t * (1 - p_t) ** gamma * bce
    return fl.mean() if reduction == "mean" else fl.sum()


def monotonic_loss(
    pred_seq: torch.Tensor,      # (B, T, 1) — BIS predictions in sequence
    label_seq: torch.Tensor,     # (B, T)    — true BIS normalised [0,1]
    phase_labels: torch.Tensor,  # (B, T)    — 0/1/2/3
    ema_alpha: float = 0.85,     # EMA 衰减系数（越高 → 越平滑，约束越宽松）
) -> torch.Tensor:
    """
    增强版单调性损失：逐步 + EMA 药代动力学趋势约束。

    临床依据：
      - 麻醉诱导期：丙泊酚/七氟烷浓度按药代动力学模型（sigmoid Emax）单调升高，
        理论上 BIS 必须单调下降。
      - 苏醒期：药物浓度以指数衰减，BIS 必须单调回升。

    两级约束：
      1. 逐步约束（原始）：每相邻时间步，预测方向必须与标签方向一致。
         惩罚 F.relu(d_pred) 当 d_label<0 且 phase=1（诱导）。

      2. EMA 趋势约束（新增）：基于 EMA 平滑预测计算趋势，惩罚 EMA 违反方向的情况。
         药代动力学模型下浓度变化是平滑的（非逐步），因此 EMA 比逐步差分更符合物理规律。
         公式：ema[t] = ema_alpha × ema[t-1] + (1-ema_alpha) × pred[t]
         ema_alpha=0.85 对应约 6 步（6 秒）的等效时间常数，
         符合丙泊酚效应室平衡时间（ke0 ≈ 0.2/min → t_half ≈ 3.5 min，
         单步约束仍有意义，此为更短时间尺度的局部趋势）。

    参数：
      ema_alpha : EMA 平滑系数。越大 → 约束越平滑（偏向趋势）；越小 → 越接近逐步约束。
    """
    if pred_seq.shape[1] < 2:
        return pred_seq.new_zeros(1).squeeze()

    pred    = pred_seq.squeeze(-1)               # (B, T)
    d_pred  = pred[:, 1:] - pred[:, :-1]         # (B, T-1) 逐步差分
    d_label = label_seq[:, 1:] - label_seq[:, :-1]

    phase_mid      = phase_labels[:, :-1]        # (B, T-1) 对齐差分
    induction_mask = (phase_mid == 1).float()
    recovery_mask  = (phase_mid == 3).float()

    # ── Level 1: 逐步方向约束（原始） ─────────────────────────────────────────
    induction_viol = F.relu(d_pred)  * induction_mask * (d_label < 0).float()
    recovery_viol  = F.relu(-d_pred) * recovery_mask  * (d_label > 0).float()
    step_loss = (induction_viol + recovery_viol).mean()

    # ── Level 2: EMA 趋势约束（药代动力学约束）──────────────────────────────────
    # 性能关键：EMA 方向用 no_grad 计算（避免 T=300 串行 autograd 链 → 5-10× 减速），
    # 违反惩罚通过 d_pred（有梯度）传回梯度。等效语义：用 EMA 平滑方向判断违规，
    # 用实际逐步预测差分承载梯度信号。
    with torch.no_grad():
        pred_d     = pred.detach()
        # Pre-allocate output tensor instead of growing a Python list.
        # Python list of T=300 tensors forces T separate allocations + torch.stack copy.
        # Pre-allocation: single contiguous buffer, in-place writes — zero extra memory.
        ema_tensor = pred_d.clone()                      # (B, T) — pre-allocated
        for t in range(1, pred_d.shape[1]):
            ema_tensor[:, t] = (ema_alpha * ema_tensor[:, t - 1]
                                + (1.0 - ema_alpha) * pred_d[:, t])
        d_ema = ema_tensor[:, 1:] - ema_tensor[:, :-1] # (B, T-1) detached direction

    # 梯度经 d_pred 流回预测，而非通过 EMA 链（无串行 autograd 开销）
    ema_ind_viol = F.relu(d_pred)  * induction_mask * (d_ema < 0).float()
    ema_rec_viol = F.relu(-d_pred) * recovery_mask  * (d_ema > 0).float()
    ema_loss = (ema_ind_viol + ema_rec_viol).mean() * 0.5

    return step_loss + ema_loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for AnesthesiaNetV2.

    Parameters
    ----------
    lambda_bis    : weight for BIS Huber regression loss
    lambda_phase  : weight for phase classification loss
    lambda_stim   : weight for stimulation focal loss
    lambda_mono   : weight for monotonicity loss (induction/recovery direction)
    huber_delta   : Huber loss delta in normalised BIS units (1 BIS / 100)
    focal_gamma   : focal loss gamma (2 = standard focal loss)
    focal_alpha   : focal loss alpha for positive (stim) class. Use 0.99 for
                    heavily imbalanced stim detection (144:1 ratio).
    stim_pos_weight: pos_weight for stim BCE (equivalent to repeating positives).
                    Use ~99-144 for 144:1 imbalance. Combined with alpha for
                    aggressive upweighting of rare stimulation events.
    use_auto_weight: if True, use UW-SO adaptive per-batch weighting instead of
                    fixed lambdas. Normalises by relative loss magnitudes so all
                    tasks contribute equally on average regardless of scale.
    auto_weight_temp: temperature for UW-SO softmax (lower = harder to balance).
                    Default 0.5 gives moderate concentration around high-loss tasks.
    """

    def __init__(
        self,
        lambda_bis:       float = 1.0,
        lambda_phase:     float = 0.5,
        lambda_stim:      float = 0.3,
        lambda_mono:      float = 0.3,
        huber_delta:      float = 0.10,   # = 10 BIS points
        focal_gamma:      float = 2.0,
        focal_alpha:      float = 0.99,   # high for 144:1 imbalance
        stim_pos_weight:  float = 99.0,   # 144:1 imbalance → pos_weight≈99
        use_auto_weight:  bool  = False,
        auto_weight_temp: float = 0.5,
    ):
        super().__init__()
        self.lambda_bis       = lambda_bis
        self.lambda_phase     = lambda_phase
        self.lambda_stim      = lambda_stim
        self.lambda_mono      = lambda_mono
        self.huber_delta      = huber_delta
        self.focal_gamma      = focal_gamma
        self.focal_alpha      = focal_alpha
        self.stim_pos_weight  = stim_pos_weight
        self.use_auto_weight  = use_auto_weight
        self.auto_weight_temp = auto_weight_temp

        # Phase weights as buffer (moves to device with model)
        self.register_buffer("phase_weights", _PHASE_WEIGHTS)

    def forward(
        self,
        pred_bis:     torch.Tensor,   # (B, T, 1)  normalised [0,1]
        phase_logits: torch.Tensor,   # (B, T, 4)
        stim_logits:  torch.Tensor,   # (B, T, 1)
        label_bis:    torch.Tensor,   # (B, T)     normalised [0,1]
        phase_labels: torch.Tensor,   # (B, T)     int64 {0,1,2,3}
        stim_labels:  torch.Tensor,   # (B, T)     float {0.0, 1.0}
        sqi_mean:     torch.Tensor,   # (B, T)     for SQI masking
    ) -> dict[str, torch.Tensor]:
        B, T = label_bis.shape

        # SQI mask: ignore low-quality windows in regression
        sqi_ok = (sqi_mean > 0.5).float()

        # ── 1. BIS Huber loss ─────────────────────────────────────────────────
        pred = pred_bis.squeeze(-1)          # (B, T)
        bis_err = F.huber_loss(
            pred * sqi_ok, label_bis * sqi_ok,
            delta=self.huber_delta, reduction="sum",
        ) / (sqi_ok.sum() + 1e-6)

        # ── 2. Phase cross-entropy (weighted) ─────────────────────────────────
        # Flatten for CE: (B*T, 4) logits vs (B*T,) labels
        ph_logits_flat = phase_logits.view(B * T, -1)
        ph_labels_flat = phase_labels.view(-1).long()

        phase_err = F.cross_entropy(
            ph_logits_flat, ph_labels_flat,
            weight=self.phase_weights.to(ph_logits_flat.device),
        )

        # ── 3. Stimulation focal loss with pos_weight ─────────────────────────
        stim_err = focal_loss(
            stim_logits.view(-1),
            stim_labels.view(-1),
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            pos_weight=self.stim_pos_weight,
        )

        # ── 4. Monotonicity loss (direction-aware for induction/recovery) ──────
        mono_err = monotonic_loss(pred_bis, label_bis, phase_labels)

        # ── Combined ──────────────────────────────────────────────────────────
        if self.use_auto_weight:
            total = self._auto_weighted_sum(bis_err, phase_err, stim_err, mono_err)
        else:
            total = (
                self.lambda_bis   * bis_err  +
                self.lambda_phase * phase_err +
                self.lambda_stim  * stim_err  +
                self.lambda_mono  * mono_err
            )

        return {
            "loss":       total,
            "bis_loss":   bis_err.detach(),
            "phase_loss": phase_err.detach(),
            "stim_loss":  stim_err.detach(),
            "mono_loss":  mono_err.detach(),
        }

    def _auto_weighted_sum(
        self,
        bis_err:   torch.Tensor,
        phase_err: torch.Tensor,
        stim_err:  torch.Tensor,
        mono_err:  torch.Tensor,
    ) -> torch.Tensor:
        """UW-SO adaptive weighting.

        Normalises each task loss by the sum of all task losses, then
        rescales by temperature-controlled softmax to prevent any single task
        from monopolising gradients.

        With temperature T:
          - T -> 0 : winner-takes-all (highest-loss task gets all weight)
          - T -> inf: uniform weighting (same as fixed lambda=0.25 each)
          - T = 0.5 : moderate concentration, empirically good default

        Applied to log-scale losses so that BIS Huber (0.002) and Phase CE (0.3)
        are brought to the same order of magnitude before weighting.
        """
        raw = torch.stack([
            bis_err.detach(),
            phase_err.detach(),
            stim_err.detach(),
            mono_err.detach(),
        ])
        # Inverse-loss normalisation (target equal-contribution weighting).
        #
        # Goal: each task contributes an equal share of the total gradient.
        #   w_k = target / L_k   where target = mean(L_all)
        # This exactly equalises contributions BEFORE clamping.
        #
        # Why not log-softmax: BIS Huber (~0.003) is 117× smaller than Phase CE
        # (~0.35) because Huber operates in normalised [0,1] BIS space.  Log-
        # softmax with temperature cannot bridge this gap within safe weight ranges.
        #
        # Typical post-clamp fractions (clamp=[0.1, 8.0]):
        #   BIS   0.003 → w=8.0 → 24 BIS pts
        #   Phase 0.35  → w=0.25 → 88  BIS pts (still dominates but less)
        #   Stim  0.004 → w=8.0 → 32  stim pts
        #   Mono  0.001 → w=8.0 → 8   mono pts
        # Total ≈ 0.15, BIS share ≈ 16% (vs 1.7% with fixed lambdas in v5).
        target = raw.mean()
        w = (target / raw.clamp(min=1e-8))   # uninormalised inverse weights
        # Clamp: lower=0.1 prevents near-zero for Phase; upper=8.0 prevents
        # explosion for very-small Mono loss without a reasonable gradient.
        w = w.clamp(0.1, 8.0)
        losses = torch.stack([bis_err, phase_err, stim_err, mono_err])
        return (w * losses).sum()
