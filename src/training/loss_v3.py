"""
loss_v3.py — MERIDIAN (AnesthesiaNetV3) 六项多任务损失

三阶段课程学习：

  Phase 1 (ep 1 ~ phase2_start-1):  L_bis + L_phase
    — 建立 EEG→BIS 基础回归 + 相位分类

  Phase 2 (phase2_start ~ phase3_start-1):  + L_stim
    — 引入心血管刺激检测 (CV 标签)

  Phase 3 (phase3_start ~ end):  + L_pkd + L_distill_pk + L_distill_vital + L_trans
    — 多模态药代动力学辅助 + 跨模态蒸馏 + CE 方向约束

L_distill_pk / L_distill_vital 已由 AnesthesiaNetV3.forward() 在模型内部
计算并存入 out dict，此处直接接收标量值（不重新计算）。

ce_velocity 加权：
  高 ce_velocity（浓度快速变化 = 过渡期）时序列的 L_pkd 和 L_trans
  被 weight = 1 + transition_boost * (velocity > vel_threshold) 上调，
  优化麻醉诱导/苏醒阶段的预测质量。
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 相位类别权重（与 v2 相同）
# pre_op=2.3%, induction=0.3%, maintenance=95.9%, recovery=1.5%
# ─────────────────────────────────────────────────────────────────────────────
_RAW_WEIGHTS = torch.tensor([
    1.0 / 0.023,
    1.0 / 0.003,
    1.0 / 0.959,
    1.0 / 0.015,
], dtype=torch.float32)
_PHASE_WEIGHTS = _RAW_WEIGHTS / _RAW_WEIGHTS.sum() * 4.0


# ─────────────────────────────────────────────────────────────────────────────
# 辅助损失函数
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.99,
    pos_weight: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss with pos_weight（与 v2 相同）。"""
    p   = torch.sigmoid(logits)
    pw  = logits.new_tensor(pos_weight)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="none")
    p_t     = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    fl = alpha_t * (1 - p_t) ** gamma * bce
    return fl.mean() if reduction == "mean" else fl.sum()


def pk_direction_loss(
    pred_bis:    torch.Tensor,  # (B, T, 1) — 归一化 BIS 预测
    ce_eq_norm:  torch.Tensor,  # (B, T)   — 归一化 CE_eq（drug_ce[:,3]）
    ce_velocity: torch.Tensor,  # (B, T)   — 归一化 |dCE/dt|（drug_ce[:,5]）
    mask_drug:   torch.Tensor,  # (B, T)   — 药物数据可用性掩码
    vel_threshold: float = 0.2, # 显著过渡期阈值
) -> torch.Tensor:
    """
    CE 方向约束损失（L_trans）。

    物理依据（MERIDIAN_v9_theory.md §5）：
      CE 上升（propofol 效应室浓度升高）→ BIS 必须下降（意识抑制加深）。
      CE 下降（苏醒期浓度下降）→ BIS 必须上升（意识恢复）。

    只对 ce_velocity > vel_threshold 的过渡期时步施加约束：
      - 浓度平台期（手术维持阶段）不约束，避免干扰正常波动。
      - 只对有药物数据的时步约束（mask_drug）。

    惩罚：
      CE 上升且 d_BIS > 0  → F.relu(d_pred)  （BIS 上升违反方向）
      CE 下降且 d_BIS < 0  → F.relu(-d_pred) （BIS 下降违反方向）
    """
    if pred_bis.shape[1] < 2:
        return pred_bis.new_zeros(1).squeeze()

    pred    = pred_bis.squeeze(-1)              # (B, T)
    d_pred  = pred[:, 1:] - pred[:, :-1]        # (B, T-1)
    d_ce    = ce_eq_norm[:, 1:] - ce_eq_norm[:, :-1]  # (B, T-1) 有符号方向

    # 用左侧时步的 velocity 和 mask（因为 d_pred[t] = pred[t+1]-pred[t]）
    vel_mid  = ce_velocity[:, :-1]              # (B, T-1)
    mask_mid = mask_drug[:, :-1].float()        # (B, T-1)

    # 过渡期有效掩码：velocity 显著 AND 有药物数据
    trans_mask = (vel_mid > vel_threshold).float() * mask_mid

    rising_viol  = F.relu(d_pred)  * (d_ce > 0).float() * trans_mask
    falling_viol = F.relu(-d_pred) * (d_ce < 0).float() * trans_mask

    n_valid = trans_mask.sum().clamp(min=1e-6)
    return (rising_viol + falling_viol).sum() / n_valid


def masked_huber_loss(
    pred:  torch.Tensor,  # (B, T, 1) 或 (B, T)
    label: torch.Tensor,  # (B, T)
    mask:  torch.Tensor,  # (B, T) float/bool
    delta: float = 0.10,
) -> torch.Tensor:
    """遮掩 Huber 损失（用于 L_pkd：只对有药物数据的时步计算）。"""
    p   = pred.squeeze(-1)          # (B, T)
    m   = mask.float()
    err = F.huber_loss(p * m, label * m, delta=delta, reduction="sum")
    n   = m.sum().clamp(min=1e-6)
    return err / n


# ─────────────────────────────────────────────────────────────────────────────
# 主损失模块
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskLossV3(nn.Module):
    """
    MERIDIAN 六项多任务损失（三阶段课程学习）。

    Parameters
    ----------
    lambda_bis          : BIS Huber 损失权重
    lambda_phase        : 相位分类损失权重
    lambda_stim         : 刺激检测 focal 损失权重
    lambda_pkd          : PK 辅助 BIS Huber 损失权重
    lambda_distill_pk   : PK 蒸馏损失权重
    lambda_distill_vital: Vital 蒸馏损失权重
    lambda_trans        : CE 方向约束损失权重
    transition_boost    : 高 ce_velocity 时步的 L_pkd/L_trans 权重放大倍数
    phase2_start_epoch  : Phase 2 开始 epoch（L_stim 激活）
    phase3_start_epoch  : Phase 3 开始 epoch（L_pkd/L_distill/L_trans 激活）
    use_auto_weight     : UW-SO 自适应权重（对主任务 bis+phase+stim）
    """

    def __init__(
        self,
        lambda_bis:           float = 1.0,
        lambda_phase:         float = 0.3,   # 理论 §4.2：相位标签误差大，不宜过强
        lambda_stim:          float = 0.5,   # 理论 §4.2：CV 标签质量高，可加强
        lambda_pkd:           float = 0.4,   # 理论 §4.2：辅助头需要足够梯度
        lambda_distill_pk:    float = 0.2,   # 理论 §4.2：蒸馏是正则化，须 < λ_bis
        lambda_distill_vital: float = 0.2,   # 理论 §4.2：同上
        lambda_trans:         float = 0.3,   # 理论 §4.2：CE 方向约束
        transition_boost:     float = 2.0,   # 高 velocity 时步放大
        vel_threshold:        float = 0.2,   # ce_velocity 过渡期阈值
        huber_delta:          float = 0.05,  # 理论 §2.6：δ=5 BIS pts（归一化空间 0.05）
        focal_gamma:          float = 2.0,
        focal_alpha:          float = 0.99,
        stim_pos_weight:      float = 99.0,
        phase2_start_epoch:   int   = 31,
        phase3_start_epoch:   int   = 61,
        use_auto_weight:      bool  = False,
        auto_weight_temp:     float = 0.5,
    ):
        super().__init__()
        self.lambda_bis           = lambda_bis
        self.lambda_phase         = lambda_phase
        self.lambda_stim          = lambda_stim
        self.lambda_pkd           = lambda_pkd
        self.lambda_distill_pk    = lambda_distill_pk
        self.lambda_distill_vital = lambda_distill_vital
        self.lambda_trans         = lambda_trans
        self.transition_boost     = transition_boost
        self.vel_threshold        = vel_threshold
        self.huber_delta          = huber_delta
        self.focal_gamma          = focal_gamma
        self.focal_alpha          = focal_alpha
        self.stim_pos_weight      = stim_pos_weight
        self.phase2_start_epoch   = phase2_start_epoch
        self.phase3_start_epoch   = phase3_start_epoch
        self.use_auto_weight      = use_auto_weight
        self.auto_weight_temp     = auto_weight_temp

        self.register_buffer("phase_weights", _PHASE_WEIGHTS)

    def get_curriculum_phase(self, epoch: int) -> int:
        """返回课程阶段（1/2/3）。"""
        if epoch < self.phase2_start_epoch:
            return 1
        elif epoch < self.phase3_start_epoch:
            return 2
        else:
            return 3

    def forward(
        self,
        # 模型输出
        pred_bis:     torch.Tensor,            # (B, T, 1)
        phase_logits: torch.Tensor,            # (B, T, 4)
        stim_logits:  torch.Tensor,            # (B, T, 1)
        # 标签
        label_bis:    torch.Tensor,            # (B, T) 归一化 [0,1]
        phase_labels: torch.Tensor,            # (B, T) int64 {0,1,2,3}
        stim_labels:  torch.Tensor,            # (B, T) float {0,1}
        sqi_mean:     torch.Tensor,            # (B, T) SQI 掩码
        # 课程阶段控制
        epoch:        int = 1,
        # Phase 3 附加项（均为可选，仅 Phase 3 提供）
        bis_pkd:      Optional[torch.Tensor] = None,  # (B,T,1) PK 辅助 BIS
        loss_distill_pk:    Optional[torch.Tensor] = None,  # 标量
        loss_distill_vital: Optional[torch.Tensor] = None,  # 标量
        drug_ce:      Optional[torch.Tensor] = None,  # (B,T,6) 用于 L_trans
        mask_drug:    Optional[torch.Tensor] = None,  # (B,T)
        ce_velocity:  Optional[torch.Tensor] = None,  # (B,T)
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict 包含 "loss"（总损失，有梯度）和各子损失（.detach()，仅用于日志）。
        """
        B, T = label_bis.shape
        cur_phase = self.get_curriculum_phase(epoch)

        # ── 1. L_bis：SQI 遮掩 Huber ─────────────────────────────────────────
        sqi_ok  = (sqi_mean > 0.5).float()
        pred_sq = pred_bis.squeeze(-1)
        bis_err = F.huber_loss(
            pred_sq * sqi_ok, label_bis * sqi_ok,
            delta=self.huber_delta, reduction="sum",
        ) / (sqi_ok.sum() + 1e-6)

        # ── 2. L_phase：加权交叉熵 ───────────────────────────────────────────
        ph_logits_flat = phase_logits.view(B * T, -1)
        ph_labels_flat = phase_labels.view(-1).long()
        phase_err = F.cross_entropy(
            ph_logits_flat, ph_labels_flat,
            weight=self.phase_weights.to(ph_logits_flat.device),
        )

        # ── 3. L_stim：Focal 损失 (Phase 2+) ────────────────────────────────
        stim_err = focal_loss(
            stim_logits.view(-1),
            stim_labels.view(-1),
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            pos_weight=self.stim_pos_weight,
        ) if cur_phase >= 2 else pred_bis.new_zeros(1).squeeze()

        # ── Phase 1/2 总损失 ──────────────────────────────────────────────────
        if cur_phase < 3:
            main_losses = [bis_err, phase_err] if cur_phase == 1 else [bis_err, phase_err, stim_err]
            if self.use_auto_weight:
                total = self._auto_weighted_sum(main_losses)
            else:
                lambdas = [self.lambda_bis, self.lambda_phase]
                if cur_phase >= 2:
                    lambdas.append(self.lambda_stim)
                total = sum(l * e for l, e in zip(lambdas, main_losses))

            return {
                "loss":          total,
                "bis_loss":      bis_err.detach(),
                "phase_loss":    phase_err.detach(),
                "stim_loss":     stim_err.detach(),
                "pkd_loss":      pred_bis.new_zeros(1).squeeze().detach(),
                "distill_pk":    pred_bis.new_zeros(1).squeeze().detach(),
                "distill_vital": pred_bis.new_zeros(1).squeeze().detach(),
                "trans_loss":    pred_bis.new_zeros(1).squeeze().detach(),
                "curriculum_phase": pred_bis.new_tensor(cur_phase),
            }

        # ── Phase 3：附加多模态损失 ───────────────────────────────────────────

        # 4. L_pkd：遮掩 Huber（PK 辅助 BIS）
        pkd_err = pred_bis.new_zeros(1).squeeze()
        if bis_pkd is not None and mask_drug is not None:
            # ce_velocity 加权：过渡期更重要
            if ce_velocity is not None:
                # 序列级权重：序列内最大 velocity 代表"过渡程度"
                seq_vel = ce_velocity.mean(-1, keepdim=True)  # (B, 1)
                boost = 1.0 + (self.transition_boost - 1.0) * (
                    seq_vel > self.vel_threshold).float()     # (B, 1)
                # 点乘 mask_drug 传入加权 Huber
                weighted_mask = mask_drug.float() * boost
            else:
                weighted_mask = mask_drug.float()
            pkd_err = masked_huber_loss(
                bis_pkd, label_bis, weighted_mask, self.huber_delta)

        # 5. L_distill：来自 model.forward()，直接使用
        distill_pk_err    = loss_distill_pk    if loss_distill_pk    is not None \
                            else pred_bis.new_zeros(1).squeeze()
        distill_vital_err = loss_distill_vital if loss_distill_vital is not None \
                            else pred_bis.new_zeros(1).squeeze()

        # 6. L_trans：CE 方向约束
        trans_err = pred_bis.new_zeros(1).squeeze()
        if drug_ce is not None and mask_drug is not None:
            ce_eq_n  = drug_ce[:, :, 3]        # CE_eq_norm
            ce_vel   = drug_ce[:, :, 5]        # ce_velocity
            if ce_velocity is not None:
                ce_vel = ce_velocity           # 已从 batch 提取（一致性）
            trans_err = pk_direction_loss(
                pred_bis, ce_eq_n, ce_vel, mask_drug, self.vel_threshold)

        # ── Phase 3 总损失 ────────────────────────────────────────────────────
        # 主任务（UW-SO 可选） + 辅助任务（固定 λ）
        main_losses = [bis_err, phase_err, stim_err]
        if self.use_auto_weight:
            main_total = self._auto_weighted_sum(main_losses)
        else:
            main_total = (self.lambda_bis   * bis_err +
                          self.lambda_phase * phase_err +
                          self.lambda_stim  * stim_err)

        aux_total = (
            self.lambda_pkd           * pkd_err          +
            self.lambda_distill_pk    * distill_pk_err   +
            self.lambda_distill_vital * distill_vital_err +
            self.lambda_trans         * trans_err
        )
        total = main_total + aux_total

        return {
            "loss":          total,
            "bis_loss":      bis_err.detach(),
            "phase_loss":    phase_err.detach(),
            "stim_loss":     stim_err.detach(),
            "pkd_loss":      pkd_err.detach(),
            "distill_pk":    distill_pk_err.detach(),
            "distill_vital": distill_vital_err.detach(),
            "trans_loss":    trans_err.detach(),
            "curriculum_phase": pred_bis.new_tensor(cur_phase),
        }

    def _auto_weighted_sum(self, loss_list: list) -> torch.Tensor:
        """
        UW-SO 自适应权重（对主任务列表）。

        同 v2 的逆损失归一化策略，阶段自适应地支持 2/3 项主任务。
        """
        raw = torch.stack([l.detach() for l in loss_list])
        target = raw.mean()
        w = (target / raw.clamp(min=1e-8)).clamp(0.1, 8.0)
        return (w * torch.stack(loss_list)).sum()
