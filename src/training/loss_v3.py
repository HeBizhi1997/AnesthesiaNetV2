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
# 相位类别权重 — sqrt(逆频率) 归一化
#
# 原纯逆频率权重 [0.39, 3.00, 0.0094, 0.60] 导致 maintenance(97.2%) 权重仅
# induction(0.3%) 的 1/320。模型在 epoch 2 后将 maintenance p→0.9999，
# 加权CE→0，97%数据零梯度 → phase head 形同虚设。
#
# sqrt(逆频率) 将梯度比从 319:1 降至 17.9:1，保持少数类上采样效应同时
# 让多数类对训练有实际贡献。
# ─────────────────────────────────────────────────────────────────────────────
_SQRT_INV_FREQ = torch.tensor([
    (1.0 / 0.023) ** 0.5,   # pre_op:     sqrt(43.5) = 6.59
    (1.0 / 0.003) ** 0.5,   # induction:  sqrt(333)  = 18.26
    (1.0 / 0.959) ** 0.5,   # maintenance: sqrt(1.04) = 1.02
    (1.0 / 0.015) ** 0.5,   # recovery:   sqrt(66.7) = 8.16
], dtype=torch.float32)
_PHASE_WEIGHTS = _SQRT_INV_FREQ / _SQRT_INV_FREQ.sum() * 4.0
# Result: [0.78, 2.15, 0.12, 0.96] — maintenance 权重从 0.0094 提升 12.8x


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
        lambda_stim:          float = 0.15,  # v11: 0.5→0.15 (stim:BIS gradient 7:1→1.2:1)
        lambda_pkd:           float = 0.4,   # 理论 §4.2：辅助头需要足够梯度
        lambda_vitald:        float = 0.4,   # VitalDHead BIS 预测（修复 v9 VitalEncoder 无梯度缺陷）
        lambda_distill_pk:    float = 0.2,   # 理论 §4.2：蒸馏是正则化，须 < λ_bis
        lambda_distill_vital: float = 0.2,   # 理论 §4.2：同上
        lambda_trans:         float = 0.3,   # 理论 §4.2：CE 方向约束
        transition_boost:     float = 2.0,   # 高 velocity 时步放大
        vel_threshold:        float = 0.2,   # ce_velocity 过渡期阈值
        huber_delta:          float = 0.05,  # 理论 §2.6：δ=5 BIS pts（归一化空间 0.05）
        bis_transition_weight: float = 1.0,  # v14: 过渡区 L_bis 权重放大（>1 启用），打击诱导/复苏 2× 误差
        bis_transition_dbis:   float = 0.02, # v14: |Δlabel/步| 阈值（0.02≈2 BIS/s），超过即过渡区
        bis_phase_weights: Optional[list] = None,  # v15: 相位平衡 L_bis 权重 [pre_op,ind,maint,rec]
                                                   #      None→[1,1,1,1] 无操作；maintenance 基准 1.0
        focal_gamma:          float = 2.0,
        focal_alpha:          float = 0.5,    # v8 validated; 0.99→9801:1 gradient ratio (broken)
        stim_pos_weight:      float = 15.0,   # v8 validated; 99→9801:1 gradient ratio (broken)
        phase2_start_epoch:   int   = 31,
        phase3_start_epoch:   int   = 61,
        stim_warmup_epochs:   int   = 5,    # v10: Phase2切换时 stim 线性热身 epoch 数
        phase3_warmup_epochs: int   = 5,    # v10: Phase3切换时蒸馏损失线性热身 epoch 数
        use_auto_weight:      bool  = False,
        auto_weight_temp:     float = 0.5,
    ):
        super().__init__()
        self.lambda_bis           = lambda_bis
        self.lambda_phase         = lambda_phase
        self.lambda_stim          = lambda_stim
        self.stim_warmup_epochs   = stim_warmup_epochs
        self.phase3_warmup_epochs = phase3_warmup_epochs
        self.lambda_pkd           = lambda_pkd
        self.lambda_vitald        = lambda_vitald
        self.lambda_distill_pk    = lambda_distill_pk
        self.lambda_distill_vital = lambda_distill_vital
        self.lambda_trans         = lambda_trans
        self.transition_boost     = transition_boost
        self.vel_threshold        = vel_threshold
        self.huber_delta          = huber_delta
        self.bis_transition_weight = bis_transition_weight
        self.bis_transition_dbis   = bis_transition_dbis
        self.focal_gamma          = focal_gamma
        self.focal_alpha          = focal_alpha
        self.stim_pos_weight      = stim_pos_weight
        self.phase2_start_epoch   = phase2_start_epoch
        self.phase3_start_epoch   = phase3_start_epoch
        self.use_auto_weight      = use_auto_weight
        self.auto_weight_temp     = auto_weight_temp

        self.register_buffer("phase_weights", _PHASE_WEIGHTS)

        # v15: 相位平衡 L_bis 权重 —— 针对 maintenance(97%) 主导导致的输出区间坍缩
        # （诊断 diag_preop：pre_op/induction/recovery 系统性欠预测，预测 p99 仅 81）。
        _bpw = torch.tensor(
            bis_phase_weights if bis_phase_weights is not None else [1.0, 1.0, 1.0, 1.0],
            dtype=torch.float32)
        self.register_buffer("bis_phase_weights", _bpw)

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
        bis_vitald:   Optional[torch.Tensor] = None,  # (B,T,1) Vital 辅助 BIS（v3 fix）
        loss_distill_pk:    Optional[torch.Tensor] = None,  # 标量
        loss_distill_vital: Optional[torch.Tensor] = None,  # 标量
        reg_distill_pk:     Optional[torch.Tensor] = None,  # 标量（教师方差正则化）
        reg_distill_vital:  Optional[torch.Tensor] = None,  # 标量（教师方差正则化）
        drug_ce:      Optional[torch.Tensor] = None,  # (B,T,6) 用于 L_trans
        mask_drug:    Optional[torch.Tensor] = None,  # (B,T)
        mask_vital:   Optional[torch.Tensor] = None,  # (B,T) Vital 数据可用性
        ce_velocity:  Optional[torch.Tensor] = None,  # (B,T)
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict 包含 "loss"（总损失，有梯度）和各子损失（.detach()，仅用于日志）。
        """
        B, T = label_bis.shape
        cur_phase = self.get_curriculum_phase(epoch)

        # ── 1. L_bis：SQI 遮掩 + 相位平衡 + 过渡区加权 Huber ─────────────────────
        # 逐元素 Huber，权重 = SQI掩码 × max(相位平衡, 过渡区放大)。
        #   · 过渡区(velocity)：|Δlabel| 斜率，针对诱导/复苏陡变段（v14）。
        #   · 相位平衡(v15)：按相位上采样 pre_op/诱导/复苏，修复 maintenance(97%)
        #     主导造成的输出区间坍缩（diag_preop：这些相位系统性欠预测 14/4/9 BIS）。
        #   两者取 max 而非相乘，避免诱导陡变段权重爆炸（6×3=18）。
        sqi_ok  = (sqi_mean > 0.5).float()
        pred_sq = pred_bis.squeeze(-1)
        huber_el = F.huber_loss(
            pred_sq, label_bis, delta=self.huber_delta, reduction="none")  # (B, T)

        # 速度型过渡权重
        if self.bis_transition_weight > 1.0 and T > 1:
            dlabel = torch.zeros_like(label_bis)
            dlabel[:, 1:] = (label_bis[:, 1:] - label_bis[:, :-1]).abs()
            is_trans = (dlabel > self.bis_transition_dbis).float()
            w_trans = 1.0 + (self.bis_transition_weight - 1.0) * is_trans
        else:
            w_trans = torch.ones_like(label_bis)

        # 相位平衡权重（phase_labels:(B,T) int{0..3}）
        w_phase = self.bis_phase_weights.to(label_bis.device)[phase_labels.clamp(0, 3)]

        w_bis = sqi_ok * torch.maximum(w_phase, w_trans)
        bis_err = (huber_el * w_bis).sum() / (w_bis.sum() + 1e-6)

        # ── 2. L_phase：加权交叉熵 ───────────────────────────────────────────
        ph_logits_flat = phase_logits.view(B * T, -1)
        ph_labels_flat = phase_labels.view(-1).long()
        phase_err = F.cross_entropy(
            ph_logits_flat, ph_labels_flat,
            weight=self.phase_weights.to(ph_logits_flat.device),
        )

        # ── 3. L_stim：Focal 损失 (Phase 2+ 全量，Phase 1 微量防退化) ──────
        # Phase 1: 仅防 stim head 权重退化（λ=0.001，约 1/150 全量），不影响 BIS 收敛
        # Phase 2: 线性热身 ep31→ep35，阶跃保护 BIS 头
        # Phase 3: 全量 λ_stim=0.15
        stim_ramp = 1.0
        if cur_phase == 1:
            effective_lambda_stim = self.lambda_stim * 0.007   # ~0.001, anti-decay only
        elif cur_phase == 2 and self.stim_warmup_epochs > 0:
            epochs_into_ph2 = epoch - self.phase2_start_epoch
            stim_ramp = min(1.0, epochs_into_ph2 / self.stim_warmup_epochs)
            effective_lambda_stim = self.lambda_stim * stim_ramp
        else:
            effective_lambda_stim = self.lambda_stim

        stim_err = focal_loss(
            stim_logits.view(-1),
            stim_labels.view(-1),
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            pos_weight=self.stim_pos_weight,
        )

        # ── Phase 1/2 总损失 ──────────────────────────────────────────────────
        if cur_phase < 3:
            main_losses = [bis_err, phase_err, stim_err]
            if self.use_auto_weight:
                total = self._auto_weighted_sum(main_losses)
            else:
                total = (self.lambda_bis   * bis_err +
                         self.lambda_phase * phase_err +
                         effective_lambda_stim * stim_err)

            return {
                "loss":          total,
                "bis_loss":      bis_err.detach(),
                "phase_loss":    phase_err.detach(),
                "stim_loss":     stim_err.detach(),
                "pkd_loss":      pred_bis.new_zeros(1).squeeze().detach(),
                "distill_pk":    pred_bis.new_zeros(1).squeeze().detach(),
                "distill_vital": pred_bis.new_zeros(1).squeeze().detach(),
                "reg_pk":        pred_bis.new_zeros(1).squeeze().detach(),
                "reg_vital":     pred_bis.new_zeros(1).squeeze().detach(),
                "trans_loss":    pred_bis.new_zeros(1).squeeze().detach(),
                "curriculum_phase": pred_bis.new_tensor(cur_phase),
            }

        # ── Phase 3：附加多模态损失 ───────────────────────────────────────────

        # 4a. L_pkd：遮掩 Huber（PK 辅助 BIS）
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

        # 4b. L_vitald：遮掩 Huber（Vital 辅助 BIS，修复 VitalEncoder 无梯度问题）
        vitald_err = pred_bis.new_zeros(1).squeeze()
        if bis_vitald is not None and mask_vital is not None:
            vitald_err = masked_huber_loss(
                bis_vitald, label_bis, mask_vital.float(), self.huber_delta)

        # 5. L_distill：来自 model.forward()，直接使用
        distill_pk_err    = loss_distill_pk    if loss_distill_pk    is not None \
                            else pred_bis.new_zeros(1).squeeze()
        distill_vital_err = loss_distill_vital if loss_distill_vital is not None \
                            else pred_bis.new_zeros(1).squeeze()

        # 5b. 教师方差正则化（防止投影坍塌到常数，v11 新增）
        reg_pk_err    = reg_distill_pk    if reg_distill_pk    is not None \
                        else pred_bis.new_zeros(1).squeeze()
        reg_vital_err = reg_distill_vital if reg_distill_vital is not None \
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
        # 辅助损失同样线性热身（防止 Phase 3 开始时蒸馏损失冲击 BIS 头，类似 stim warmup）
        # v9 Phase 3 ep61: dv=0.73 立即全量激活 → BIS 回归被噪声梯度淹没
        ph3_ramp = 1.0
        if self.phase3_warmup_epochs > 0:
            epochs_into_ph3 = epoch - self.phase3_start_epoch   # 0 at first Phase3 epoch
            ph3_ramp = min(1.0, epochs_into_ph3 / self.phase3_warmup_epochs)

        # 主任务（UW-SO 可选） + 辅助任务（固定 λ，带热身）
        main_losses = [bis_err, phase_err, stim_err]
        if self.use_auto_weight:
            main_total = self._auto_weighted_sum(main_losses)
        else:
            main_total = (self.lambda_bis   * bis_err +
                          self.lambda_phase * phase_err +
                          self.lambda_stim  * stim_err)

        aux_total = ph3_ramp * (
            self.lambda_pkd           * pkd_err          +
            self.lambda_vitald        * vitald_err        +
            self.lambda_distill_pk    * distill_pk_err   +
            self.lambda_distill_vital * distill_vital_err +
            self.lambda_trans         * trans_err         +
            0.01                      * reg_pk_err        +   # small fixed weight
            0.01                      * reg_vital_err         # small fixed weight
        )
        total = main_total + aux_total

        return {
            "loss":          total,
            "bis_loss":      bis_err.detach(),
            "phase_loss":    phase_err.detach(),
            "stim_loss":     stim_err.detach(),
            "pkd_loss":      pkd_err.detach(),
            "vitald_loss":   vitald_err.detach(),
            "distill_pk":    distill_pk_err.detach(),
            "distill_vital": distill_vital_err.detach(),
            "reg_pk":        reg_pk_err.detach(),
            "reg_vital":     reg_vital_err.detach(),
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
