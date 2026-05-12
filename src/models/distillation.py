"""
distillation.py — BYOL 风格跨模态知识蒸馏头

数学基础（见 MERIDIAN_v9_theory.md §2.4）：

  L_distill = 2 - 2 * <normalize(z_student), normalize(z_teacher)>
             = ||normalize(z_s) - normalize(z_t)||²_F

  stop-gradient 施加于教师侧输出，防止模式坍塌。

  学生（EEG 编码器）通过蒸馏学习将 EEG 表征对齐到：
    1. PK/PD 教师（h_pk）：学习推断药代动力学状态
    2. Vitals 教师（h_vital）：学习推断生理应激状态

v11 修复（原设计缺陷）：
  原设计中 proj_t_pk / proj_t_vital 完全冻结在随机初始化状态（torch.no_grad 包裹整个
  教师投影），导致教师提供固定随机目标而非有意义的 PK/Vital 结构化投影。

  修复方案（BYOL 不对称 EMA）：
    1. 教师输出 .detach() — 蒸馏损失梯度只流入学生（防模式坍塌）
    2. 教师通过方差正则化 + EMA 更新保持有意义（不再是固定随机）
    3. EMA: θ_teacher ← τ·θ_teacher + (1-τ)·θ_student（仅对维度匹配的输出层）
    4. 方差正则化：惩罚投影 std < 0.5 的维度（防止坍塌到常数）
"""

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationHead(nn.Module):
    """
    蒸馏投影头（学生和教师共用此结构）。

    h → Linear(in_dim, d_proj) → GELU → Linear(d_proj, d_proj)
    """

    def __init__(self, in_dim: int, d_proj: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj),
        )
        self.d_proj = d_proj

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, in_dim) → z: (B, T, d_proj)"""
        return self.proj(h)


class CrossModalDistillation(nn.Module):
    """
    双教师跨模态蒸馏模块（v11 EMA 修复版）。

    包含：
      - proj_s_pk    : EEG → PK 方向的学生投影头
      - proj_s_vital : EEG → Vital 方向的学生投影头
      - proj_t_pk    : PK 教师投影头（EMA 更新，输出 detach）
      - proj_t_vital : Vital 教师投影头（EMA 更新，输出 detach）

    EMA 更新：仅对维度匹配的输出层 (Linear(d_proj, d_proj))，
    因为第一层的输入维度不同（d_student=128 vs d_pk/vital=64）。
    """

    def __init__(
        self,
        d_student: int = 128,   # EEG 编码器输出维度
        d_pk: int      = 64,    # PK 编码器输出维度
        d_vital: int   = 64,    # Vital 编码器输出维度
        d_proj: int    = 64,    # 投影空间维度
        ema_tau: float = 0.996, # EMA 衰减系数（BYOL 默认 0.996）
    ):
        super().__init__()
        self.ema_tau = ema_tau

        # 学生投影头（两个方向独立）
        self.proj_s_pk    = DistillationHead(d_student, d_proj)
        self.proj_s_vital = DistillationHead(d_student, d_proj)

        # 教师投影头
        self.proj_t_pk    = DistillationHead(d_pk,    d_proj)
        self.proj_t_vital = DistillationHead(d_vital, d_proj)

        # 初始化教师投影头 — 拷贝学生对应层的初始权重（维度匹配的输出层）
        # 第一层维度不同（d_student≠d_pk），保持独立随机初始化
        self._init_teacher_from_student(self.proj_t_pk, self.proj_s_pk)
        self._init_teacher_from_student(self.proj_t_vital, self.proj_s_vital)

    @staticmethod
    def _init_teacher_from_student(
        teacher: DistillationHead, student: DistillationHead
    ):
        """拷贝输出层权重（d_proj→d_proj，维度匹配）。"""
        # proj[2] 是 Linear(d_proj, d_proj)，teacher 和 student 维度一致
        with torch.no_grad():
            teacher.proj[2].weight.copy_(student.proj[2].weight)
            teacher.proj[2].bias.copy_(student.proj[2].bias)

    @torch.no_grad()
    def update_teacher_ema(self):
        """
        EMA 更新教师投影头（每训练步调用一次，在 optimizer.step() 后）。

        仅更新维度匹配的输出层 Linear(d_proj, d_proj)。
        第一层 Linear(in_dim, d_proj) 输入维度不同，通过方差正则化独立学习。
        """
        tau = self.ema_tau
        for t_head, s_head in [
            (self.proj_t_pk, self.proj_s_pk),
            (self.proj_t_vital, self.proj_s_vital),
        ]:
            # 仅输出层维度匹配（d_proj → d_proj）
            t_head.proj[2].weight.mul_(tau).add_(
                s_head.proj[2].weight, alpha=1.0 - tau
            )
            t_head.proj[2].bias.mul_(tau).add_(
                s_head.proj[2].bias, alpha=1.0 - tau
            )

    @staticmethod
    def _variance_regularization(
        z: torch.Tensor, mask: torch.Tensor, min_std: float = 0.3
    ) -> torch.Tensor:
        """
        方差正则化：惩罚投影 std < min_std 的维度，防止教师坍塌到常数输出。

        z    : (B, T, D) 教师投影
        mask : (B, T)    有效时步掩码
        """
        mask_f = mask.float().unsqueeze(-1)  # (B, T, 1)
        n_valid = mask_f.sum().clamp(min=1e-6)
        # 仅对有效时步计算 std
        z_masked = z * mask_f
        mean = z_masked.sum(dim=[0, 1]) / n_valid  # (D,)
        diff = (z - mean.unsqueeze(0).unsqueeze(0)) * mask_f
        var = (diff ** 2).sum(dim=[0, 1]) / n_valid  # (D,)
        std = torch.sqrt(var + 1e-8)
        # 惩罚 std < min_std 的维度
        penalty = F.relu(min_std - std).mean()
        return penalty

    @staticmethod
    def _cosine_distill_loss(
        z_s: torch.Tensor,      # (B, T, D) 学生投影
        z_t: torch.Tensor,      # (B, T, D) 教师投影（已 detach）
        mask: torch.Tensor,     # (B, T) 有效时步掩码（1=有教师数据）
    ) -> torch.Tensor:
        """
        BYOL 余弦蒸馏损失（对有效时步取均值）。

        L = 2 - 2 * cos_sim(normalize(z_s), normalize(z_t))
          ∈ [0, 4]，越小越好
        """
        z_s_norm = F.normalize(z_s, dim=-1)
        z_t_norm = F.normalize(z_t, dim=-1)
        cos_sim = (z_s_norm * z_t_norm).sum(-1)          # (B, T)
        loss_per_step = 2.0 - 2.0 * cos_sim              # (B, T) ∈ [0, 4]

        mask_f = mask.float()
        n_valid = mask_f.sum().clamp(min=1e-6)
        loss = (loss_per_step * mask_f).sum() / n_valid
        return loss

    def forward(
        self,
        h_eeg: torch.Tensor,            # (B, T, d_student) EEG 隐状态
        h_pk: torch.Tensor,             # (B, T, d_pk)      PK 编码
        h_vital: Optional[torch.Tensor] = None,  # (B, T, d_vital) Vital 编码（None=跳过）
        mask_drug: torch.Tensor = None,         # (B, T) 药物数据可用性
        mask_vital: Optional[torch.Tensor] = None,  # (B, T) 生命体征可用性
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns
        -------
        loss_pk      : PK 蒸馏损失（标量，梯度 → student）
        loss_vital   : Vital 蒸馏损失（标量，梯度 → student），h_vital=None 时返回 None
        reg_pk       : PK 教师方差正则化（标量，梯度 → teacher）
        reg_vital    : Vital 教师方差正则化（标量，梯度 → teacher），h_vital=None 时返回 None
        """
        # 学生投影（梯度流入 EEG 编码器）
        z_s_pk = self.proj_s_pk(h_eeg)      # (B, T, d_proj)

        # PK 教师投影
        z_t_pk_raw = self.proj_t_pk(h_pk)
        z_t_pk     = z_t_pk_raw.detach()     # (B, T, d_proj)

        mask_pk_bool = mask_drug > 0.5

        # PK 蒸馏损失（梯度只流向 student）
        loss_pk = self._cosine_distill_loss(z_s_pk, z_t_pk, mask_pk_bool)

        # PK 方差正则化（梯度流向 teacher）
        reg_pk = self._variance_regularization(z_t_pk_raw, mask_pk_bool)

        # Vital 蒸馏（仅在 h_vital 提供时计算）
        if h_vital is not None and mask_vital is not None:
            z_s_vital    = self.proj_s_vital(h_eeg)
            z_t_vital_raw = self.proj_t_vital(h_vital)
            z_t_vital    = z_t_vital_raw.detach()

            mask_vit_bool = mask_vital > 0.5

            loss_vital = self._cosine_distill_loss(z_s_vital, z_t_vital, mask_vit_bool)
            reg_vital  = self._variance_regularization(z_t_vital_raw, mask_vit_bool)
        else:
            loss_vital = None
            reg_vital  = None

        return loss_pk, loss_vital, reg_pk, reg_vital
