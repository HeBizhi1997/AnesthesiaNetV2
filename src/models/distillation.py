"""
distillation.py — BYOL 风格跨模态知识蒸馏头

数学基础（见 MERIDIAN_v9_theory.md §2.4）：

  L_distill = 2 - 2 * <normalize(z_student), normalize(z_teacher)>
             = ||normalize(z_s) - normalize(z_t)||²_F

  stop-gradient 施加于教师侧，防止模式坍塌。

  学生（EEG 编码器）通过蒸馏学习将 EEG 表征对齐到：
    1. PK/PD 教师（h_pk）：学习推断药代动力学状态
    2. Vitals 教师（h_vital）：学习推断生理应激状态

每个教师使用独立的投影头，避免 EEG 表征被强制同时
与两个不同信息源完全对齐（理论文档 §2.4.2）。
"""

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationHead(nn.Module):
    """
    单向蒸馏投影头（学生侧）。

    学生表征 h_s → proj_s(h_s) → 归一化余弦蒸馏损失。
    教师表征通过 stop-gradient 提供目标。
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
    双教师跨模态蒸馏模块。

    包含：
      - proj_s_pk    : EEG → PK 方向的学生投影头
      - proj_s_vital : EEG → Vital 方向的学生投影头
      - proj_t_pk    : PK 教师投影头（输出 stop-grad）
      - proj_t_vital : Vital 教师投影头（输出 stop-grad）
    """

    def __init__(
        self,
        d_student: int = 128,   # EEG 编码器输出维度
        d_pk: int      = 64,    # PK 编码器输出维度
        d_vital: int   = 64,    # Vital 编码器输出维度
        d_proj: int    = 64,    # 投影空间维度
    ):
        super().__init__()

        # 学生投影头（两个方向独立）
        self.proj_s_pk    = DistillationHead(d_student, d_proj)
        self.proj_s_vital = DistillationHead(d_student, d_proj)

        # 教师投影头
        self.proj_t_pk    = DistillationHead(d_pk,    d_proj)
        self.proj_t_vital = DistillationHead(d_vital, d_proj)

    @staticmethod
    def _cosine_distill_loss(
        z_s: torch.Tensor,      # (B, T, D) 学生投影
        z_t: torch.Tensor,      # (B, T, D) 教师投影（stop-grad 已施加）
        mask: torch.Tensor,     # (B, T) 有效时步掩码（1=有教师数据）
    ) -> torch.Tensor:
        """
        BYOL 余弦蒸馏损失（对有效时步取均值）。

        L = 2 - 2 * cos_sim(normalize(z_s), normalize(z_t))
          ∈ [0, 4]，越小越好

        mask 确保仅对教师数据可用的时步计算损失。
        """
        z_s_norm = F.normalize(z_s, dim=-1)
        z_t_norm = F.normalize(z_t, dim=-1)
        # 逐时步余弦相似度
        cos_sim = (z_s_norm * z_t_norm).sum(-1)   # (B, T)
        loss_per_step = 2.0 - 2.0 * cos_sim        # (B, T) ∈ [0, 4]

        # Masked mean（仅有教师数据的时步）
        mask_f = mask.float()
        n_valid = mask_f.sum().clamp(min=1e-6)
        loss = (loss_per_step * mask_f).sum() / n_valid
        return loss

    def forward(
        self,
        h_eeg: torch.Tensor,       # (B, T, d_student) EEG 隐状态
        h_pk: torch.Tensor,        # (B, T, d_pk)      PK 编码
        h_vital: torch.Tensor,     # (B, T, d_vital)   Vital 编码
        mask_drug: torch.Tensor,   # (B, T)             药物数据可用性
        mask_vital: torch.Tensor,  # (B, T)             生命体征可用性
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        loss_pk    : PK 蒸馏损失（标量）
        loss_vital : Vital 蒸馏损失（标量）
        """
        # 学生投影（梯度流入 EEG 编码器）
        z_s_pk    = self.proj_s_pk(h_eeg)      # (B, T, d_proj)
        z_s_vital = self.proj_s_vital(h_eeg)   # (B, T, d_proj)

        # 教师投影（完全 stop-gradient：梯度既不流入 PKEncoder/VitalEncoder，
        # 也不流入教师投影头参数本身）
        #
        # 正确性依据（BYOL 理论 §2.4.1）：
        #   若 proj_t_pk 允许被 distillation 梯度更新，它可以学习将所有 h_pk
        #   映射到同一单位向量（"合谋"），使 cos_sim→1 而不传递任何信息 → 模式坍塌。
        #   完全 no_grad 后，proj_t_pk 保持随机初始化不变，提供固定随机投影。
        #   由 Johnson-Lindenstrauss 引理，随机投影保持结构，蒸馏目标仍然有意义。
        #   PKEncoder 的有用表征通过 L_pkd → h_pk → z_t_pk (固定投影) 间接传递给学生。
        with torch.no_grad():
            z_t_pk    = self.proj_t_pk(h_pk)
            z_t_vital = self.proj_t_vital(h_vital)

        # 有效掩码转换为 bool
        mask_pk_bool  = mask_drug  > 0.5   # (B, T)
        mask_vit_bool = mask_vital > 0.5   # (B, T)

        loss_pk    = self._cosine_distill_loss(z_s_pk,    z_t_pk,    mask_pk_bool)
        loss_vital = self._cosine_distill_loss(z_s_vital, z_t_vital, mask_vit_bool)

        return loss_pk, loss_vital
