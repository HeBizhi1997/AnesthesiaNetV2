"""
vital_encoder.py — 生命体征快照编码器（训练时教师网络）

输入：vitals (B, T, 5)
  列含义：[HR, SpO2, MBP, ETCO2, BT]（已全局归一化）

设计：每个时步独立编码（MLP，无时序依赖），
因为生命体征快照是"状态"而非轨迹——当前值比历史轨迹更重要。
但加入前后的差分特征（Δvitals）以捕捉瞬态变化。

输出：h_vital (B, T, d_v)
推理时此模块完全丢弃。
"""

from __future__ import annotations
import torch
import torch.nn as nn


class VitalEncoder(nn.Module):
    """
    生命体征快照编码器。

    输入维度：5（HR/SpO2/MBP/ETCO2/BT）
    + 5 差分特征（Δvitals）= 10 维有效输入

    输出维度：d_v（默认 64）
    """

    def __init__(
        self,
        in_dim: int   = 5,
        hidden: int   = 64,
        d_v: int      = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 有效输入维度 = 原始 + 差分
        eff_dim = in_dim * 2

        self.net = nn.Sequential(
            nn.Linear(eff_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_v),
            nn.LayerNorm(d_v),
        )

        self.d_v = d_v

    def forward(self, vitals: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        vitals : (B, T, 5)

        Returns
        -------
        h_vital : (B, T, d_v)
        """
        # 计算时步间差分（捕捉 HR 突升、BP 突降等瞬态变化）
        # delta[t] = vitals[t] - vitals[t-1]，第 0 步用 0 填充
        delta = torch.zeros_like(vitals)
        delta[:, 1:, :] = vitals[:, 1:, :] - vitals[:, :-1, :]

        x = torch.cat([vitals, delta], dim=-1)   # (B, T, 10)
        h_vital = self.net(x)                     # (B, T, d_v)
        return h_vital
