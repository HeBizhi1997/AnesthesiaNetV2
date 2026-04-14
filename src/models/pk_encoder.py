"""
pk_encoder.py — 药代动力学时序编码器（训练时教师网络）

输入：drug_ce (B, T, 6)
  列含义：[CE_prop_raw, CE_rftn_raw, MAC_raw, CE_eq_norm, CE_eq_lagged, ce_velocity]

输出：h_pk (B, T, d_pk)
  时序 GRU 编码，捕捉 CE 轨迹的时序结构（上升/下降/平台期）

设计：轻量 GRU（hidden=64）+ 线性投影 → d_pk=64
推理时此模块完全丢弃，不参与推理。
"""

from __future__ import annotations
import torch
import torch.nn as nn


class PKEncoder(nn.Module):
    """
    药代动力学时序编码器。

    输入维度：6（CE_prop/rftn/mac/eq_norm/lagged/velocity）
    输出维度：d_pk（默认 64）

    训练时只使用 mask_drug > 0 的时步（通过 masked loss 而非结构化 mask，
    因为 GRU 需要连续时序）。
    """

    def __init__(
        self,
        in_dim: int  = 6,
        hidden: int  = 64,
        d_pk: int    = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 输入投影：将原始 PK 特征映射到隐空间
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # 时序建模：GRU 捕捉 CE 的历史轨迹
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, d_pk),
            nn.LayerNorm(d_pk),
        )

        self.d_pk = d_pk

    def forward(
        self,
        drug_ce: torch.Tensor,   # (B, T, 6)
        hx: torch.Tensor = None, # 初始隐状态（可选）
    ) -> torch.Tensor:
        """
        Returns
        -------
        h_pk : (B, T, d_pk) — 每个时步的 PK 编码
        """
        x = self.input_proj(drug_ce)           # (B, T, hidden)
        h_seq, _ = self.gru(x, hx)            # (B, T, hidden)
        h_pk = self.out_proj(h_seq)            # (B, T, d_pk)
        return h_pk
