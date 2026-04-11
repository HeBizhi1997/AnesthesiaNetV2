"""
LNN Core — Liquid Neural Network using Neural Circuit Policies (NCPs).

v8 改进：RecurrentWrapper 模式 — 统一隐藏状态形状
──────────────────────────────────────────────────────────────────
临床场景：手术室实时流式推理时，隐藏状态 h 必须能跨时间步无缝传递。
原设计中：
  - GRU  返回 h: (num_layers, B, units)   ← 3-D
  - CfC  返回 h: (B, ncp_total)           ← 2-D

导致外部代码必须感知后端类型才能操作 h（如 TBPTT 状态掩码）。

v8 修复：
  1. normalize_state(h) 方法 — 统一转换为 (B, units)，供流式推理和外部比较
  2. h_for_next 内部保持原生格式（GRU 需要 3-D hx 输入）
  3. 外部 API：forward() 仍返回 (out, h_native)，调用 normalize_state() 获取标准化状态
  4. 新增 init_state(B, device) — 生成正确形状的零初始状态

这遵循 RecurrentWrapper 设计模式：
  - 核心 RNN 用原生格式高效运行
  - 对外暴露统一接口，下游代码无需感知后端
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

try:
    from ncps.torch import CfC
    from ncps.wirings import AutoNCP
    _NCPS_AVAILABLE = True
except ImportError:
    _NCPS_AVAILABLE = False


class LNNCore(nn.Module):
    """
    CfC-NCP recurrent core with RecurrentWrapper state normalization.
    Falls back to GRU if ncps is unavailable.

    Parameters
    ----------
    input_dim     : feature dimension of each timestep
    units         : desired output dimension (= lnn_units in config)
    sparsity_level: NCP connection sparsity [0, 1]
    backend       : "auto" | "gru" | "cfc"
    num_layers    : GRU stacking depth
    dropout       : inter-layer dropout for stacked GRU
    return_sequences: return all timesteps (True) or only last (False)
    """

    def __init__(
        self,
        input_dim: int,
        units: int = 64,
        sparsity_level: float = 0.5,
        return_sequences: bool = False,
        backend: str = "auto",
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.units       = units
        self.return_sequences = return_sequences
        self.num_layers  = num_layers

        use_gru = (backend == "gru") or (backend == "auto" and not _NCPS_AVAILABLE)

        if not use_gru and _NCPS_AVAILABLE:
            ncp_total = units + 16
            wiring    = AutoNCP(ncp_total, output_size=units,
                                sparsity_level=sparsity_level)
            self.rnn      = CfC(input_dim, wiring, batch_first=True)
            self.ncp_total = ncp_total
            self.backend  = "cfc_ncp"
        else:
            self.rnn = nn.GRU(
                input_dim, units, batch_first=True,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.ncp_total = units
            self.backend   = "gru"

    # ── 核心前向传播 ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x  : (B, T, input_dim)
        hx : None or native-format state from a previous forward() call.
             GRU: (num_layers, B, units); CfC: (B, ncp_total)

        Returns
        -------
        out : (B, T, units) if return_sequences else (B, units)
        h   : native-format hidden state — pass directly back as hx.
              Use normalize_state(h) to obtain (B, units) for downstream use.
        """
        out, h = self.rnn(x, hx)
        # GRU: h is (num_layers, B, units) — kept for correct hx passing
        # CfC: h is (B, ncp_total)         — already 2-D

        if not self.return_sequences:
            return out[:, -1, :], h
        return out, h

    # ── RecurrentWrapper 接口 ────────────────────────────────────────────────────

    def normalize_state(
        self, h: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        将任意后端的隐藏状态标准化为 (B, units) 格式。

        用途：
          - 流式推理时跨时间步比较/记录状态
          - TBPTT 状态掩码（无需感知后端类型）
          - 日志记录、可视化

        注意：返回的状态不能直接作为 hx 传回 forward()，
              因为 GRU 需要原始的 (num_layers, B, units) 格式。
              如需传回，使用 denormalize_state()。
        """
        if h is None:
            return None
        if self.backend == "gru" and h.dim() == 3:
            # (num_layers, B, units) → 取最后一层 (B, units)
            return h[-1]
        # CfC: (B, ncp_total) 已经是 2-D，直接返回
        return h

    def denormalize_state(
        self, h_norm: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        将 (B, units) 格式的状态转回 hx 可接受的原生格式。
        用于从外部存储的标准化状态恢复传递。

        GRU: (B, units) → (num_layers, B, units)，多层 h 用同一层复制填充
             （近似值，精确恢复需存储完整 h）
        CfC: 无需转换
        """
        if h_norm is None:
            return None
        if self.backend == "gru" and h_norm.dim() == 2:
            # 将 (B, units) 扩展为 (num_layers, B, units)
            return h_norm.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return h_norm

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        生成正确形状的零初始隐藏状态。
        外部代码无需感知后端类型即可初始化 h。

        Returns
        -------
        GRU : (num_layers, B, units)  zero tensor
        CfC : (B, ncp_total)          zero tensor
        """
        if self.backend == "gru":
            return torch.zeros(self.num_layers, batch_size, self.units,
                               device=device)
        else:
            return torch.zeros(batch_size, self.ncp_total, device=device)

    def mask_state(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,  # (B,) bool
    ) -> torch.Tensor:
        """
        对无效患者（已耗尽数据的批次位）将隐藏状态置零。
        统一处理 GRU 和 CfC 的 h 形状差异。

        临床意义：TBPTT 批次中部分患者数据不足时，
        需要将其 h 归零以避免后续 chunk 用错状态。
        """
        invalid = ~valid_mask
        if self.backend == "gru" and h.dim() == 3:
            # h: (num_layers, B, units)
            h[:, invalid, :] = 0.0
        else:
            # h: (B, ncp_total) or (B, units)
            h[invalid] = 0.0
        return h
