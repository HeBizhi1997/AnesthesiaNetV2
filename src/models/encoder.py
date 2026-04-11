"""
CNN Waveform Encoder — dilated convolutions with residual projection.

感受野设计（k=7，dilation 1/2/4/16）：
  Layer 1: dilation=1  → RF ~55 ms   — 瞬时振幅、波形细节
  Layer 2: dilation=2  → RF ~200 ms  — θ/α 节律（4-13 Hz）
  Layer 3: dilation=4  → RF ~406 ms  — δ 波一个周期（约 400 ms）
  Layer 4: dilation=16 → RF ~1100 ms — 爆发抑制静默期（BSR：0.5-2 s）★新增
  GlobalAvgPool → 整个 4 s 窗口的完整感受野

v8 新增：bsr_layer=True（默认开启）
  在主卷积栈后追加一层 dilation=16 + 残差跳跃连接，专门捕捉深度麻醉时的
  爆发-抑制静默期（Burst Suppression）。BSR 是 BIS < 40 的关键临床标志，
  原始 3 层设计仅覆盖到 ~406 ms，无法检测 0.5-2 s 的抑制段。

临床意义：
  BSR（Burst Suppression Ratio）直接决定 BIS 的低值估计精度。
  dilation=16 使 CNN 对抑制期"静默模式"敏感，提升深麻醉段（BIS < 40）的预测。

Two projection modes (controlled by global_pool):

  global_pool=True  (default, recommended for EEG windows):
    CNN → BSR(optional) → GlobalAvgPool → Linear(128,256) + skip → 128
    参数量: ~95K (无BSR) / ~130K (有BSR)

  global_pool=False  (legacy, large):
    CNN → MaxPool1d×3 → flat → 512 → 128
    仅在空间位置信息关键时使用（不常用）。

Input : (batch, n_channels, window_samples)
Output: (batch, out_dim=128)
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class WaveformEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int = 2,
        window_samples: int = 512,    # 4 s × 128 Hz
        conv_channels: List[int] = (32, 64, 128),
        kernel_size: int = 7,
        global_pool: bool = True,     # True = lightweight; False = legacy large
        bsr_layer: bool = True,       # ★ 新增：dilation=16 的 BSR 检测层
        use_grad_checkpoint: bool = True,  # 梯度检查点：用计算换内存
    ):
        """
        bsr_layer : 是否添加 dilation=16 的 BSR 专用层。
            True（默认）: 在 conv_channels[-1] 通道上追加一层 dilation=16 卷积 + 残差，
                          使感受野从 ~406 ms 扩展到 ~1.1 s，专门捕捉爆发抑制静默期。
            False : 与原始 v7 架构完全相同。
        """
        super().__init__()
        self._bsr_layer = bsr_layer

        # ── 主卷积栈（dilation 1 / 2 / 4）─────────────────────────────────────
        dilations = [1 << i for i in range(len(conv_channels))]   # [1, 2, 4, ...]

        layers: List[nn.Module] = []
        in_ch = n_channels
        for out_ch, dil in zip(conv_channels, dilations):
            padding = dil * (kernel_size // 2)   # same-length output
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=padding, dilation=dil),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            if not global_pool:
                layers.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        cnn_out = conv_channels[-1]   # 128

        # ── BSR 层：dilation=16，~1.1 s 感受野 ─────────────────────────────────
        # 感受野计算：RF_prev(≈406ms) + (k-1)×d = 43 + 6×16 = 139 samples ≈ 1.09 s
        # 残差跳跃连接：防止梯度消失，允许模型选择是否使用 BSR 特征
        if bsr_layer and global_pool:
            bsr_dil = 16
            bsr_pad = bsr_dil * (kernel_size // 2)
            self.bsr_conv = nn.Sequential(
                nn.Conv1d(cnn_out, cnn_out, kernel_size=kernel_size,
                          padding=bsr_pad, dilation=bsr_dil),
                nn.BatchNorm1d(cnn_out),
                nn.GELU(),
            )
            # 可学习的跳跃权重：初始化为 sigmoid(-3)≈0.047，近似禁用 BSR 层
            # 让模型先用主干 CNN 建立稳定表征，之后 gate 随训练自适应打开
            # sigmoid(0.5)=0.62 的初始值在早期 epoch 引入过多随机噪声
            self.bsr_gate = nn.Parameter(torch.tensor(-3.0))
        else:
            self.bsr_conv = None
            self.bsr_gate = None

        if global_pool:
            self.pool = nn.AdaptiveAvgPool1d(1)   # → (B, cnn_out, 1)
        else:
            self.pool = None

        # ── 残差投影 ─────────────────────────────────────────────────────────────
        if global_pool:
            # Lightweight residual projection: 128 → 256 → 128 + skip 128 → 128
            flat_dim = cnn_out
            self.proj_main = nn.Sequential(
                nn.Linear(flat_dim, flat_dim * 2),
                nn.GELU(),
                nn.Linear(flat_dim * 2, flat_dim),
                nn.LayerNorm(flat_dim),
                nn.GELU(),
            )
            self.proj_skip = nn.Linear(flat_dim, flat_dim, bias=False)
        else:
            # Legacy: large bottleneck from full spatial flatten
            length = window_samples
            for _ in conv_channels:
                length = length // 2
            flat_dim = cnn_out * length   # e.g. 128 × 64 = 8192
            self.proj_main = nn.Sequential(
                nn.Linear(flat_dim, 512),
                nn.GELU(),
                nn.Linear(512, cnn_out),
                nn.LayerNorm(cnn_out),
                nn.GELU(),
            )
            self.proj_skip = nn.Linear(flat_dim, cnn_out, bias=False)

        self.out_dim = cnn_out
        self._global_pool = global_pool
        self._use_grad_ckpt = use_grad_checkpoint

    def _conv_bsr(self, wave: torch.Tensor) -> torch.Tensor:
        """CNN stack + optional BSR layer — wrapped for gradient checkpointing."""
        x = self.conv(wave)
        if self.bsr_conv is not None:
            gate = torch.sigmoid(self.bsr_gate)
            x = x + gate * self.bsr_conv(x)
        return x

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """
        wave: (B, n_channels, window_samples)
        """
        # Gradient checkpointing: recompute CNN activations during backward
        # instead of storing them — saves ~5 GB at B*T=9600, cost: ~20% slower backward.
        # Use self.training (not wave.requires_grad — input data never requires grad).
        if self._use_grad_ckpt and self.training:
            x = grad_checkpoint(self._conv_bsr, wave, use_reentrant=False)
        else:
            x = self._conv_bsr(wave)

        # Pool
        if self.pool is not None:
            x = self.pool(x)              # (B, cnn_out, 1)

        flat = x.flatten(1)              # (B, flat_dim)
        return self.proj_main(flat) + self.proj_skip(flat)   # (B, out_dim)
