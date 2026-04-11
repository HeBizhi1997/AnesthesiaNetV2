"""
AnesthesiaNet — Full model for anesthesia depth estimation.

Architecture:

  ┌──────────────────────────────────────────────────────────┐
  │  Input per timestep t:                                    │
  │    wave_t    : (B, n_ch, win_samp)  filtered EEG          │
  │    features_t: (B, n_feat)          hand-crafted features  │
  │    sqi_t     : (B, n_ch)            signal quality         │
  └──────────────────────────────────────────────────────────┘
           │                       │
           ▼                       ▼
   WaveformEncoder          FeatureProjector
    (Dilated 1D-CNN)          (Linear + LN)
           │                       │
           └───────────┬───────────┘
                       ▼
              Concatenate + fuse
                       ▼
                  LNN Core (CfC/NCP)   ← return_sequences=True
                  [across time axis]
                       ▼
             Regression Head applied at every timestep
               pred_seq : (B, T, 1)
                       ▼
             pred = pred_seq[:, -1, :]  returned as main output

Usage:
  pred, pred_seq, h = model(wave, features, sqi)
  - pred     : (B, 1)    BIS at last timestep in [0, 1]
  - pred_seq : (B, T, 1) BIS at every timestep (for L_Mono during training)
  - h        : (B, ncp_total) hidden state for streaming inference
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .encoder import WaveformEncoder
from .lnn_core import LNNCore


class AnesthesiaNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 2,
        window_samples: int = 512,
        n_features: int = 24,
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
        cnn_kernel: int = 7,
        lnn_units: int = 64,
        lnn_sparsity: float = 0.5,
        lnn_backend: str = "auto",
        dropout: float = 0.2,
    ):
        super().__init__()

        # --- Waveform branch ---
        self.wave_enc = WaveformEncoder(
            n_channels=n_channels,
            window_samples=window_samples,
            conv_channels=list(cnn_channels),
            kernel_size=cnn_kernel,
        )
        wave_dim = self.wave_enc.out_dim  # 128

        # --- Feature branch: features + sqi concatenated ---
        feat_in = n_features + n_channels
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        feat_dim = 64

        # --- Fusion ---
        fused_dim = wave_dim + feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, lnn_units),
            nn.LayerNorm(lnn_units),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- LNN Core  (return_sequences=True → (B, T, units)) ---
        self.lnn = LNNCore(
            input_dim=lnn_units,
            units=lnn_units,
            sparsity_level=lnn_sparsity,
            return_sequences=True,
            backend=lnn_backend,
        )

        # --- Regression head (applied to every timestep) ---
        # Sigmoid maps output to [0, 1]; ×100 at inference for BIS scale
        self.head = nn.Sequential(
            nn.Linear(lnn_units, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._n_features = n_features
        self._n_channels = n_channels

    def forward(
        self,
        wave: torch.Tensor,       # (B, seq_len, n_ch, win_samp)
        features: torch.Tensor,   # (B, seq_len, n_feat)
        sqi: torch.Tensor,        # (B, seq_len, n_ch)
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pred     : (B, 1)     — BIS prediction for last timestep in [0, 1]
        pred_seq : (B, T, 1)  — BIS at every timestep (use for L_Mono)
        h        : (B, ncp_total) — hidden state for streaming
        """
        B, T, n_ch, W = wave.shape

        # ── Vectorized encoding: merge T into batch dim ──────────────────
        # Old: for t in range(T): CNN(wave[:,t])  → T serial GPU kernel calls
        # New: CNN(wave.view(B*T, n_ch, W))       → 1 GPU kernel call on B×T items
        # Speedup ≈ T× (=10×) for the encoder stage.

        wave_flat = wave.view(B * T, n_ch, W)                            # (B*T, n_ch, W)
        wave_emb  = self.wave_enc(wave_flat)                             # (B*T, 128)

        feat_flat = features.view(B * T, -1)                             # (B*T, n_feat)
        sqi_flat  = sqi.view(B * T, -1)                                  # (B*T, n_ch)
        feat_emb  = self.feat_proj(torch.cat([feat_flat, sqi_flat], -1)) # (B*T, 64)

        fused = self.fusion(torch.cat([wave_emb, feat_emb], -1))         # (B*T, lnn_units)
        seq   = fused.view(B, T, -1)                                     # (B, T, lnn_units)
        lnn_out, h = self.lnn(seq, hx)          # (B, T, units), (B, ncp_total)

        pred_seq = self.head(lnn_out)            # (B, T, 1)
        pred = pred_seq[:, -1, :]               # (B, 1) — last timestep

        return pred, pred_seq, h

    @torch.no_grad()
    def predict_single(
        self,
        wave: torch.Tensor,       # (n_ch, win_samp) — one window
        features: torch.Tensor,   # (n_feat,)
        sqi: torch.Tensor,        # (n_ch,)
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Convenience wrapper for real-time single-step inference."""
        wave = wave.unsqueeze(0).unsqueeze(0)           # (1, 1, n_ch, win_samp)
        features = features.unsqueeze(0).unsqueeze(0)   # (1, 1, n_feat)
        sqi = sqi.unsqueeze(0).unsqueeze(0)             # (1, 1, n_ch)
        pred, _, h = self.forward(wave, features, sqi, hx)
        bis = float(pred[0, 0]) * 100.0
        return bis, h

    @classmethod
    def from_config(cls, cfg: dict) -> "AnesthesiaNet":
        m = cfg["model"]
        n_ch = len(cfg["eeg"]["channels"])
        win = int(cfg["windowing"]["window_sec"] * cfg["eeg"]["srate"])
        n_feat = m.get("feature_dim", 24)
        backend = m.get("lnn_backend", "auto")
        return cls(
            n_channels=n_ch,
            window_samples=win,
            n_features=n_feat,
            cnn_channels=tuple(m["cnn_channels"]),
            cnn_kernel=m["cnn_kernel"],
            lnn_units=m["lnn_units"],
            lnn_sparsity=m["lnn_sparsity"],
            lnn_backend=backend,
        )
