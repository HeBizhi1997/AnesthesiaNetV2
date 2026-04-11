"""
AnesthesiaNetV2 - Multi-task architecture for surgical phase-aware BIS prediction.

Architecture overview:
  ┌──────────────────────────────────────────────────────────┐
  │  EEG windows (B,T,n_ch,W)  features (B,T,F)  sqi (B,T,n_ch) │
  └──────────────────┬───────────────────────────────────────┘
                     │
             ┌───────▼────────┐
             │ WaveformEncoder│  dilated CNN + residual skip  (B*T, 128)
             │ FeatProjector  │  hand-crafted F-dim → 64       (B*T,  64)
             │ FusionLayer    │  concat + linear → d_model     (B*T, d_model)
             └───────┬────────┘
                     │ seq: (B, T, d_model)
             ┌───────▼────────┐
             │   LNNCore      │  GRU (default) or CfC/NCP
             │  + SQI Inertia │  freeze output at low-SQI steps
             └───┬───┬───┬───┘
                 │   │   │
         ┌───────┘   │   └──────────────┐
         │           │                  │
  ┌──────▼──────┐ ┌──▼──────────┐ ┌────▼────────────┐
  │ Phase Head  │ │  Stim Head  │ │  BIS Head       │
  │ 4-class CE  │ │ binary BCE  │ │  phase-gated    │
  └──────┬──────┘ └──┬──────────┘ └────┬────────────┘
         │  phase_prob│                 │
         └────────────┴─────────────────┘
                           │
                 ┌─────────▼─────────┐
                 │  Stim Corrector   │
                 │  BIS_final = raw  │
                 │  + f(stim_prob)   │
                 └─────────┬─────────┘
                       pred_bis (B, T, 1)

Changes vs original V2:
  - WaveformEncoder (encoder.py, residual bottleneck) replaces DilatedCNNEncoder
  - LNNCore (lnn_core.py) replaces bare nn.GRU → backend selectable via config
    (backend="gru" → cuDNN fused GRU, same weights/speed as before)
    (backend="cfc" → CfC/NCP with adaptive time constants, requires ncps)
  - SQI Inertia Mode: after temporal encoding, timesteps where mean SQI < threshold
    propagate the previous clean state forward instead of the noisy update.
    Prevents ESU/artifact windows from corrupting the temporal context.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import WaveformEncoder
from .lnn_core import LNNCore


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class PhaseGatedBISHead(nn.Module):
    """
    BIS regression head conditioned on phase probabilities.

    Each phase has a dedicated linear projection. The final prediction
    is a soft mixture: pred = Σ P(phase_i) * head_i(h)
    This allows phase-specific bias and scaling without hard routing.
    """

    def __init__(self, in_dim: int, n_phases: int = 4):
        super().__init__()
        self.phase_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Linear(in_dim // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_phases)
        ])
        self.n_phases = n_phases

    def forward(self, h: torch.Tensor,
                phase_probs: torch.Tensor) -> torch.Tensor:
        """
        h           : (B, T, in_dim)
        phase_probs : (B, T, n_phases)  — softmax probabilities
        returns     : (B, T, 1)
        """
        phase_preds = torch.stack(
            [head(h).squeeze(-1) for head in self.phase_heads], dim=-1
        )   # (B, T, n_phases)
        pred = (phase_preds * phase_probs).sum(-1, keepdim=True)   # (B, T, 1)
        return pred


class StimulationCorrector(nn.Module):
    """
    Learns a correction offset when a stimulation event is detected.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.stim_scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, h: torch.Tensor,
                stim_prob: torch.Tensor) -> torch.Tensor:
        """
        h         : (B, T, in_dim)
        stim_prob : (B, T, 1)
        returns   : (B, T, 1)
        """
        inp = torch.cat([h, stim_prob], dim=-1)
        correction = self.net(inp) * stim_prob * self.stim_scale.clamp(0.0, 0.3)
        return correction


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class AnesthesiaNetV2(nn.Module):
    """
    Multi-task anesthesia depth monitor.

    Tasks:
      1. Phase classification  (4 classes: pre-op / induction / maintenance / recovery)
      2. Stimulation detection (binary)
      3. BIS regression        (phase-gated)

    Architectural improvements over the original V2:
      - WaveformEncoder: shared encoder from encoder.py with residual bottleneck skip
      - LNNCore: configurable GRU or CfC/NCP temporal model
      - SQI Inertia Mode: hold last-clean-state output at low-SQI timesteps
    """

    N_PHASES = 4

    def __init__(
        self,
        n_eeg_channels: int   = 2,
        cnn_channels: list    = None,
        cnn_kernel: int       = 7,
        window_samples: int   = 512,   # 4 s × 128 Hz
        feature_dim: int      = 24,
        d_model: int          = 128,
        gru_layers: int       = 2,
        gru_dropout: float    = 0.15,
        lnn_backend: str      = "gru",  # "gru" | "cfc" | "auto"
        sqi_inertia_threshold: float = 0.5,  # 0 = disabled
        bsr_layer: bool       = True,   # dilation=16 BSR detection layer
        grad_checkpoint: bool = True,   # gradient checkpointing to save VRAM
    ):
        super().__init__()
        cnn_channels = cnn_channels or [32, 64, 128]

        # ── Shared per-window encoder ─────────────────────────────────────────
        # WaveformEncoder: dilated CNN + residual bottleneck skip → 128-dim
        self.wave_enc = WaveformEncoder(
            n_channels=n_eeg_channels,
            window_samples=window_samples,
            conv_channels=cnn_channels,
            kernel_size=cnn_kernel,
            global_pool=True,   # lightweight: GlobalAvgPool → residual 128→256→128
            bsr_layer=bsr_layer,
            use_grad_checkpoint=grad_checkpoint,
        )
        cnn_out = self.wave_enc.out_dim   # 128

        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim + n_eeg_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ── Temporal model: LNNCore (GRU by default, CfC optional) ───────────
        self.temporal = LNNCore(
            input_dim=d_model,
            units=d_model,
            return_sequences=True,
            backend=lnn_backend,
            num_layers=gru_layers,
            dropout=gru_dropout,
        )

        # SQI inertia threshold — 0 means disabled
        self.sqi_inertia_threshold = sqi_inertia_threshold

        # ── Task heads ────────────────────────────────────────────────────────
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, self.N_PHASES),
        )

        self.stim_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.bis_head = PhaseGatedBISHead(d_model, self.N_PHASES)
        self.stim_corrector = StimulationCorrector(d_model)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        wave: torch.Tensor,       # (B, T, n_ch, win_samp)
        features: torch.Tensor,   # (B, T, n_feat)
        sqi: torch.Tensor,        # (B, T, n_ch)
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pred_bis    : (B, T, 1)   — BIS prediction [0, 1]
        phase_logits: (B, T, 4)   — raw logits for phase classification
        stim_logits : (B, T, 1)   — raw logit for stimulation event
        correction  : (B, T, 1)   — stimulation correction applied to BIS
        h           : hidden state for streaming (GRU: 3-D, CfC: 2-D)
        """
        B, T, n_ch, W = wave.shape

        # ── Per-window encoding ───────────────────────────────────────────────
        wave_flat = wave.view(B * T, n_ch, W)
        wave_emb  = self.wave_enc(wave_flat)                        # (B*T, 128)

        feat_flat = features.view(B * T, -1)
        sqi_flat  = sqi.view(B * T, -1)
        feat_emb  = self.feat_proj(
            torch.cat([feat_flat, sqi_flat], -1)
        )                                                            # (B*T, 64)

        fused = self.fusion(torch.cat([wave_emb, feat_emb], -1))    # (B*T, d_model)
        seq   = fused.view(B, T, -1)                                 # (B, T, d_model)

        # ── Temporal model ────────────────────────────────────────────────────
        h_seq, h = self.temporal(seq, hx)                           # (B, T, d_model)

        # ── SQI Inertia Mode ──────────────────────────────────────────────────
        # When per-window mean SQI falls below threshold, propagate the last
        # clean hidden state forward rather than the artifact-corrupted update.
        # This prevents ESU/electrode-pop events from derailing the BIS estimate.
        #
        # Performance-critical: the naive Python loop creates a T=300 autograd chain
        # identical to the EMA bug, causing 5-10× slowdown in backward.
        # Fix: precompute "last valid source index" in no_grad (O(T) scan, no autograd),
        # then do a single gather() — autograd depth = O(1) instead of O(T).
        if self.sqi_inertia_threshold > 0.0 and T > 1:
            sqi_ok = (sqi.mean(-1) >= self.sqi_inertia_threshold)   # (B, T) bool
            with torch.no_grad():
                # For each (b, t): last_src[b, t] = last index <= t where sqi_ok
                last_src = torch.arange(T, device=h_seq.device).unsqueeze(0).expand(B, -1).clone()
                for t in range(1, T):
                    carry = ~sqi_ok[:, t]                            # (B,) — bad window
                    last_src[:, t] = torch.where(carry, last_src[:, t - 1], last_src[:, t])
                # Expand to (B, T, D) for gather
                idx = last_src.unsqueeze(-1).expand(-1, -1, h_seq.shape[-1])
            # Single gather: gradient flows to all valid h_seq positions in O(1) depth
            h_seq = h_seq.gather(1, idx)                             # (B, T, D)

        # ── Task heads ────────────────────────────────────────────────────────
        phase_logits = self.phase_head(h_seq)                        # (B, T, 4)
        stim_logits  = self.stim_head(h_seq)                         # (B, T, 1)

        phase_probs  = F.softmax(phase_logits, dim=-1)               # (B, T, 4)
        stim_prob    = torch.sigmoid(stim_logits)                    # (B, T, 1)

        bis_raw    = self.bis_head(h_seq, phase_probs)               # (B, T, 1)
        correction = self.stim_corrector(h_seq, stim_prob)           # (B, T, 1)
        pred_bis   = (bis_raw + correction).clamp(0.0, 1.0)          # (B, T, 1)

        return pred_bis, phase_logits, stim_logits, correction, h

    @classmethod
    def from_config(cls, cfg: dict) -> "AnesthesiaNetV2":
        m = cfg["model"]
        t = cfg["training"]
        eeg = cfg["eeg"]
        wind = cfg["windowing"]
        win_samp = int(wind["window_sec"] * eeg["srate"])
        return cls(
            n_eeg_channels=len(eeg["channels"]),
            cnn_channels=m.get("cnn_channels", [32, 64, 128]),
            cnn_kernel=m.get("cnn_kernel", 7),
            window_samples=win_samp,
            feature_dim=m.get("feature_dim", 24),
            d_model=m.get("d_model", 128),
            gru_layers=m.get("gru_layers", 2),
            gru_dropout=m.get("gru_dropout", 0.15),
            lnn_backend=m.get("lnn_backend", "gru"),
            sqi_inertia_threshold=m.get("sqi_inertia_threshold", 0.5),
            bsr_layer=m.get("bsr_layer", True),
            grad_checkpoint=m.get("grad_checkpoint", True),
        )
