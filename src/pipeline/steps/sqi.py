"""
Signal Quality Index (SQI) computation.

Scores each channel on [0, 1]. A score < threshold indicates the
window is too noisy for reliable BIS estimation; the LNN core
should enter "lazy mode" (hold last output) for such windows.

Three sub-scores combined with harmonic mean:
  1. Autocorrelation score — healthy EEG is smooth; noise is jagged.
  2. Kurtosis score       — normal EEG is near-Gaussian (kurtosis ~3).
  3. Spectral score       — fraction of power in 0.5-47 Hz should be high.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import welch

from ..base import EEGStep
from ..context import EEGContext


class SQIComputer(EEGStep):
    def __init__(self, cfg: Dict[str, Any]):
        self.min_score = cfg.get("min_score", 0.5)
        self.kurtosis_thresh = cfg.get("kurtosis_thresh", 5.0)
        self.high_freq_ratio_thresh = cfg.get("high_freq_ratio_thresh", 0.4)

    def process(self, ctx: EEGContext) -> EEGContext:
        scores = np.zeros(ctx.n_channels, dtype=np.float32)
        for ch in range(ctx.n_channels):
            x = ctx.data[ch].astype(np.float64)

            # --- Clipping / flat-line guard ---
            if np.ptp(x) < 1e-6:   # completely flat → electrode off
                scores[ch] = 0.0
                continue

            # 1. Autocorrelation smoothness
            ac = np.correlate(x - x.mean(), x - x.mean(), mode="full")
            ac = ac[len(ac) // 2:]
            ac /= ac[0] + 1e-12
            # A smooth signal has high AC at lag-1
            ac_score = float(np.clip(ac[1], 0, 1))

            # 2. Kurtosis (excess): healthy EEG ≈ 0; impulse noise >> 0
            kurt = float(kurtosis(x, fisher=True))
            # Map kurtosis to [0,1]: 0→1.0, thresh→0.0
            kurtosis_score = float(np.clip(1.0 - abs(kurt) / self.kurtosis_thresh, 0, 1))

            # 3. Spectral quality: power fraction in EEG band vs total
            f, pxx = welch(x, fs=ctx.fs, nperseg=min(256, len(x)))
            eeg_mask = (f >= 0.5) & (f <= 47.0)
            total_pow = pxx.sum() + 1e-12
            eeg_pow = pxx[eeg_mask].sum()
            spectral_score = float(np.clip(eeg_pow / total_pow, 0, 1))
            # penalise if high-freq ratio is too large (ESU indicator)
            high_mask = f > 50.0
            high_ratio = pxx[high_mask].sum() / total_pow
            if high_ratio > self.high_freq_ratio_thresh:
                spectral_score *= 0.5

            # Harmonic mean of the three sub-scores
            sub = np.array([ac_score, kurtosis_score, spectral_score], dtype=np.float64)
            sub = np.clip(sub, 1e-6, 1.0)
            scores[ch] = float(len(sub) / np.sum(1.0 / sub))

        ctx.sqi = scores
        ctx.artifacts["sqi_scores"] = scores.copy()
        return ctx

    def validate(self, ctx: EEGContext) -> None:
        super().validate(ctx)
        assert ctx.sqi is not None, "SQIComputer did not set ctx.sqi"
