"""
Spectral Entropy for anesthesia depth monitoring.

Implements GE Datex-Ohmeda Entropy Module algorithm:
  Reference: Viertiö-Oja et al., Acta Anaesthesiol Scand 2004;48:154-161

State Entropy  (SE): Shannon spectral entropy over 0.8–32 Hz  → scaled [0, 91]
Response Entropy (RE): Shannon spectral entropy over 0.8–47 Hz  → scaled [0, 100]

Stability design:
  1. 8-second rolling buffer → stable Welch PSD (4-s window, 0.25 Hz resolution)
  2. EMA temporal smoothing (α=0.30, ~3.3-epoch time constant)
  3. Artifact rejection: flatline, saturation, sudden amplitude spike

Clinical thresholds (GE standard):
  SE ≥ 60   likely awake / light sedation
  SE 40–60  optimal anesthesia zone
  SE < 40   deep anesthesia
  RE – SE ≥ 10  EMG present → possible nociception
"""

import collections
import numpy as np
from scipy.signal import welch
from loguru import logger


class EntropyProcessor:

    SE_LO, SE_HI = 0.8, 32.0   # Hz boundaries for State Entropy
    RE_LO, RE_HI = 0.8, 47.0   # Hz boundaries for Response Entropy (includes EMG)

    BUFFER_SEC = 8              # Rolling PSD window length (seconds)
    EMA_ALPHA  = 0.30           # Temporal smoothing factor (~3.3-epoch time constant)

    # Artifact thresholds — calibrated for BIS-monitor EEG (VitalDB data)
    # Large slow waves during deep anesthesia regularly reach 200-400 µV;
    # only true lead-off (>3000 µV) or digital saturation should be rejected.
    FLATLINE_STD   = 0.05       # µV — below this → disconnected electrode
    SATURATION_AMP = 3000.0     # µV — raised from 1000; valid delta can be 300-500 µV
    SPIKE_RATIO    = 20.0       # epoch std > 20× recent baseline → artefact

    def __init__(self, sample_rate: int = 256):
        self.fs = sample_rate
        self._buf: collections.deque = collections.deque(
            maxlen=self.BUFFER_SEC * sample_rate
        )
        self._se_ema: float | None = None
        self._re_ema: float | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, eeg: np.ndarray) -> dict:
        """
        eeg: 1-D float array, one new chunk (any length ≥ 32 samples).
        Accumulates data in an 8-second rolling buffer and applies EMA
        smoothing to suppress epoch-to-epoch jitter.
        Returns {"se": float|None, "re": float|None}.
        """
        if eeg.ndim != 1 or len(eeg) < 32:
            return {"se": self._se_ema, "re": self._re_ema}

        # Reject artifactual chunks — keep last EMA, don't corrupt the buffer
        if self._is_artifact(eeg):
            logger.debug("Entropy: artifact chunk skipped")
            return {"se": self._se_ema, "re": self._re_ema}

        self._buf.extend(eeg.tolist())

        # Wait until buffer has at least 2 seconds of clean data
        if len(self._buf) < 2 * self.fs:
            return {"se": None, "re": None}

        buf = np.array(self._buf)
        try:
            freqs, psd = self._psd(buf)
            se_raw = self._band_entropy(psd, freqs, self.SE_LO, self.SE_HI)
            re_raw = self._band_entropy(psd, freqs, self.RE_LO, self.RE_HI)

            se = float(np.clip(se_raw * 91.0,  0.0, 91.0))
            re = float(np.clip(re_raw * 100.0, 0.0, 100.0))

            # EMA smoothing
            if self._se_ema is None:
                self._se_ema, self._re_ema = se, re
            else:
                a = self.EMA_ALPHA
                self._se_ema = a * se + (1.0 - a) * self._se_ema
                self._re_ema = a * re + (1.0 - a) * self._re_ema

            logger.debug(
                f"Entropy  SE={self._se_ema:.1f}  RE={self._re_ema:.1f}  "
                f"RE-SE={self._re_ema - self._se_ema:.1f}  "
                f"buf={len(self._buf)}/{self._buf.maxlen}"
            )
            return {"se": self._se_ema, "re": self._re_ema}

        except Exception as exc:
            logger.warning(f"Entropy computation failed: {exc}")
            return {"se": self._se_ema, "re": self._re_ema}

    def reset(self):
        """Call at the start of a new monitoring session to clear stale state."""
        self._buf.clear()
        self._se_ema = None
        self._re_ema = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_artifact(self, eeg: np.ndarray) -> bool:
        """
        Returns True when the chunk is clearly artifactual.
        Detects: flatline/disconnection, hard saturation, sudden amplitude spike.
        """
        std     = float(np.std(eeg))
        max_amp = float(np.max(np.abs(eeg)))

        if std < self.FLATLINE_STD:
            return True
        if max_amp > self.SATURATION_AMP:
            return True

        # Sudden spike: compare vs. last 1-second baseline in buffer
        if len(self._buf) >= self.fs:
            recent_std = float(np.std(list(self._buf)[-self.fs:]))
            if recent_std > 0 and std > self.SPIKE_RATIO * recent_std:
                return True

        return False

    def _psd(self, eeg: np.ndarray):
        """
        Welch PSD on the full rolling buffer.
        4-second window → 0.25 Hz frequency resolution.
        8-second buffer → 3 overlapping segments → stable spectral estimate.
        """
        nperseg = min(4 * self.fs, len(eeg))
        return welch(eeg, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2)

    @staticmethod
    def _band_entropy(psd: np.ndarray, freqs: np.ndarray,
                      f_lo: float, f_hi: float) -> float:
        """
        Normalized Shannon spectral entropy ∈ [0, 1] for [f_lo, f_hi].

          p_i   = P_i / ΣP_i        (probability distribution within band)
          H     = -Σ p_i · ln(p_i)  (Shannon entropy, nats)
          H_max = ln(N_bins)         (maximum for uniform distribution)
          return H / H_max
        """
        mask   = (freqs >= f_lo) & (freqs <= f_hi)
        n_bins = int(mask.sum())
        if n_bins < 2:
            return 0.0

        p     = psd[mask]
        total = float(p.sum())
        if total < 1e-15:
            return 0.0

        p     = p / total
        p_nz  = p[p > 1e-15]
        H     = -float(np.sum(p_nz * np.log(p_nz)))
        H_max = float(np.log(n_bins))

        return float(np.clip(H / H_max, 0.0, 1.0))
