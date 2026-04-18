"""
Spectral Entropy for anesthesia depth monitoring.

Implements GE Datex-Ohmeda Entropy Module algorithm:
  Reference: Viertiö-Oja et al., Acta Anaesthesiol Scand 2004;48:154-161

State Entropy  (SE): Shannon spectral entropy over 0.8–32 Hz  → scaled [0, 91]
Response Entropy (RE): Shannon spectral entropy over 0.8–47 Hz  → scaled [0, 100]

Algorithm:
  1. Estimate PSD via Welch (4-second window, 50% overlap)
  2. Normalize PSD within each band → probability distribution p_i
  3. Shannon entropy  H = -Σ p_i · ln(p_i)
  4. Normalize:       H_norm = H / ln(N_bins)   ∈ [0, 1]
  5. Scale:           SE = H_norm × 91,  RE = H_norm × 100

Verification:
  - White noise (uniform PSD)  → SE = 91, RE = 100  (maximum disorder)
  - Pure sine wave (1 bin)     → SE = 0,  RE = 0    (perfect regularity)
  - Deep anesthesia (narrow delta) → SE ≈ 20–40
  - Awake / light sedation     → SE ≈ 60–85

Clinical thresholds (GE standard):
  SE ≥ 60   likely awake / light
  SE 40–60  optimal anesthesia zone
  SE < 40   deep anesthesia
  RE – SE ≥ 10  EMG present → possible nociception
"""

import numpy as np
from scipy.signal import welch
from loguru import logger


class EntropyProcessor:

    # Frequency band boundaries (Hz)
    SE_LO, SE_HI = 0.8, 32.0
    RE_LO, RE_HI = 0.8, 47.0

    def __init__(self, sample_rate: int = 256):
        self.fs = sample_rate

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, eeg: np.ndarray) -> dict:
        """
        eeg : 1-D float array (n_samples,) – single EEG channel.
        Returns dict with keys 'se' (float|None) and 're' (float|None).
        Minimum required length: 2 × fs samples.
        """
        if eeg.ndim != 1 or len(eeg) < 2 * self.fs:
            return {"se": None, "re": None}

        try:
            freqs, psd = self._psd(eeg)
            se_raw = self._band_entropy(psd, freqs, self.SE_LO, self.SE_HI)
            re_raw = self._band_entropy(psd, freqs, self.RE_LO, self.RE_HI)

            se = float(np.clip(se_raw * 91.0,  0.0, 91.0))
            re = float(np.clip(re_raw * 100.0, 0.0, 100.0))

            logger.debug(f"Entropy  SE={se:.1f}  RE={re:.1f}  RE-SE={re-se:.1f}")
            return {"se": se, "re": re}
        except Exception as exc:
            logger.warning(f"Entropy computation failed: {exc}")
            return {"se": None, "re": None}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _psd(self, eeg: np.ndarray):
        """Welch PSD with 4-second window → ~0.25 Hz frequency resolution."""
        nperseg = min(4 * self.fs, len(eeg))
        return welch(eeg, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2)

    @staticmethod
    def _band_entropy(psd: np.ndarray, freqs: np.ndarray,
                      f_lo: float, f_hi: float) -> float:
        """
        Normalized Shannon spectral entropy ∈ [0, 1] for [f_lo, f_hi].

          p_i   = P_i / Σ P_i        (probability distribution within band)
          H     = -Σ p_i · ln(p_i)   (Shannon entropy, nats)
          H_max = ln(N_bins)          (maximum entropy for uniform distribution)
          return H / H_max
        """
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        n_bins = int(mask.sum())
        if n_bins < 2:
            return 0.0

        p = psd[mask]
        total = float(p.sum())
        if total < 1e-15:
            return 0.0

        p = p / total                       # normalize → probability distribution
        p_nz = p[p > 1e-15]                # drop zero-power bins (log undefined)
        H = -float(np.sum(p_nz * np.log(p_nz)))
        H_max = np.log(n_bins)              # maximum possible entropy

        return float(np.clip(H / H_max, 0.0, 1.0))
