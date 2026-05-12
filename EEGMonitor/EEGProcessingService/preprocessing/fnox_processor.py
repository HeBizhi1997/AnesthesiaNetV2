"""
fNox — Fused Nociception Index  [0, 99]

Reverse-engineered to match qNOX (Conox/Fresenius Kabi) clinical behavior.

─── Core design rationale ───────────────────────────────────────────────────────

qNOX and qCON (≈BIS) measure two parallel physiological dimensions:
  qCON : cortical hypnosis / consciousness suppression  (low-frequency EEG)
  qNOX : nociceptive reactivity / analgesia adequacy   (high-frequency EEG + EMG)

Primary driver: RE (Response Entropy, 0.8–47 Hz)

RE already encodes both dimensions in a single scalar:
  • Low-frequency part  (≈ SE)  reflects propofol/hypnotic depth
  • High-frequency part (32-47 Hz) reflects opioid-suppressible EMG activity

When opioids suppress nociceptive EMG, RE drops toward SE → lower fNox.
When opioid effect fades or nociceptive stimulus breaks through, RE rises → higher fNox.

This naturally captures the clinical time-series asymmetry:
  Induction  : qCON/BIS drops FIRST (propofol fast); RE stays high until opioid
               takes effect → fNox remains elevated longer than BIS  ✓
  Emergence  : RE/fNox rises BEFORE BIS (opioid wears off first, propofol lasts
               longer) → fNox predicts arousal 45-90 s ahead of BIS  ✓

─── Primary formula ─────────────────────────────────────────────────────────────

  fNox_base = clip(RE × 0.90 − 5.0, 0, 99)

  Calibration (using post-propofol-correction RE values):
    Adequate analgesia  RE ≈ 49  →  fNox ≈ 39  (target 30-50) ✓
    Awake               RE ≈ 80  →  fNox ≈ 67  ✓
    Deep anesthesia     RE ≈ 35  →  fNox ≈ 27  ✓
    Nociceptive event   RE ≈ 70  →  fNox ≈ 58  (alert >50)   ✓

  TIVA regression cross-check (clinical literature):
    qCON = 0.79 × qNOX + 5.8  →  at fNox=40: BIS ≈ 37.4
    qNOX = 0.78 × qCON + 26.1 →  population mean fit (not the target-zone condition)

─── BIS trend supplement (MR compensation) ─────────────────────────────────────

  When neuromuscular blockade (NMB) prevents EMG, RE cannot rise even when pain
  stimulus is present.  Rising BIS (cortical breakthrough) is then the only signal.

  Normal mode: trend_add = clip(max(0, ΔBIS_30s) × 0.5, 0, 15)
  MR mode    : trend_add = clip(max(0, ΔBIS_30s) × 2.0, 0, 30)

─── Additional guards ───────────────────────────────────────────────────────────

  ESU artefact : RE jump > 15 in one epoch → freeze RE at last valid value
  MR detection : rolling 2-min median |RE-SE| < 1.5 → activate MR mode
  Asymmetric EMA: α=0.50 on rise (fast alarm), α=0.20 on fall (slow decay)
"""

import collections
import numpy as np
from loguru import logger


class FNoxProcessor:

    # Primary scaling:  fNox_base = RE * RE_SCALE + RE_OFFSET
    RE_SCALE  = 0.90
    RE_OFFSET = -5.0

    # BIS trend supplement
    TREND_SCALE_NORMAL = 0.5
    TREND_SCALE_MR     = 2.0
    TREND_CAP_NORMAL   = 15.0
    TREND_CAP_MR       = 30.0

    # Asymmetric EMA: quick alarm rise, slow recovery
    ALPHA_UP   = 0.50
    ALPHA_DOWN = 0.20

    # Muscle-relaxant detection
    MR_WINDOW    = 120   # rolling epochs (≈120 s at 1 Hz pipeline)
    MR_MIN_FILL  = 60    # minimum fill before enabling MR detection
    MR_THRESHOLD = 1.5   # median |RE-SE| below this → NMB suspected

    # Electrocautery guard
    ESU_SPIKE = 15.0   # RE jump > 15 in one epoch → artefact

    def __init__(self):
        self._bis_buf: collections.deque = collections.deque(maxlen=30)
        self._de_buf:  collections.deque = collections.deque(maxlen=self.MR_WINDOW)
        self._prev_re: float | None = None
        self._fnox_ema: float | None = None
        self._mr_mode: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, bis: float | None, se: float | None, re: float | None) -> dict:
        """
        One call per pipeline epoch (1 s).
        Returns {"fnox": float|None, "mr_mode": bool}.
        """
        if bis is None or se is None or re is None or np.isnan(bis):
            return {"fnox": self._fnox_ema, "mr_mode": self._mr_mode}

        se_f  = float(se)
        bis_f = float(bis)

        # ── ESU artefact guard ────────────────────────────────────────────────
        re_eff = float(re)
        if self._prev_re is not None and abs(re_eff - self._prev_re) > self.ESU_SPIKE:
            logger.debug(f"fNox: ESU spike RE {self._prev_re:.1f}→{re_eff:.1f}, frozen")
            re_eff = self._prev_re
        self._prev_re = re_eff

        delta_e = re_eff - se_f

        # ── Muscle-relaxant detection (rolling median of |ΔE|) ────────────────
        self._de_buf.append(abs(delta_e))
        if len(self._de_buf) >= self.MR_MIN_FILL:
            self._mr_mode = float(np.median(list(self._de_buf))) < self.MR_THRESHOLD

        # ── Primary: RE-based analgesia index ────────────────────────────────
        fNox_base = float(np.clip(re_eff * self.RE_SCALE + self.RE_OFFSET, 0.0, 99.0))

        # ── BIS trend supplement (crucial in MR mode) ─────────────────────────
        self._bis_buf.append(bis_f)
        if len(self._bis_buf) >= self._bis_buf.maxlen:
            bis_rise = max(0.0, bis_f - self._bis_buf[0])
            if self._mr_mode:
                trend_add = float(np.clip(bis_rise * self.TREND_SCALE_MR, 0.0, self.TREND_CAP_MR))
            else:
                trend_add = float(np.clip(bis_rise * self.TREND_SCALE_NORMAL, 0.0, self.TREND_CAP_NORMAL))
        else:
            trend_add = 0.0

        fnox_raw = float(np.clip(fNox_base + trend_add, 0.0, 99.0))

        # ── Asymmetric EMA ────────────────────────────────────────────────────
        if self._fnox_ema is None:
            self._fnox_ema = fnox_raw
        else:
            alpha = self.ALPHA_UP if fnox_raw > self._fnox_ema else self.ALPHA_DOWN
            self._fnox_ema = alpha * fnox_raw + (1.0 - alpha) * self._fnox_ema

        self._fnox_ema = float(np.clip(self._fnox_ema, 0.0, 99.0))

        logger.debug(
            f"fNox={self._fnox_ema:.1f}  base={fNox_base:.1f}  trend={trend_add:.1f}  "
            f"RE={re_eff:.1f}  SE={se_f:.1f}  ΔE={delta_e:.1f}  MR={self._mr_mode}"
        )
        return {"fnox": self._fnox_ema, "mr_mode": self._mr_mode}

    def reset(self):
        self._bis_buf.clear()
        self._de_buf.clear()
        self._prev_re = None
        self._fnox_ema = None
        self._mr_mode = False
