"""
Nociceptive Stress Index (NSI) — multimodal analgesia adequacy indicator.

NSI ∈ [0, 100]:
  0   = optimal analgesia, no nociceptive stress
  100 = severe nociceptive stress / analgesia failure

Two-component fusion model:

  f(ΔE)      = max(0, RE-SE-2)² × 0.35    subcortical EMG component
               Captures facial-muscle microactivity from inadequate analgesia
               even when the patient is unconscious (spinal/brainstem reflex arc).
               Quadratic amplification creates a dead-zone for measurement noise
               (ΔE<2) and rapidly escalates for genuine pain signals (ΔE>10).

  g(dBIS/dt) = max(0, ΔBIS₃₀ₛ / 30) × 90  cortical arousal component
               Captures breakthrough pain that reaches the cortex: sudden BIS
               rise (beta/gamma surge) even when RE-SE is suppressed by muscle
               relaxants.

Dynamic gating (α, β weights) based on BIS context:
  BIS 40–60  → α=0.80, β=0.40  cortex suppressed → any ΔE is genuine nociception
  BIS 60–75  → linear interpolation between zones
  BIS > 75   → α=0.20, β=0.80  emerging → ΔE may include voluntary EMG artefact

NSI = clip(α·f(ΔE) + β·g(dBIS/dt), 0, 100)
"""

import collections
import numpy as np
from loguru import logger


class NSIProcessor:

    DEADBAND  = 2.0    # ΔE below this is measurement noise, not pain signal
    F_SCALE   = 0.35   # Quadratic EMG scale (ΔE=18 → 89.6 → capped at 80)
    F_MAX     = 80.0
    G_SCALE   = 90.0   # dBIS/dt: BIS rise 1pt/s → 90 contribution
    G_MAX     = 80.0
    EMA_ALPHA = 0.30   # Temporal smoothing (~3-epoch time constant)
    BIS_HIST  = 30     # 30 epochs × 1s/epoch = 30-second BIS window

    def __init__(self):
        self._bis_buf: collections.deque = collections.deque(maxlen=self.BIS_HIST)
        self._nsi_ema: float | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, bis: float | None, se: float | None, re: float | None) -> dict:
        """
        Compute NSI from current BIS, SE, RE (all scalars, one per epoch).
        Returns {"nsi": float|None}.
        """
        if bis is None or se is None or re is None or np.isnan(bis):
            return {"nsi": self._nsi_ema}

        # 1. Subcortical EMG component
        effective_de = max(0.0, re - se - self.DEADBAND)
        f_emg = min(effective_de ** 2 * self.F_SCALE, self.F_MAX)

        # 2. Cortical arousal component (30-second BIS derivative)
        self._bis_buf.append(float(bis))
        if len(self._bis_buf) >= self.BIS_HIST:
            delta_bis_30s = float(bis) - self._bis_buf[0]
            g_arousal = min(max(0.0, delta_bis_30s / 30.0) * self.G_SCALE, self.G_MAX)
        else:
            g_arousal = 0.0

        # 3. Dynamic weight gating
        if bis <= 60.0:
            alpha_w, beta_w = 0.80, 0.40
        elif bis <= 75.0:
            t = (bis - 60.0) / 15.0
            alpha_w = 0.80 - 0.60 * t
            beta_w  = 0.40 + 0.40 * t
        else:
            alpha_w, beta_w = 0.20, 0.80

        nsi_raw = alpha_w * f_emg + beta_w * g_arousal
        nsi = float(np.clip(nsi_raw, 0.0, 100.0))

        if self._nsi_ema is None:
            self._nsi_ema = nsi
        else:
            self._nsi_ema = self.EMA_ALPHA * nsi + (1.0 - self.EMA_ALPHA) * self._nsi_ema

        logger.debug(
            f"NSI={self._nsi_ema:.1f}  f={f_emg:.1f}  g={g_arousal:.1f}  "
            f"ΔE={re - se:.1f}  α={alpha_w:.2f}  β={beta_w:.2f}"
        )
        return {"nsi": self._nsi_ema}

    def reset(self):
        self._bis_buf.clear()
        self._nsi_ema = None
