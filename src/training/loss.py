"""
Composite loss for anesthesia depth regression.

L = L_base + λ₁·L_Mono

where L_base is a per-sample SQI-weighted combination:
    L_base = sqi · LogCosh(pred, target) + (1 − sqi) · (pred − 0.5)²

Why LogCosh instead of MSE:
  MSE gradient scales as error². In the normalised [0,1] space a 1-BIS error
  contributes only 0.0001 MSE — tiny gradients near convergence.
  LogCosh behaves like L2 for small errors and L1 for large errors, giving
  robust, well-scaled gradients even for EEG windows contaminated by
  electrosurgical (ESU) artefacts that raise outlier BIS predictions.

Why SQI-weighted L_base (replaces separate L_Physio):
  When SQI ≈ 1 (clean signal), the model is fully supervised by the true label.
  When SQI → 0 (artefact / flat-line), the target effectively becomes 0.5
  (neutral prediction), preventing the model from "learning" artefact-induced
  extreme values.  This is a continuous relaxation of the hard SQI < 0.3 mask.

L_Mono — induction monotonicity (P2 fix):
  Uses pred_seq (B, T, 1) from AnesthesiaNet, which now returns predictions at
  every timestep via return_sequences=True in LNNCore.
  diff = torch.diff(pred_seq, dim=1)  # (B, T-1, 1)
  Penalises BIS increases (positive diff) when the final label is in the
  induction zone (BIS > 0.6).  This matches the clinical expectation that
  BIS should fall monotonically from ~1.0 to ~0.4 during induction.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


def _log_cosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)."""
    return torch.abs(x) + torch.log1p(torch.exp(-2.0 * torch.abs(x))) - 0.6931471805599453


class AnesthesiaLoss(nn.Module):
    def __init__(
        self,
        lambda_monotonic: float = 0.3,
        lambda_physio: float = 0.0,          # kept for API compat; now absorbed into base
        induction_threshold: float = 0.6,    # BIS > 0.6 → induction zone
    ):
        super().__init__()
        self.lam_mono = lambda_monotonic
        self.induction_thr = induction_threshold
        # lambda_physio is retired: SQI weighting is now baked into L_base

    def forward(
        self,
        pred: torch.Tensor,                     # (B, 1)        in [0, 1]
        target: torch.Tensor,                   # (B,)          in [0, 1]
        sqi: torch.Tensor,                      # (B,)          mean SQI in [0, 1]
        pred_seq: Optional[torch.Tensor] = None, # (B, T, 1)    all-step preds
        label_seq: Optional[torch.Tensor] = None,# (B, T)       all-step targets
    ) -> dict:
        pred_sq = pred.squeeze(-1)    # (B,)
        target  = target.float()
        sqi     = sqi.float().clamp(0.0, 1.0)

        # ── SQI-weighted base loss ───────────────────────────────────────────
        # Clean signal:  optimise towards true label (LogCosh)
        # Noisy signal:  pull prediction towards 0.5 (neutral)
        log_cosh = _log_cosh(pred_sq - target)            # (B,)
        neutral  = (pred_sq - 0.5) ** 2                   # (B,)
        loss_base = (sqi * log_cosh + (1.0 - sqi) * neutral).mean()

        # ── Monotonic penalty (P2 fix — uses full sequence) ─────────────────
        # Penalise rising BIS during induction for sequences where the final
        # label is still in the induction zone.
        loss_mono = torch.tensor(0.0, device=pred.device)
        if (self.lam_mono > 0
                and pred_seq is not None
                and pred_seq.shape[1] > 1):

            # Which batch items end in induction zone?
            induction_mask = target >= self.induction_thr   # (B,)

            if induction_mask.any():
                ps = pred_seq.squeeze(-1)                    # (B, T)
                diff = torch.diff(ps, dim=1)                 # (B, T-1)

                # Penalise positive diffs (BIS increasing) in induction rows
                increases = torch.clamp(diff[induction_mask], min=0.0)
                if increases.numel() > 0:
                    loss_mono = (increases ** 2).mean()

        total = loss_base + self.lam_mono * loss_mono

        return {
            "loss":      total,
            "base":      loss_base.item(),
            "monotonic": loss_mono.item() if isinstance(loss_mono, torch.Tensor)
                         else float(loss_mono),
        }
