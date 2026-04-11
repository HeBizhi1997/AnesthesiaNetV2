"""
EEGStep — abstract base class every processing step must implement.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .context import EEGContext


class EEGStep(ABC):
    """All pipeline steps inherit from this class."""

    @abstractmethod
    def process(self, ctx: EEGContext) -> EEGContext:
        """Apply this step's transformation to ctx.data in-place or return new ctx."""

    def validate(self, ctx: EEGContext) -> None:
        """Basic sanity checks — override to add step-specific validation."""
        if np.isnan(ctx.data).any():
            raise ValueError(f"[{self.__class__.__name__}] produced NaN values.")
        if np.isinf(ctx.data).any():
            raise ValueError(f"[{self.__class__.__name__}] produced Inf values.")
        rms = float(np.sqrt(np.mean(ctx.data ** 2)))
        if rms > 5000.0:
            raise ValueError(
                f"[{self.__class__.__name__}] signal RMS={rms:.1f} is abnormally high."
            )

    def __call__(self, ctx: EEGContext) -> EEGContext:
        return self.process(ctx)
