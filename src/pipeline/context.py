"""
EEGContext — the state carrier passed through every pipeline step.
Holds raw/processed signal, metadata, processing history, and
intermediate artifacts (for frontend export or debugging).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class EEGContext:
    """
    Shape convention: data is always (n_channels, n_samples).
    For the BIS monitor: n_channels=2 (Fp1=EEG1, Fp2=EEG2).
    """
    data: np.ndarray            # (n_channels, n_samples)  float32
    fs: float                   # sampling rate in Hz
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Signal quality index per channel [0, 1], updated by SQI step
    sqi: Optional[np.ndarray] = None   # (n_channels,)

    # Extracted feature vector, set by FeatureExtractor step
    features: Optional[np.ndarray] = None   # (n_features,)

    # Clinical label attached at dataset build time (not pipeline time)
    label: Optional[float] = None   # BIS value 0-100

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs

    def clone_data(self) -> np.ndarray:
        """Return a copy of current data (for artifact snapshots)."""
        return self.data.copy()

    def log(self, step_name: str, **kwargs) -> None:
        entry = {"step": step_name}
        entry.update(kwargs)
        entry["rms"] = float(np.sqrt(np.mean(self.data ** 2)))
        self.history.append(entry)
