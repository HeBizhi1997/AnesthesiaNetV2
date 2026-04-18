from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ProcessRequest(BaseModel):
    sample_rate: int = Field(256, ge=64, le=2048)
    channel_count: int = Field(4, ge=1, le=32)
    start_time: str = ""
    # Shape: (n_samples, n_channels)
    eeg_data: list[list[float]]
    pulse_wave: list[float] = []
    spo2: Optional[float] = None
    heart_rate: Optional[float] = None


class ProcessResponse(BaseModel):
    # EEG components
    raw_eeg: list[float] = []
    delta_wave: list[float] = []
    theta_wave: list[float] = []
    alpha_wave: list[float] = []
    beta_wave: list[float] = []
    gamma_wave: list[float] = []

    # Band power ratios (0-1)
    delta_power: float = 0.0
    theta_power: float = 0.0
    alpha_power: float = 0.0
    beta_power: float = 0.0
    gamma_power: float = 0.0

    # DSA
    dsa_matrix: list[list[float]] = []
    dsa_frequencies: list[float] = []
    dsa_times: list[float] = []

    # Depth of anesthesia
    bis: Optional[float] = None
    sqi: float = 0.0
    se: Optional[float] = None   # State Entropy   0–91
    re: Optional[float] = None   # Response Entropy 0–100

    # Vitals
    heart_rate: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    pulse_wave: list[float] = []
    spo2: Optional[float] = None
