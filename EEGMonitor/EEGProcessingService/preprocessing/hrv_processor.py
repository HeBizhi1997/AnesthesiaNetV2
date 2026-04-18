import numpy as np
from scipy.signal import find_peaks


class HRVProcessor:
    """Computes HR and HRV from pulse oximetry (PPG) waveform."""

    def __init__(self, sample_rate: int = 256):
        self.fs = sample_rate

    def compute(self, pulse_wave: np.ndarray) -> dict:
        if len(pulse_wave) < self.fs:
            return {"hrv_rmssd": None, "hr": None, "pulse_wave": pulse_wave.tolist()}

        # Normalise
        pw = pulse_wave - np.mean(pulse_wave)
        std = np.std(pw)
        if std > 0:
            pw /= std

        # Find peaks (R-peaks / pulse peaks)
        min_distance = int(self.fs * 0.4)  # ≥ 0.4 s between beats → ≤150 bpm
        peaks, _ = find_peaks(pw, height=0.3, distance=min_distance)

        if len(peaks) < 2:
            return {"hrv_rmssd": None, "hr": None, "pulse_wave": pulse_wave.tolist()}

        rr_samples = np.diff(peaks)
        rr_ms = rr_samples / self.fs * 1000.0

        hr = float(60_000.0 / np.mean(rr_ms)) if len(rr_ms) > 0 else None
        rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else None

        return {
            "hrv_rmssd": rmssd,
            "hr": hr,
            "pulse_wave": pulse_wave.tolist(),
            "rr_intervals_ms": rr_ms.tolist(),
        }
