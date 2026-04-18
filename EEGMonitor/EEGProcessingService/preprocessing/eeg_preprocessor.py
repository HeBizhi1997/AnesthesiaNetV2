import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
from loguru import logger


class EEGPreprocessor:
    """
    EEG preprocessing for real-time display.

    Band definitions (per doc/eeg成分.md):
      δ  0.5 – 4   Hz   deep sleep / deep anesthesia
      θ  4   – 7   Hz   light sedation
      α  8   – 13  Hz   awake/relaxed
      β  13  – 30  Hz   active cognition
      γ  30  – 47  Hz   high-frequency; capped at 47 Hz to avoid EMG contamination

    Pipeline:
      1. Broadband bandpass  0.5–47 Hz  (SOS, zero-phase)
      2. 50 Hz notch
      3. Band wave extraction via per-band SOS bandpass
      4. Band power ratios   (Welch PSD, trapz integration, relative to 5-band sum)
      5. DSA spectrogram
      6. SQI heuristic
    """

    # Band boundaries aligned with doc/eeg成分.md and training feature extractor
    BANDS = {
        "delta": (0.5,  4.0),
        "theta": (4.0,  7.0),   # doc: 4-7 Hz
        "alpha": (8.0,  13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 47.0),  # cap at 47 Hz – above is EMG territory
    }

    def __init__(self, sample_rate: int = 256):
        self.fs = sample_rate

    # ── Public entry point ────────────────────────────────────────────────────

    def preprocess(self, eeg_data: np.ndarray) -> dict:
        """
        eeg_data: (n_samples, n_channels) – uses channel 0 for all analysis.
        Returns a dict ready to be JSON-serialised for the C# frontend.
        """
        n_samples, n_channels = eeg_data.shape
        if n_samples < 64:
            return self._empty_result()

        raw_ch = eeg_data[:, 0].copy()

        # 1. Broadband filter: 0.5–47 Hz bandpass + 50 Hz notch
        filtered = self._bandpass(raw_ch, 0.5, 47.0)
        filtered = self._notch(filtered, 50.0)

        # 2. Extract per-band waveforms by narrow bandpass on the broadband signal
        waves = {band: self._bandpass(filtered, lo, hi)
                 for band, (lo, hi) in self.BANDS.items()}

        # 3. Band power ratios (Welch PSD, relative to 5-band total)
        powers = self._band_powers(filtered)

        # 4. DSA spectrogram
        freqs, times, dsa_db = self._dsa(filtered)

        # 5. Signal quality index
        sqi = self._sqi(raw_ch, filtered)

        return {
            "raw_eeg":         raw_ch.tolist(),
            "delta_wave":      waves["delta"].tolist(),
            "theta_wave":      waves["theta"].tolist(),
            "alpha_wave":      waves["alpha"].tolist(),
            "beta_wave":       waves["beta"].tolist(),
            "gamma_wave":      waves["gamma"].tolist(),
            "delta_power":     powers["delta"],
            "theta_power":     powers["theta"],
            "alpha_power":     powers["alpha"],
            "beta_power":      powers["beta"],
            "gamma_power":     powers["gamma"],
            "dsa_matrix":      dsa_db.tolist(),
            "dsa_frequencies": freqs.tolist(),
            "dsa_times":       times.tolist(),
            "sqi":             sqi,
        }

    # ── Filters ──────────────────────────────────────────────────────────────

    def _bandpass(self, data: np.ndarray, lo: float, hi: float,
                  order: int = 4) -> np.ndarray:
        """
        Zero-phase Butterworth bandpass using SOS form.
        SOS is numerically stable even for narrow low-frequency bands (delta 0.5-4 Hz).
        Clamps hi to 0.99×Nyquist to prevent boundary failures at any sample rate.
        """
        nyq = self.fs / 2.0
        lo_c = max(lo, 0.01)
        hi_c = min(hi, nyq * 0.99)
        if lo_c >= hi_c:
            return np.zeros_like(data)
        sos = butter(order, [lo_c / nyq, hi_c / nyq], btype="bandpass", output="sos")
        return sosfiltfilt(sos, data)

    def _notch(self, data: np.ndarray, freq: float, q: float = 30.0) -> np.ndarray:
        nyq = self.fs / 2.0
        if freq >= nyq:
            return data
        b, a = iirnotch(freq / nyq, q)
        return filtfilt(b, a, data)

    # ── Spectral analysis ─────────────────────────────────────────────────────

    def _band_powers(self, eeg: np.ndarray) -> dict:
        """
        Relative band powers summing to 1.0 across the 5 defined bands.

        Uses Welch PSD for robust spectral estimation:
          - nperseg = min(fs, len) gives 1 Hz frequency resolution
          - noverlap = 50% reduces variance
        Integration via trapz over each band's frequency bins.
        """
        nperseg = min(self.fs, len(eeg))
        freqs, psd = welch(eeg, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2)

        raw_powers = {}
        for band, (lo, hi) in self.BANDS.items():
            idx = (freqs >= lo) & (freqs < hi)
            raw_powers[band] = float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0

        # Relative normalisation: each value = fraction of 5-band total
        total = sum(raw_powers.values()) or 1.0
        return {k: v / total for k, v in raw_powers.items()}

    def _dsa(self, eeg: np.ndarray,
             nperseg: int = 128,
             noverlap: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Density Spectral Array — short-time power spectrogram.
        noverlap=96 (75%) gives smooth time resolution at 1s updates.
        Frequency axis limited to 0–40 Hz (clinical BIS/DOA range).
        """
        f, t, Sxx = signal.spectrogram(
            eeg, fs=self.fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )
        mask = f <= 40.0
        Sxx_db = 10.0 * np.log10(Sxx[mask] + 1e-12)
        return f[mask], t, Sxx_db

    # ── Signal Quality Index ──────────────────────────────────────────────────

    def _sqi(self, raw: np.ndarray, filtered: np.ndarray) -> float:
        """
        Three-component SQI [0, 100]:
          1. Flatline guard  — std < 0.1 µV → electrode off
          2. Amplitude guard — saturation > 1000 µV or weak < 1 µV
          3. HF artefact     — power >47 Hz vs total (EMG/ESU noise)

        Note: high-amplitude delta during deep anesthesia (200-500 µV) is
        normal – do NOT penalise it. Only true saturation (>1000 µV) is bad.
        """
        std = float(np.std(filtered))
        if std < 0.1:
            return 0.0

        max_amp = float(np.max(np.abs(filtered)))
        if max_amp > 1000.0:
            return 5.0  # saturated electrode

        # Weak signal penalty only (std < 1 µV means electrode barely touching)
        amp_score = float(np.clip(std / 1.0, 0.0, 1.0)) if std < 1.0 else 1.0

        # HF noise: power > 47 Hz in raw signal vs total — EMG/ESU indicator
        nyq = self.fs / 2.0
        if nyq > 50:
            f, pxx = welch(raw, fs=self.fs, nperseg=min(self.fs, len(raw)))
            hf_ratio = float(pxx[f > 47.0].sum() / (pxx.sum() + 1e-12))
            # Gentle penalty: hf_ratio=0.3 → score=0.55; hf_ratio=0.5 → score=0.25
            hf_score = float(np.clip(1.0 - hf_ratio * 1.5, 0.25, 1.0))
        else:
            hf_score = 1.0

        return float(np.clip(100.0 * amp_score * hf_score, 0.0, 100.0))

    @staticmethod
    def _empty_result() -> dict:
        return {
            "raw_eeg": [], "delta_wave": [], "theta_wave": [],
            "alpha_wave": [], "beta_wave": [], "gamma_wave": [],
            "delta_power": 0.0, "theta_power": 0.0, "alpha_power": 0.0,
            "beta_power": 0.0, "gamma_power": 0.0,
            "dsa_matrix": [], "dsa_frequencies": [], "dsa_times": [],
            "sqi": 0.0,
        }
