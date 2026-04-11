"""
Signal filtering steps for EEG preprocessing.

Two usage contexts:
  1. RecordingFilter — applied to full recording (thousands of seconds).
     Uses MNE zero-phase FIR filters: optimal for EEG, no phase distortion.

  2. WindowFilter — applied to 4-second windows (512 samples).
     Uses scipy SOS (sosfiltfilt): numerically stable, handles short signals.
     MNE FIR is inappropriate here (filter length > signal length).

Pipeline order (V3.0, doc/EEG数据处理流程.md):
  1. MedianSpikeRemoval  — non-linear impulse spike removal (ESU)
  2. HighpassFilter      — baseline drift removal (0.5 Hz)
  3. WaveletDenoiser     — adaptive ESU high-frequency suppression
  4. NotchFilter(s)      — 50 Hz power line harmonics
  5. LowpassFilter       — anti-alias (47 Hz)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import medfilt, butter, sosfiltfilt, iirnotch, filtfilt
import pywt
import mne


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _butter_sos(cutoff, fs, btype, order=4):
    nyq = fs / 2.0
    wn = np.asarray(cutoff) / nyq
    return butter(order, wn, btype=btype, output="sos")


def _apply_sos(sos, data: np.ndarray) -> np.ndarray:
    """Apply SOS filter to (n_channels, n_samples) array."""
    return np.stack(
        [sosfiltfilt(sos, data[ch]) for ch in range(data.shape[0])],
        axis=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Recording-level filters  (input: full EEG array, n_ch × n_samples, float64)
# ─────────────────────────────────────────────────────────────────────────────

def recording_filter(
    data: np.ndarray,
    fs: float,
    highpass: float = 0.5,
    lowpass: float = 47.0,
    notch_freqs: list = None,
    median_kernel_ms: float = 20.0,
    wavelet: str = "db4",
    wavelet_level: int = 5,
    esu_energy_thresh: float = 3.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Full-recording preprocessing pipeline using MNE zero-phase FIR filters.
    data: (n_channels, n_samples)  float64
    Returns filtered data (n_channels, n_samples) float32
    """
    if notch_freqs is None:
        notch_freqs = [50.0]

    data = data.astype(np.float64)

    # 1. Median spike removal (non-linear, ESU impulse suppression)
    kernel_n = max(3, int(round(median_kernel_ms * fs / 1000.0)))
    if kernel_n % 2 == 0:
        kernel_n += 1
    data = np.stack([medfilt(data[ch], kernel_size=kernel_n)
                     for ch in range(data.shape[0])], axis=0)

    # 2. High-pass FIR (MNE) — baseline drift removal
    data = mne.filter.filter_data(
        data, sfreq=fs, l_freq=highpass, h_freq=None,
        method="fir", fir_design="firwin", verbose=verbose,
    )

    # 3. Wavelet adaptive denoising (ESU high-freq suppression)
    data = np.stack([_wavelet_denoise(data[ch], wavelet, wavelet_level,
                                      esu_energy_thresh)
                     for ch in range(data.shape[0])], axis=0)

    # 4. Notch filter(s) — power line harmonics
    data = mne.filter.notch_filter(
        data, Fs=fs, freqs=np.array(notch_freqs, dtype=float),
        method="fir", verbose=verbose,
    )

    # 5. Low-pass FIR (MNE) — anti-alias
    data = mne.filter.filter_data(
        data, sfreq=fs, l_freq=None, h_freq=lowpass,
        method="fir", fir_design="firwin", verbose=verbose,
    )

    return data.astype(np.float32)


def _wavelet_denoise(x: np.ndarray, wavelet: str, level: int,
                     esu_thresh: float) -> np.ndarray:
    """Wavelet soft-thresholding with ESU detection on a 1-D signal."""
    n = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=level)

    # MAD-based noise estimate from finest detail
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12

    # Detect ESU: check if high-freq sub-band energy is anomalously high
    hf_energy = sum(np.sum(c ** 2) for c in coeffs[1:3])
    total_energy = sum(np.sum(c ** 2) for c in coeffs) + 1e-12
    esu_active = (hf_energy / total_energy) > 0.5

    threshold = sigma * np.sqrt(2 * np.log(max(n, 2)))
    if esu_active:
        threshold *= esu_thresh
        coeffs[1] = np.zeros_like(coeffs[1])
        if len(coeffs) > 2:
            coeffs[2] = np.zeros_like(coeffs[2])

    coeffs[3:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[3:]]
    return pywt.waverec(coeffs, wavelet)[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Window-level filter steps  (EEGStep subclasses for per-window use)
# Used in tests / real-time inference where recording-level is not available
# ─────────────────────────────────────────────────────────────────────────────

from ..base import EEGStep
from ..context import EEGContext


class MedianSpikeRemoval(EEGStep):
    """Non-linear median filter — impulse/ESU spike suppression."""

    def __init__(self, kernel_ms: float = 20.0, fs: float = 128.0):
        n = max(3, int(round(kernel_ms * fs / 1000.0)))
        self._kernel = n if n % 2 == 1 else n + 1

    def process(self, ctx: EEGContext) -> EEGContext:
        ctx.data = np.stack(
            [medfilt(ctx.data[ch].astype(np.float64),
                     kernel_size=self._kernel).astype(np.float32)
             for ch in range(ctx.n_channels)],
            axis=0,
        )
        return ctx


class HighpassFilter(EEGStep):
    """SOS Butterworth high-pass — baseline drift removal (short windows)."""

    def __init__(self, cutoff: float = 0.5, order: int = 4, fs: float = 128.0):
        self._sos = _butter_sos(cutoff, fs, "high", order)

    def process(self, ctx: EEGContext) -> EEGContext:
        ctx.data = _apply_sos(self._sos, ctx.data.astype(np.float64)).astype(np.float32)
        return ctx


class LowpassFilter(EEGStep):
    """SOS Butterworth low-pass — anti-alias / HF rejection (short windows)."""

    def __init__(self, cutoff: float = 47.0, order: int = 4, fs: float = 128.0):
        self._sos = _butter_sos(cutoff, fs, "low", order)

    def process(self, ctx: EEGContext) -> EEGContext:
        ctx.data = _apply_sos(self._sos, ctx.data.astype(np.float64)).astype(np.float32)
        return ctx


class NotchFilter(EEGStep):
    """IIR notch filter at a specific frequency (short windows)."""

    def __init__(self, freq: float = 50.0, q: float = 30.0, fs: float = 128.0):
        w0 = freq / (fs / 2.0)
        b, a = iirnotch(w0, q)
        self._b, self._a = b, a

    def process(self, ctx: EEGContext) -> EEGContext:
        ctx.data = np.stack(
            [filtfilt(self._b, self._a,
                      ctx.data[ch].astype(np.float64)).astype(np.float32)
             for ch in range(ctx.n_channels)],
            axis=0,
        )
        return ctx


class WaveletDenoiser(EEGStep):
    """Wavelet adaptive denoising with ESU detection (short windows)."""

    def __init__(self, wavelet: str = "db4", level: int = 5,
                 esu_thresh: float = 3.0):
        self.wavelet = wavelet
        self.level = level
        self.esu_thresh = esu_thresh

    def process(self, ctx: EEGContext) -> EEGContext:
        ctx.data = np.stack(
            [_wavelet_denoise(ctx.data[ch].astype(np.float64),
                              self.wavelet, self.level,
                              self.esu_thresh).astype(np.float32)
             for ch in range(ctx.n_channels)],
            axis=0,
        )
        return ctx
