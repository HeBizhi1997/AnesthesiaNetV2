"""
Advanced EEG features with mathematical depth beyond band power ratios.

Feature categories:
  Spectral morphology  : SEF50, spectral centroid, 1/f slope (already in features.py)
  Complexity/entropy   : PE, LZC (already), Multiscale Entropy, Hjorth
  Bispectral/coupling  : Bicoherence, PAC (α-δ, α-γ)
  Fractal/nonlinear    : DFA α exponent
  Burst suppression    : BSR multi-threshold (already)
  Sleep-specific       : Sigma power, slow oscillation, theta/beta, spindle correlate
  Anesthesia-specific  : Alpha-delta coupling, beta-delta antagonism
  Time-domain          : Hjorth Activity, Mobility, Complexity

All functions accept (x: 1-D array, fs: float) and return a scalar or small vector.
"""
import numpy as np
from scipy.signal import welch, hilbert, butter, sosfiltfilt, detrend
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════════
# Hjorth Parameters (Hjorth, 1970) — 3 time-domain descriptors
#   Activity  = variance(x)                — signal power
#   Mobility  = sqrt(var(dx/dt)/var(x))    — mean frequency (proportional)
#   Complexity = Mobility(dx/dt)/Mobility(x) — bandwidth (proportional)
#
# Computational cost: O(n) — extremely cheap (derivative + variance)
# Clinical evidence:
#   - Mobility decreases with deepening anesthesia (Ferenets et al., 2006)
#   - Complexity distinguishes burst suppression from continuous EEG
#   - Used in commercial anesthesia monitors (CSM, Danmeter)
# ═══════════════════════════════════════════════════════════════════════════════

def hjorth_activity(x: np.ndarray) -> float:
    return float(np.var(x) + 1e-12)


def hjorth_mobility(x: np.ndarray) -> float:
    dx = np.diff(x)
    var_x = np.var(x) + 1e-12
    var_dx = np.var(dx) + 1e-12
    return float(np.sqrt(var_dx / var_x))


def hjorth_complexity(x: np.ndarray) -> float:
    dx = np.diff(x)
    return float(hjorth_mobility(dx) / max(hjorth_mobility(x), 1e-12))


# ═══════════════════════════════════════════════════════════════════════════════
# Detrended Fluctuation Analysis (Peng et al., 1995)
#   Measures long-range temporal correlations (self-similarity).
#   α = 0.5  → white noise (no correlation)
#   α ≈ 1.0  → 1/f noise (scale-invariant, healthy brain)
#   α < 0.5  → anti-correlated
#   α > 1.0  → non-stationary, strongly correlated
#
# Clinical evidence:
#   - DFA α decreases during propofol anesthesia (Jospin et al., 2007)
#   - DFA α differs between NREM and REM sleep (Lee et al., 2002)
#   - α near 1.0 in awake resting state; drops toward 0.5 under anesthesia
#
# Computational cost: O(n log n) — FFT-based detrending, then linear regression
# ═══════════════════════════════════════════════════════════════════════════════

def dfa_alpha(x: np.ndarray, fs: float,
              scales: np.ndarray | None = None) -> float:
    """
    Compute DFA α exponent over logarithmically-spaced window scales.

    Returns α ∈ [0, 3] where higher = stronger long-range correlation.
    """
    n = len(x)
    if n < 64:
        return 0.5  # not enough data, return white-noise default

    y = np.cumsum(x - x.mean())

    if scales is None:
        # Scales from 4 samples to n/4, log-spaced, ~10 points
        scales = np.unique(np.logspace(
            np.log10(4), np.log10(max(4, n // 4)), 10
        ).astype(int))

    flucts = []
    for s in scales:
        if s < 4 or s > n // 2:
            continue
        n_segments = n // s
        rms_list = []
        for v in range(n_segments):
            seg = y[v * s:(v + 1) * s]
            t_seg = np.arange(s, dtype=np.float64)
            if len(seg) < 2:
                continue
            # Linear detrend
            coeffs = np.polyfit(t_seg, seg, 1)
            trend = np.polyval(coeffs, t_seg)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_list:
            flucts.append((float(s), float(np.mean(rms_list))))

    if len(flucts) < 3:
        return 0.5

    scales_arr = np.array([f[0] for f in flucts])
    flucts_arr = np.array([f[1] for f in flucts])

    # log-log linear fit: log(F) = α * log(s) + C
    log_s = np.log10(scales_arr)
    log_f = np.log10(flucts_arr + 1e-12)
    alpha = float(np.polyfit(log_s, log_f, 1)[0])

    return float(np.clip(alpha, 0.0, 3.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Multiscale Sample Entropy (Costa et al., 2002)
#   Quantifies signal complexity across multiple time scales.
#   Healthy systems show complexity at multiple scales.
#   Disease/anesthesia → loss of complexity (reduced entropy at higher scales).
#
#   MSE_k = SampEn( coarse-grained(x, scale=k) )
#
# Clinical evidence:
#   - MSE decreases under propofol anesthesia (Liang et al., 2015)
#   - MSE distinguishes consciousness levels (Schartner et al., 2015)
#   - Higher scales (3-5) most sensitive to anesthesia depth
#
# Computational cost: O(n²/m) per scale — expensive at large scales
#   Optimized: use scales 2, 3, 5 only; approximate with template matching
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample Entropy with relative tolerance r (fraction of std)."""
    n = len(x)
    if n < m + 2:
        return 0.0
    r_abs = r * np.std(x)
    if r_abs < 1e-12:
        return 0.0

    def _count_matches(template_len):
        count = 0
        templates = np.array([x[i:i + template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            dists = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(dists < r_abs)
        return max(count, 1)

    A = _count_matches(m + 1)
    B = _count_matches(m)
    return float(-np.log(A / B)) if A > 0 and B > 0 else 0.0


def multiscale_entropy(x: np.ndarray, fs: float,
                       scales: tuple = (2, 3, 5)) -> list[float]:
    """Compute Sample Entropy at multiple coarse-graining scales."""
    results = []
    for scale in scales:
        if len(x) < scale * 4:
            results.append(0.0)
            continue
        # Coarse-grain: average non-overlapping windows of length 'scale'
        n_coarse = len(x) // scale
        coarse = np.array([x[i * scale:(i + 1) * scale].mean()
                          for i in range(n_coarse)])
        results.append(_sample_entropy(coarse, m=2, r=0.2))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Alpha-Delta Phase-Amplitude Coupling (Tort et al., 2010)
#   PAC = |mean(A_alpha * exp(i * phi_delta))| / mean(A_alpha)
#   where A_alpha = Hilbert envelope of alpha-band signal
#         phi_delta = Hilbert phase of delta-band signal
#
# Clinical evidence:
#   - α-δ PAC increases dramatically under propofol (Purdon et al., 2013)
#   - This is the electrophysiological signature of GABA_A agonism
#   - ABSENT in natural sleep (spindle PAC is α-σ, not α-δ)
#   - Strongest single-feature discriminator of anesthesia vs sleep
#
# Also compute: α-γ PAC (gamma amplitude modulated by alpha phase)
#   - Present in awake cognition; disrupted by anesthesia
# ═══════════════════════════════════════════════════════════════════════════════

def pac_modulation_index(x: np.ndarray, fs: float,
                         phase_band: tuple = (0.5, 4.0),    # delta phase
                         amp_band: tuple = (8.0, 13.0),     # alpha amplitude
                         ) -> float:
    """
    Phase-Amplitude Coupling Modulation Index (Tort et al.).
    Returns value in [0, 1]; higher = stronger coupling.
    """
    n = len(x)
    if n < int(fs * 0.5):
        return 0.0
    try:
        nyq = fs / 2.0
        # Phase band filter
        sos_phase = butter(4, [phase_band[0] / nyq, phase_band[1] / nyq],
                          btype="bandpass", output="sos")
        phase_signal = sosfiltfilt(sos_phase, x)
        phi = np.angle(hilbert(phase_signal))

        # Amplitude band filter
        sos_amp = butter(4, [amp_band[0] / nyq, amp_band[1] / nyq],
                        btype="bandpass", output="sos")
        amp_signal = sosfiltfilt(sos_amp, x)
        env = np.abs(hilbert(amp_signal))

        # Modulation Index
        z = np.mean(env * np.exp(1j * phi))
        return float(np.abs(z) / (np.mean(env) + 1e-12))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Spectral Edge Frequencies (multiple percentiles)
#   SEF50, SEF90, SEF95 = frequencies below which 50%, 90%, 95% of power lies.
#   Normalized by 47 Hz (EEG band limit).
#
#   SEF50: robust estimate of median frequency (less sensitive to noise than SEF95)
#   SEF90: sensitive to beta/gamma shift
#   SEF95: standard spectral edge (already in features.py)
# ═══════════════════════════════════════════════════════════════════════════════

def spectral_edge(pxx: np.ndarray, freqs: np.ndarray,
                  percentile: float = 50.0, norm_hz: float = 47.0) -> float:
    cumsum = np.cumsum(pxx)
    total = cumsum[-1] + 1e-12
    idx = np.searchsorted(cumsum, percentile / 100.0 * total)
    idx = min(idx, len(freqs) - 1)
    return float(np.clip(freqs[idx] / norm_hz, 0.0, 1.0))


def spectral_centroid(pxx: np.ndarray, freqs: np.ndarray,
                      norm_hz: float = 47.0) -> float:
    """Center of mass of the spectrum, normalized to [0,1]."""
    total = pxx.sum() + 1e-12
    centroid = float((freqs * pxx).sum() / total)
    return float(np.clip(centroid / norm_hz, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Sleep-specific features
# ═══════════════════════════════════════════════════════════════════════════════

def sigma_power_ratio(pxx: np.ndarray, freqs: np.ndarray,
                      sigma_lo: float = 12.0, sigma_hi: float = 15.0) -> float:
    """Relative power in sigma (spindle) band: 12-15 Hz."""
    mask = (freqs >= sigma_lo) & (freqs < sigma_hi)
    total = pxx.sum() + 1e-12
    return float(pxx[mask].sum() / total)


def slow_delta_ratio(pxx: np.ndarray, freqs: np.ndarray,
                     s_lo: float = 0.5, s_mid: float = 2.0,
                     s_hi: float = 4.0) -> float:
    """Slow delta (0.5-2Hz) / fast delta (2-4Hz) ratio. >1.5 suggests deep sleep."""
    slow = pxx[(freqs >= s_lo) & (freqs < s_mid)].sum() + 1e-12
    fast = pxx[(freqs >= s_mid) & (freqs < s_hi)].sum() + 1e-12
    return float(np.clip(slow / fast, 0.1, 10.0))


def slow_oscillation_power(pxx: np.ndarray, freqs: np.ndarray,
                           so_lo: float = 0.3, so_hi: float = 1.0) -> float:
    """Power in slow oscillation band (<1 Hz), characteristic of N3 deep sleep."""
    mask = (freqs >= so_lo) & (freqs < so_hi)
    total = pxx.sum() + 1e-12
    return float(pxx[mask].sum() / total)


def theta_beta_ratio(pxx: np.ndarray, freqs: np.ndarray,
                     t_lo: float = 4.0, t_hi: float = 8.0,
                     b_lo: float = 13.0, b_hi: float = 30.0) -> float:
    """Theta/Beta ratio — drowsiness marker. High = drowsy/sleep onset."""
    theta_p = pxx[(freqs >= t_lo) & (freqs < t_hi)].sum() + 1e-12
    beta_p = pxx[(freqs >= b_lo) & (freqs < b_hi)].sum() + 1e-12
    return float(np.clip(theta_p / beta_p, 0.1, 20.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Bicoherence — normalized bispectrum (mathematical basis of BIS)
#   B(f1, f2) = E[X(f1) · X(f2) · X*(f1+f2)]
#   Bicoherence = |B(f1,f2)|² / (E[|X(f1)X(f2)|²] · E[|X(f1+f2)|²])
#
#   Captures quadratic phase coupling between frequency components.
#   This is the core mathematics behind the Bispectral Index (BIS).
#
# Clinical evidence:
#   - Bicoherence in δ-α region is the PRIMARY marker of propofol effect
#   - Bicoherence peak shifts from β-γ (awake) to δ-α (anesthesia)
#   - This is the single feature that motivated the BIS monitor's design
#
# Computational cost: O(n³) for full bispectrum — too expensive per window.
#   Optimized: compute bicoherence only in δ-α interaction region (0.5-4Hz × 8-13Hz)
#   using Welch-averaged bispectrum with 2-second segments.
# ═══════════════════════════════════════════════════════════════════════════════

def bicoherence_peak(x: np.ndarray, fs: float,
                     f_lo: float = 0.5, f_hi: float = 13.0,
                     nperseg: int = 128) -> float:
    """
    Maximum bicoherence in the δ-α interaction region (0.5-13 Hz).
    Uses direct FFT-based bispectrum estimation.

    Returns peak bicoherence ∈ [0, 1]; higher = stronger non-linear coupling.
    """
    n = len(x)
    if n < nperseg * 2:
        return 0.0

    try:
        noverlap = nperseg // 2
        step = nperseg - noverlap
        n_segs = (n - nperseg) // step

        if n_segs < 2:
            return 0.0

        # Compute FFT for each segment
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        # Only keep frequencies up to f_hi for efficiency
        f_mask = freqs <= f_hi
        freqs_sel = freqs[f_mask]
        n_f = len(freqs_sel)

        # Accumulate bispectrum
        bispec = np.zeros((n_f, n_f), dtype=np.complex128)
        norm_denom = np.zeros((n_f, n_f), dtype=np.float64)

        for seg_idx in range(n_segs):
            seg_start = seg_idx * step
            seg = x[seg_start:seg_start + nperseg] * np.hanning(nperseg)
            X = np.fft.rfft(seg)[f_mask]

            for i in range(n_f):
                fi = freqs_sel[i]
                for j in range(i, n_f):
                    fj = freqs_sel[j]
                    fk = fi + fj
                    if fk > f_hi:
                        continue
                    # Find index for fk
                    k = np.argmin(np.abs(freqs_sel - fk))
                    bispec[i, j] += X[i] * X[j] * np.conj(X[k])
                    norm_denom[i, j] += np.abs(X[i] * X[j]) ** 2 * np.abs(X[k]) ** 2

        # Bicoherence = |B|² / norm
        with np.errstate(divide='ignore', invalid='ignore'):
            bicoh = np.abs(bispec) ** 2 / (norm_denom + 1e-12)
            bicoh = np.clip(bicoh, 0.0, 1.0)

        # Take maximum in δ-α region (0.5-4Hz phase driver, 8-13Hz amplitude)
        d_mask = (freqs_sel >= 0.5) & (freqs_sel <= 4.0)
        a_mask = (freqs_sel >= 8.0) & (freqs_sel <= 13.0)
        if d_mask.any() and a_mask.any():
            region = bicoh[np.ix_(d_mask, a_mask)]
            return float(np.max(region))
        return float(np.max(bicoh))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# EMG power (raw muscle activity estimator)
#   RMS of signal after 30-47 Hz bandpass → EMG contamination level.
#   Already have gamma_emg_ratio but RMS gives absolute EMG level.
#   Important for:
#     - REM detection (EMG atonia)
#     - Nociception detection (EMG spike under light anesthesia)
#     - Arousal detection
# ═══════════════════════════════════════════════════════════════════════════════

def emg_rms(x: np.ndarray, fs: float,
            emg_lo: float = 30.0, emg_hi: float = 47.0) -> float:
    """RMS of EMG band (30-47 Hz) signal, normalized by total RMS."""
    try:
        nyq = fs / 2.0
        sos = butter(4, [emg_lo / nyq, min(emg_hi, nyq * 0.99) / nyq],
                    btype="bandpass", output="sos")
        emg_signal = sosfiltfilt(sos, x)
        total_rms = np.std(x) + 1e-12
        emg_rms_val = np.std(emg_signal)
        return float(np.clip(emg_rms_val / total_rms, 0.0, 1.0))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Amplitude-aware features (capture raw µV-scale information typically lost)
# ═══════════════════════════════════════════════════════════════════════════════

def peak_to_peak_ratio(x: np.ndarray, percentile: float = 90.0) -> float:
    """(P95 - P5) / (P75 - P25) — robust peak-to-peak vs interquartile range."""
    p = np.percentile(x, [5, 25, 75, 95])
    iqr = p[2] - p[1] + 1e-12
    return float((p[3] - p[0]) / iqr)


def envelope_variance(x: np.ndarray) -> float:
    """Coefficient of variation of Hilbert envelope — amplitude modulation depth."""
    from scipy.signal import hilbert as sp_hilbert
    env = np.abs(sp_hilbert(x))
    return float(np.std(env) / (np.mean(env) + 1e-12))


def zero_crossing_rate(x: np.ndarray, fs: float) -> float:
    """Zero-crossings per second, normalized by Nyquist (fs/2)."""
    zc = np.sum(np.abs(np.diff(np.signbit(x)))) / 2.0
    rate = zc / (len(x) / fs)
    return float(np.clip(rate / (fs / 2), 0.0, 1.0))


def burst_suppression_duration(x: np.ndarray, fs: float,
                                threshold_uv: float = 5.0) -> float:
    """Mean duration of suppression episodes (seconds)."""
    suppressed = np.abs(x) < threshold_uv
    durations = []; run = 0
    for v in suppressed:
        if v: run += 1
        elif run > 0: durations.append(run / fs); run = 0
    if run > 0: durations.append(run / fs)
    return float(np.mean(durations)) if durations else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-window aggregation (captures temporal dynamics across ~30s)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureHistory:
    """Rolling history for multi-window feature aggregation."""

    def __init__(self, window_sec: float = 30.0, epoch_sec: float = 1.0):
        self.maxlen = int(window_sec / epoch_sec)
        self.buffers: dict = {}

    def update(self, features: dict) -> dict:
        agg = {}
        for key, val in features.items():
            if key not in self.buffers:
                self.buffers[key] = deque(maxlen=self.maxlen)
            self.buffers[key].append(val)
            buf = self.buffers[key]
            arr = np.array(list(buf), dtype=np.float64)
            agg[f'{key}_mean'] = float(np.mean(arr))
            agg[f'{key}_std'] = float(np.std(arr)) if len(arr) >= 3 else 0.0
            agg[f'{key}_trend'] = float(arr[-1] - arr[0]) if len(arr) >= 2 else 0.0
        return agg

    def reset(self):
        self.buffers.clear()
