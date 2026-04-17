"""
batch_processor.py — Vectorized SQI + feature computation for all windows in one file.

Instead of calling scipy.signal.welch N times (one per 4s window), we batch all
windows into (N, T) arrays and compute PSD in a single vectorized FFT pass.

Expected speedup vs sequential pipeline (profiled on 128Hz, 4s windows):
  - Welch: 3ms/window × 10k windows = 30s  →  batch FFT: ~0.5s  (60x)
  - BSR / LZC / PE: vectorized with numpy  →  ~3x each
  - Combined within-file speedup: ~10-20x
  - Plus ProcessPoolExecutor across files: ~16x on 24-core machine
  - Total: 75 files × 33s sequential = ~41 min  →  ~2-3 min

EMG 分离策略（v10）：
  - 不依赖 47 Hz 以上频段（超出 Nyquist 或低通截止，恒为零）
  - 改用两个在标准 EEG 带宽内可靠计算的 EMG 污染指标：
      1. spectral_slope  : 1/f 谱斜率（EMG 污染使谱变平，斜率趋近 0）
      2. gamma_emg_ratio : P(30-47 Hz) / P(0.5-30 Hz)（EMG 优先污染 gamma 段）
  - 与 FeatureExtractor（推理路径）使用完全相同的计算，确保训练/推理特征一致

Public API:
    BatchProcessor(cfg, fs)
    processor.compute(windows)  # windows: (N, n_ch, T) → (sqi_arr, feat_arr)
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Batch Welch PSD  (matches scipy.signal.welch defaults exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _batch_welch(
    windows_2d: np.ndarray,   # (N, T)
    fs: float,
    nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD estimate for N windows simultaneously.

    Matches scipy.signal.welch(x, fs=fs, nperseg=nperseg) output.
    Default overlap = nperseg // 2, hann window, density scaling.

    Returns:
        freqs : (F,)    frequency bins in Hz
        pxx   : (N, F)  one-sided power spectral density
    """
    N, T = windows_2d.shape
    nperseg = min(nperseg, T)
    noverlap = nperseg // 2
    step = nperseg - noverlap

    # Build Hann window + normalisation factor
    hann = np.hanning(nperseg).astype(np.float64)
    win_norm = (hann ** 2).sum()   # for density scaling

    # Collect segment start indices
    seg_starts = np.arange(0, T - nperseg + 1, step)
    if len(seg_starts) == 0:
        seg_starts = np.array([0])
        nperseg = T
        hann = np.hanning(nperseg).astype(np.float64)
        win_norm = (hann ** 2).sum()

    n_segs = len(seg_starts)

    # Extract segments: (N, n_segs, nperseg)
    idx = seg_starts[:, np.newaxis] + np.arange(nperseg)[np.newaxis, :]  # (n_segs, nperseg)
    segs = windows_2d[:, idx]   # (N, n_segs, nperseg)

    # Detrend (subtract mean of each segment — matches scipy default 'constant')
    segs = segs - segs.mean(axis=-1, keepdims=True)

    # Apply window
    segs = segs * hann[np.newaxis, np.newaxis, :]

    # FFT: (N, n_segs, F)
    fft_out = np.fft.rfft(segs, axis=-1)
    pxx_segs = (np.abs(fft_out) ** 2) / (win_norm * fs)

    # Average over segments
    pxx = pxx_segs.mean(axis=1)   # (N, F)

    # One-sided: double all bins except DC and Nyquist
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    if nperseg % 2 == 0:
        pxx[:, 1:-1] *= 2.0
    else:
        pxx[:, 1:] *= 2.0

    return freqs, pxx


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized feature helpers
# ──────────────────────────────────────────────────────────────────────────────

def _batch_band_powers(
    pxx: np.ndarray,    # (N, F)
    freqs: np.ndarray,  # (F,)
    bands: Dict[str, list],
    band_names: list,
) -> np.ndarray:
    """Return (N, len(band_names)) relative band power matrix."""
    total = pxx.sum(axis=1, keepdims=True) + 1e-12   # (N, 1)
    out = np.zeros((len(pxx), len(band_names)), dtype=np.float32)
    for i, name in enumerate(band_names):
        if name not in bands:
            continue
        lo, hi = bands[name]
        mask = (freqs >= lo) & (freqs < hi)
        out[:, i] = pxx[:, mask].sum(axis=1) / total[:, 0]
    return out


def _batch_sef95(pxx: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """SEF95 for N windows.  Returns (N,) frequency values in Hz."""
    cumsum = np.cumsum(pxx, axis=1)
    total = cumsum[:, -1:]    # (N, 1)
    target = 0.95 * total
    # For each row, find first index where cumsum >= target
    # Use searchsorted per row
    N = pxx.shape[0]
    sef = np.zeros(N, dtype=np.float32)
    for i in range(N):
        idx = np.searchsorted(cumsum[i], target[i, 0])
        idx = min(idx, len(freqs) - 1)
        sef[i] = freqs[idx]
    return sef


def _batch_permutation_entropy(
    windows_2d: np.ndarray,   # (N, T)
    order: int = 6,
    delay: int = 1,
) -> np.ndarray:
    """
    Permutation Entropy for N windows (vectorized argsort, looped counting).

    Returns (N,) values in [0, 1].
    """
    N, T = windows_2d.shape
    n_snippets = T - (order - 1) * delay
    if n_snippets <= 0:
        return np.zeros(N, dtype=np.float32)

    # Build snippet offset indices once
    offsets = np.arange(order) * delay                        # (order,)
    starts  = np.arange(n_snippets)                           # (n_snippets,)
    # Gather all snippets: (N, n_snippets, order)
    idx = starts[:, np.newaxis] + offsets[np.newaxis, :]      # (n_snippets, order)
    snippets = windows_2d[:, idx]                             # (N, n_snippets, order)

    # Ordinal ranks for each snippet: (N, n_snippets, order)
    ranks = np.argsort(snippets, axis=-1).astype(np.int8)

    # Encode each rank permutation as a single integer (base=order)
    base = np.array([order ** i for i in range(order - 1, -1, -1)], dtype=np.int32)
    pattern_ids = (ranks.astype(np.int32) * base).sum(axis=-1)  # (N, n_snippets)

    max_id = order ** order
    max_pe = math.log2(math.factorial(order))
    pe_values = np.empty(N, dtype=np.float32)

    for i in range(N):
        counts = np.bincount(pattern_ids[i], minlength=max_id)
        nonzero = counts[counts > 0]
        probs = nonzero / n_snippets
        pe = -np.sum(probs * np.log2(probs + 1e-12))
        pe_values[i] = pe / (max_pe + 1e-12)

    return pe_values


def _batch_lzc(windows_2d: np.ndarray) -> np.ndarray:
    """
    Lempel-Ziv Complexity for N windows.
    Binary signal: above/below median per window.
    Returns (N,) normalised values.
    """
    N, n = windows_2d.shape
    medians = np.median(windows_2d, axis=1, keepdims=True)
    binary = (windows_2d > medians).astype(np.uint8)   # (N, n)
    norm = n / np.log2(n + 1e-12) if n > 1 else 1.0

    lzc_vals = np.empty(N, dtype=np.float32)
    for i in range(N):
        s = binary[i]
        c = 1
        l = 1
        j = 1
        # Build sub-string lookup from a set (O(n) amortised)
        seen: set = set()
        k = 0
        while j + l <= n:
            sub = s[j: j + l].tobytes()
            if sub in seen or s[k: k + l].tobytes() == sub:
                # extend current component
                l += 1
            else:
                c += 1
                seen.add(s[k: k + l].tobytes())
                k = j
                j += l
                l = 1
        lzc_vals[i] = c / norm
    return lzc_vals


def _batch_bsr(windows_2d: np.ndarray, thresholds: list) -> np.ndarray:
    """
    Multi-threshold BSR for N windows.
    Returns (N, len(thresholds)) array.
    """
    envelope = np.abs(windows_2d)                    # (N, T)
    n = windows_2d.shape[1]
    out = np.zeros((len(windows_2d), len(thresholds)), dtype=np.float32)
    for j, thr in enumerate(thresholds):
        out[:, j] = (envelope < thr).sum(axis=1) / n
    return out


def _batch_spectral_slope(
    pxx: np.ndarray,    # (N, F)
    freqs: np.ndarray,  # (F,)
    lo: float = 1.0,
    hi: float = 47.0,
) -> np.ndarray:
    """
    1/f 谱斜率（log-log 线性回归），N 窗口批量计算。

    物理意义：
      EEG  : 斜率 ≈ -2 ~ -4（陡降，神经振荡 1/f 特性）
      EMG  : 斜率 ≈  0 ~ -1（平坦，宽频肌电噪声）
      混合 : 斜率介于两者之间，gamma 段被 EMG 抬高

    返回 (N,) 归一化值 ∈ [0, 1]：
      1 → 斜率 ≤ -4（纯 EEG）
      0 → 斜率 ≥  0（纯 EMG / 白噪声）
    """
    mask = (freqs >= lo) & (freqs <= hi) & (freqs > 0)
    if mask.sum() < 2:
        return np.full(len(pxx), 0.5, dtype=np.float32)

    f_log = np.log10(freqs[mask])                          # (F,)
    p_log = np.log10(pxx[:, mask] + 1e-12)                 # (N, F)

    f_mean = f_log.mean()
    f_cen  = f_log - f_mean                                # (F,)
    f_var  = (f_cen ** 2).sum() + 1e-12

    p_cen  = p_log - p_log.mean(axis=1, keepdims=True)    # (N, F)
    slopes = (p_cen * f_cen).sum(axis=1) / f_var          # (N,)  负值

    # 归一化：斜率 -4 → 1.0，斜率 0 → 0.0，超出范围截断
    return np.clip(-slopes / 4.0, 0.0, 1.0).astype(np.float32)


def _batch_gamma_emg_ratio(
    pxx: np.ndarray,        # (N, F)
    freqs: np.ndarray,      # (F,)
    gamma_lo: float = 30.0,
    gamma_hi: float = 47.0,
    low_lo:   float = 0.5,
    low_hi:   float = 30.0,
) -> np.ndarray:
    """
    Gamma 段相对低频段的功率比，作为 EMG gamma 污染指标。

    物理意义：
      丙泊酚麻醉下 EEG 以 delta/theta 为主 → 比值低（<0.15）
      EMG 污染优先抬高 gamma（30-47 Hz）→ 比值升高
      比值 = P(gamma) / [P(gamma) + P(low)]  ∈ (0, 1)

    返回 (N,) 值，高值表示 EMG 污染程度高。
    """
    g_mask = (freqs >= gamma_lo) & (freqs < gamma_hi)
    l_mask = (freqs >= low_lo)   & (freqs < low_hi)

    p_gamma = pxx[:, g_mask].sum(axis=1) + 1e-12          # (N,)
    p_low   = pxx[:, l_mask].sum(axis=1) + 1e-12          # (N,)

    ratio = p_gamma / (p_gamma + p_low)
    return ratio.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Batch SQI
# ──────────────────────────────────────────────────────────────────────────────

def _batch_sqi(
    windows_2d: np.ndarray,   # (N, T)
    freqs: np.ndarray,        # (F,) — from batch_welch
    pxx: np.ndarray,          # (N, F)
    kurtosis_thresh: float = 5.0,
    high_freq_ratio_thresh: float = 0.4,
    fs: float = 128.0,
) -> np.ndarray:
    """
    Returns (N,) SQI scores in [0, 1].

    Three sub-scores combined with harmonic mean (matches SQIComputer):
      1. Autocorrelation lag-1 smoothness
      2. Excess-kurtosis score
      3. Spectral quality ratio (with ESU penalty)
    """
    N, T = windows_2d.shape

    # --- flat-line guard ---
    ptp = windows_2d.max(axis=1) - windows_2d.min(axis=1)   # (N,)

    # 1. Autocorrelation at lag 1 (vectorized)
    x = windows_2d - windows_2d.mean(axis=1, keepdims=True)
    var = (x ** 2).mean(axis=1) + 1e-12
    ac1 = (x[:, :-1] * x[:, 1:]).mean(axis=1) / var         # (N,)
    ac_score = np.clip(ac1, 0.0, 1.0).astype(np.float32)

    # 2. Excess kurtosis (vectorized; fisher=True → subtract 3)
    std = np.sqrt(var)
    z = x / std[:, np.newaxis]
    kurt = (z ** 4).mean(axis=1) - 3.0                       # (N,)
    kurt_score = np.clip(1.0 - np.abs(kurt) / kurtosis_thresh, 0.0, 1.0).astype(np.float32)

    # 3. Spectral quality
    total_pow = pxx.sum(axis=1) + 1e-12                      # (N,)
    eeg_mask = (freqs >= 0.5) & (freqs <= 47.0)
    eeg_pow  = pxx[:, eeg_mask].sum(axis=1)
    spec_score = np.clip(eeg_pow / total_pow, 0.0, 1.0).astype(np.float32)

    high_mask = freqs > 50.0
    if high_mask.any():
        high_ratio = pxx[:, high_mask].sum(axis=1) / total_pow
        spec_score[high_ratio > high_freq_ratio_thresh] *= 0.5

    # Harmonic mean of 3 sub-scores
    sub = np.stack([ac_score, kurt_score, spec_score], axis=1)   # (N, 3)
    sub = np.clip(sub, 1e-6, 1.0)
    sqi = 3.0 / np.sum(1.0 / sub, axis=1)                        # (N,)

    # Flat-line → 0
    sqi[ptp < 1e-6] = 0.0

    return sqi.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Public class
# ──────────────────────────────────────────────────────────────────────────────

class BatchProcessor:
    """
    Processes an entire file's worth of EEG windows in one vectorized pass.

    Usage::

        processor = BatchProcessor(cfg, fs=128.0)
        sqi_arr, feat_arr = processor.compute(windows)
        # windows  : (N, n_ch, T)   np.float32, already normalised
        # sqi_arr  : (N, n_ch)      float32
        # feat_arr : (N, n_feat)    float32, n_feat = n_ch * feats_per_ch + 2
        #
        # v9  config (standard bands only):          feats_per_ch=11  →  n_feat=24
        # v10 config (+ spectral_slope + gamma_emg): feats_per_ch=13  →  n_feat=28
        #
        # 特征顺序（每通道）：
        #   [0-4]  relative band powers (δ θ α β γ)
        #   [5]    permutation entropy
        #   [6]    SEF95  (归一化到 [0,1])
        #   [7]    LZC complexity
        #   [8-10] multi-threshold BSR
        #   [11]   spectral_slope    (EMG 指标：谱斜率，1=EEG-like，0=EMG-like)
        #   [12]   gamma_emg_ratio   (EMG 指标：gamma 段污染比)
        # 末尾两维（跨通道）：
        #   [-2]   alpha asymmetry
        #   [-1]   mean SQI
        #
        # 与 FeatureExtractor（推理路径）特征顺序完全一致，确保训练/推理不偏移。
    """

    # 只使用标准 EEG 频段（extras 已由 spectral_slope / gamma_emg_ratio 替代）
    _STANDARD_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

    def __init__(self, cfg: Dict[str, Any], fs: float = 128.0):
        self.fs = fs
        feat_cfg = cfg.get("features", {})
        sqi_cfg  = cfg.get("sqi", {})

        raw_bands = feat_cfg.get("bands", {})
        # 只保留标准五段；忽略 emg_lo/emg_hi 等超出 Nyquist 或低通截止的频段
        self.band_names = [k for k in self._STANDARD_BANDS if k in raw_bands]
        self.bands      = {k: raw_bands[k] for k in self.band_names}

        self.pe_order  = feat_cfg.get("permutation_entropy", {}).get("order", 6)
        self.pe_delay  = feat_cfg.get("permutation_entropy", {}).get("delay", 1)
        self.compute_sef         = feat_cfg.get("sef95", True)
        self.compute_lzc         = feat_cfg.get("lzc", True)
        self.compute_bsr         = feat_cfg.get("bsr", True)
        self.bsr_thresholds      = feat_cfg.get("bsr_thresholds_uv", [2.0, 5.0, 10.0])
        # v10 EMG 分离算法特征（在低通截止内可靠计算）
        self.compute_slope       = feat_cfg.get("spectral_slope", False)
        self.compute_gamma_ratio = feat_cfg.get("gamma_emg_ratio", False)

        self.kurtosis_thresh        = sqi_cfg.get("kurtosis_thresh", 5.0)
        self.high_freq_ratio_thresh = sqi_cfg.get("high_freq_ratio_thresh", 0.4)

    @property
    def feats_per_channel(self) -> int:
        n = len(self.band_names) + 1   # N bands + PE
        if self.compute_sef:
            n += 1
        if self.compute_lzc:
            n += 1
        if self.compute_bsr:
            n += len(self.bsr_thresholds)
        if self.compute_slope:
            n += 1
        if self.compute_gamma_ratio:
            n += 1
        return n

    def compute(
        self,
        windows: np.ndarray,   # (N, n_ch, T)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            sqi_arr  : (N, n_ch)   float32
            feat_arr : (N, n_feat) float32
        """
        N, n_ch, T = windows.shape
        nperseg = min(256, T)

        # ── Compute PSD for all windows × all channels at once ────────────
        # Reshape to (N*n_ch, T) for a single batch FFT call
        flat = windows.reshape(N * n_ch, T).astype(np.float64)
        freqs, pxx_flat = _batch_welch(flat, self.fs, nperseg=nperseg)
        # pxx_flat: (N*n_ch, F)
        pxx = pxx_flat.reshape(N, n_ch, -1)          # (N, n_ch, F)

        # ── SQI (per channel, then stored as-is) ─────────────────────────
        sqi_arr = np.zeros((N, n_ch), dtype=np.float32)
        for ch in range(n_ch):
            sqi_arr[:, ch] = _batch_sqi(
                windows[:, ch, :].astype(np.float64),
                freqs,
                pxx[:, ch, :],
                kurtosis_thresh=self.kurtosis_thresh,
                high_freq_ratio_thresh=self.high_freq_ratio_thresh,
                fs=self.fs,
            )

        # ── Per-channel features ──────────────────────────────────────────
        fpc = self.feats_per_channel
        channel_feats = np.zeros((N, n_ch, fpc), dtype=np.float32)

        for ch in range(n_ch):
            x  = windows[:, ch, :].astype(np.float64)   # (N, T)
            p  = pxx[:, ch, :]                           # (N, F)

            col = 0

            # Band powers (5 standard + any extra, e.g. emg_lo/emg_hi for v10)
            n_bands = len(self.band_names)
            bp = _batch_band_powers(p, freqs, self.bands, self.band_names)  # (N, n_bands)
            channel_feats[:, ch, col:col + n_bands] = bp
            col += n_bands

            # Permutation entropy
            pe = _batch_permutation_entropy(x, self.pe_order, self.pe_delay)
            channel_feats[:, ch, col] = pe
            col += 1

            # SEF95 (normalised to [0, 1])
            if self.compute_sef:
                sef = _batch_sef95(p, freqs)
                channel_feats[:, ch, col] = sef / self.fs * 2.0
                col += 1

            # LZC
            if self.compute_lzc:
                lzc = _batch_lzc(x)
                channel_feats[:, ch, col] = lzc
                col += 1

            # Multi-threshold BSR
            if self.compute_bsr:
                bsr = _batch_bsr(x, self.bsr_thresholds)   # (N, n_thr)
                n_thr = len(self.bsr_thresholds)
                channel_feats[:, ch, col:col + n_thr] = bsr
                col += n_thr

            # EMG 谱斜率（1/f exponent）：平坦 → 多 EMG，陡降 → 纯 EEG
            if self.compute_slope:
                slope = _batch_spectral_slope(p, freqs)     # (N,)
                channel_feats[:, ch, col] = slope
                col += 1

            # Gamma 污染比：P(30-47 Hz) / [P(30-47) + P(0.5-30)]
            if self.compute_gamma_ratio:
                gamma_r = _batch_gamma_emg_ratio(p, freqs)  # (N,)
                channel_feats[:, ch, col] = gamma_r
                col += 1

        # ── Inter-channel features ────────────────────────────────────────
        # Alpha asymmetry (Fp1 vs Fp2)
        if n_ch >= 2:
            alpha_lo, alpha_hi = 8.0, 13.0
            a_mask = (freqs >= alpha_lo) & (freqs < alpha_hi)
            a1 = pxx[:, 0, :][:, a_mask].sum(axis=1) + 1e-12   # (N,)
            a2 = pxx[:, 1, :][:, a_mask].sum(axis=1) + 1e-12   # (N,)
            asymmetry = (np.log(a1) - np.log(a2)).astype(np.float32)
        else:
            asymmetry = np.zeros(N, dtype=np.float32)

        # Mean SQI across channels (appended as last feature — filled later)
        sqi_mean = sqi_arr.mean(axis=1)   # (N,)

        # Flatten per-channel features and concatenate inter-channel
        # Layout: [ch0_feats | ch1_feats | asymmetry | sqi_mean]
        flat_ch = channel_feats.reshape(N, n_ch * fpc)
        feat_arr = np.concatenate(
            [flat_ch,
             asymmetry[:, np.newaxis],
             sqi_mean[:, np.newaxis]],
            axis=1,
        ).astype(np.float32)

        return sqi_arr, feat_arr
