"""
Feature extraction step — v12 (38-dim with advanced features).

特征顺序（每通道，与 BatchProcessor 完全一致）：
  [0-4]   relative band powers (δ θ α β γ)
  [5]     Permutation Entropy (PE)
  [6]     SEF95 (normalised to [0,1] by 47 Hz)
  [7]     Lempel-Ziv Complexity (LZC)
  [8-10]  multi-threshold Burst Suppression Ratio (BSR: 2/5/10 uV)
  [11]    spectral_slope (1/f exponent, 1=EEG-like 0=EMG-like)
  [12]    gamma_emg_ratio (P[30-47] / P[0.5-47])
  [13]    sigma_power (12-15 Hz spindle band)
  [14]    slow_oscillation (0.1-0.5 Hz, deep anesthesia biomarker)
  [15]    zero_crossing_rate (normalised to [0,1])
  [16]    hjorth_mobility (mean frequency proxy)
  [17]    hjorth_complexity (signal bandwidth)

跨通道特征（末尾）：
  alpha asymmetry (log ratio Fp1/Fp2)
  mean SQI

总维度（v12，2ch）：18 × 2 + 2 = 38

BatchProcessor（离线预处理）与 FeatureExtractor（实时推理）使用相同计算，
确保训练特征与推理特征逐位对齐，不产生偏移。
"""

from __future__ import annotations
import math
from typing import Any, Dict, List
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.signal import hilbert as sp_hilbert

from ..base import EEGStep
from ..context import EEGContext
from .advanced_features import (
    zero_crossing_rate, hjorth_mobility, hjorth_complexity
)


# ------------------------------------------------------------------ #
# Utility functions                                                    #
# ------------------------------------------------------------------ #

def _band_powers(pxx: np.ndarray, freqs: np.ndarray,
                 bands: Dict[str, list]) -> Dict[str, float]:
    total = pxx.sum() + 1e-12
    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[name] = float(pxx[mask].sum() / total)
    return result


def _sef95(pxx: np.ndarray, freqs: np.ndarray) -> float:
    """Spectral Edge Frequency: frequency below which 95% of power lies."""
    cumsum = np.cumsum(pxx)
    total = cumsum[-1]
    if total < 1e-12:
        return 0.0
    idx = np.searchsorted(cumsum, 0.95 * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _permutation_entropy(x: np.ndarray, order: int = 6, delay: int = 1) -> float:
    """
    Permutation Entropy (Bandt & Pompe, 2002).
    Normalised to [0, 1]: 0 = perfectly regular, 1 = maximally random.
    """
    n = len(x)
    if n < order * delay:
        return 0.0
    # Build ordinal patterns
    patterns = {}
    count = 0
    for i in range(n - (order - 1) * delay):
        snippet = x[i: i + order * delay: delay]
        key = tuple(np.argsort(snippet))
        patterns[key] = patterns.get(key, 0) + 1
        count += 1
    probs = np.array(list(patterns.values()), dtype=np.float64) / count
    pe = -np.sum(probs * np.log2(probs + 1e-12))
    max_pe = np.log2(math.factorial(order))
    return float(pe / (max_pe + 1e-12))


def _burst_suppression_ratio(x: np.ndarray, threshold_uv: float) -> float:
    """
    BSR at a single amplitude threshold.
    After per-patient normalisation the signal is in relative units (σ),
    so thresholds are also relative (divided by the patient's MAD-σ in loader).
    We compute envelope via |x| rather than raw amplitude to handle ESU residuals
    that raise the noise floor without being true burst activity.
    """
    envelope = np.abs(x)
    suppressed = np.sum(envelope < threshold_uv)
    return float(suppressed / len(x))


def _multi_bsr(x: np.ndarray,
               thresholds: list) -> List[float]:
    """
    Multi-threshold BSR: one ratio per threshold level.
    E.g., thresholds=[2.0, 5.0, 10.0] after normalisation
    roughly correspond to <0.1σ, <0.25σ, <0.5σ of a healthy EEG.
    """
    return [_burst_suppression_ratio(x, thr) for thr in thresholds]


def _pac_modulation_index(
    x: np.ndarray,
    fs: float,
    lo_band: tuple = (8.0, 13.0),   # alpha phase
    hi_band: tuple = (30.0, 47.0),  # gamma amplitude
) -> float:
    """
    Phase-Amplitude Coupling (PAC) Modulation Index (Tort et al., 2010).

    Measures the strength of coupling between the PHASE of low-frequency
    oscillations (alpha 8-13 Hz) and the AMPLITUDE of high-frequency
    oscillations (gamma 30-47 Hz).

    Clinical significance:
      - Healthy/awake: strong alpha-gamma PAC in prefrontal cortex
      - Propofol induction: PAC collapses rapidly at LOC (loss of consciousness)
      - Maintenance: near-zero PAC; distinguishes maintenance from induction
      - Recovery: gradual PAC restoration correlates with returning awareness

    Formula: MI = |mean(A_gamma * exp(i * phi_alpha))| / mean(A_gamma)
    Range  : [0, 1]  — 0 = no coupling, higher = stronger phase-gating

    Returns 0.0 on short windows or filter failures (safe fallback).
    """
    n = len(x)
    if n < int(fs * 0.5):   # need at least 0.5 s for meaningful bandpass
        return 0.0
    try:
        nyq = fs / 2.0
        # Alpha phase via Hilbert
        sos_lo = butter(4, [lo_band[0] / nyq, lo_band[1] / nyq],
                        btype="bandpass", output="sos")
        phi = np.angle(sp_hilbert(sosfiltfilt(sos_lo, x)))

        # Gamma envelope via Hilbert
        sos_hi = butter(4, [hi_band[0] / nyq, hi_band[1] / nyq],
                        btype="bandpass", output="sos")
        amp = np.abs(sp_hilbert(sosfiltfilt(sos_hi, x)))

        # Modulation index
        z = np.mean(amp * np.exp(1j * phi))
        return float(np.abs(z) / (np.mean(amp) + 1e-12))
    except Exception:
        return 0.0


def _lzc(x: np.ndarray) -> float:
    """
    Lempel-Ziv Complexity of a binarised signal (above/below median).
    Normalised by N/log2(N).
    """
    n = len(x)
    if n < 4:
        return 0.0
    binary = (x > np.median(x)).astype(int)
    s = "".join(map(str, binary))
    # Standard LZC algorithm (Kaspar & Schuster 1987)
    c, l, i = 1, 1, 1
    while i + l <= n:
        if s[i: i + l] in s[:i]:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    norm = (n / np.log2(n + 1e-12)) if n > 1 else 1.0
    return float(c / norm)


def _spectral_slope(pxx: np.ndarray, freqs: np.ndarray,
                    lo: float = 1.0, hi: float = 47.0) -> float:
    """
    1/f 谱斜率（log-log 线性回归），单窗口版本。

    EEG 神经信号：斜率 ≈ -2 ~ -4（1/f 幂律，陡降）
    EMG 肌电污染：斜率 ≈  0 ~ -1（宽频近白噪声，平坦）

    返回归一化值 ∈ [0, 1]：
      1.0 → 斜率 ≤ -4（纯 EEG 特征）
      0.0 → 斜率 ≥  0（纯 EMG / 白噪声）
    """
    mask = (freqs >= lo) & (freqs <= hi) & (freqs > 0)
    if mask.sum() < 2:
        return 0.5
    f_log  = np.log10(freqs[mask])
    p_log  = np.log10(pxx[mask] + 1e-12)
    f_mean = f_log.mean()
    f_cen  = f_log - f_mean
    f_var  = (f_cen ** 2).sum() + 1e-12
    slope  = float(((p_log - p_log.mean()) * f_cen).sum() / f_var)
    return float(np.clip(-slope / 4.0, 0.0, 1.0))


def _sigma_power_ratio(pxx: np.ndarray, freqs: np.ndarray,
                       sigma_lo: float = 12.0, sigma_hi: float = 15.0) -> float:
    """Relative power in sigma (spindle) band: 12-15 Hz."""
    mask = (freqs >= sigma_lo) & (freqs < sigma_hi)
    total = pxx.sum() + 1e-12
    return float(pxx[mask].sum() / total)


def _gamma_emg_ratio(pxx: np.ndarray, freqs: np.ndarray,
                     gamma_lo: float = 30.0, gamma_hi: float = 47.0,
                     low_lo: float = 0.5,   low_hi: float = 30.0) -> float:
    """
    Gamma 段相对低频段的功率比，作为 EMG gamma 污染指标，单窗口版本。

    丙泊酚麻醉下：delta/theta 主导 → 比值低（<0.15）
    EMG 污染时  ：gamma（30-47 Hz）被优先抬高 → 比值升高

    返回 P(gamma) / [P(gamma) + P(low)] ∈ (0, 1)
    """
    g_mask = (freqs >= gamma_lo) & (freqs < gamma_hi)
    l_mask = (freqs >= low_lo)   & (freqs < low_hi)
    p_gamma = float(pxx[g_mask].sum()) + 1e-12
    p_low   = float(pxx[l_mask].sum()) + 1e-12
    return float(p_gamma / (p_gamma + p_low))


# ------------------------------------------------------------------ #
# Step class                                                           #
# ------------------------------------------------------------------ #

class FeatureExtractor(EEGStep):
    """
    Computes the full feature vector and stores it in ctx.features.
    Does NOT modify ctx.data.

    特征顺序与 BatchProcessor 完全一致（训练/推理对齐）：
      per-channel: [bands | PE | SEF95 | LZC | BSR×3 | slope? | gamma_ratio?]
      inter-ch:    [alpha_asymmetry | mean_SQI]
    """

    BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

    def __init__(self, cfg: Dict[str, Any], fs: float = 128.0):
        self.fs = fs
        raw_bands = cfg.get("bands", {})
        # 只使用标准五段（与 BatchProcessor 一致，忽略配置中的 emg_lo/hi 等额外频段）
        self.bands = {k: raw_bands[k] for k in self.BAND_NAMES if k in raw_bands}
        self.pe_order = cfg.get("permutation_entropy", {}).get("order", 6)
        self.pe_delay = cfg.get("permutation_entropy", {}).get("delay", 1)
        self.compute_sef         = cfg.get("sef95", True)
        self.compute_lzc         = cfg.get("lzc", True)
        self.compute_bsr         = cfg.get("bsr", True)
        self.bsr_thresholds      = cfg.get("bsr_thresholds_uv", [2.0, 5.0, 10.0])
        # v10 EMG 分离算法特征（与 BatchProcessor 共用同一计算逻辑）
        self.compute_slope       = cfg.get("spectral_slope", False)
        self.compute_gamma_ratio = cfg.get("gamma_emg_ratio", False)
        # PAC: alpha-gamma 相位-幅度耦合（丙泊酚敏感，默认关闭）
        # 注意：PAC 不在 BatchProcessor 中实现，开启后需同步更新 BatchProcessor
        self.compute_pac = cfg.get("pac", False)
        # v12 advanced features
        self.compute_sigma        = cfg.get("sigma_power", False)
        self.compute_slow         = cfg.get("slow_oscillation", False)  # replaces DFA
        self.compute_zcr          = cfg.get("zero_crossing_rate", False)
        self.compute_hjorth_mob   = cfg.get("hjorth_mobility", False)
        self.compute_hjorth_comp  = cfg.get("hjorth_complexity", False)

    def _channel_features(self, x: np.ndarray) -> np.ndarray:
        """
        Return 1-D feature array for a single channel.

        特征布局（与 BatchProcessor.compute() 逐位对齐）：
          [0-4]  relative band powers (δ θ α β γ)
          [5]    permutation entropy
          [6]    SEF95 (归一化到 [0,1])
          [7]    LZC complexity
          [8-10] multi-threshold BSR
          [11]   spectral_slope    (若 compute_slope=True)
          [12]   gamma_emg_ratio   (若 compute_gamma_ratio=True)
          [13]   PAC               (若 compute_pac=True，默认关闭)
          [13+n_pac]    sigma_power         (v12 新增)
          [14+n_pac]    slow_oscillation     (v13: 0.1-0.5Hz, 替代 DFA)
          [15+n_pac]    zero_crossing_rate   (v12 新增)
          [16+n_pac]    hjorth_mobility      (v12 新增)
          [17+n_pac]    hjorth_complexity    (v12 新增)
        """
        nperseg = min(256, len(x))
        freqs, pxx = welch(x, fs=self.fs, nperseg=nperseg)

        feats: List[float] = []

        # Relative band powers (5 standard bands)
        bp = _band_powers(pxx, freqs, self.bands)
        for name in self.BAND_NAMES:
            feats.append(bp.get(name, 0.0))

        # Permutation Entropy
        feats.append(_permutation_entropy(x, self.pe_order, self.pe_delay))

        # SEF95 — normalize by EEG band upper limit (47 Hz) for better dynamic range.
        # Previously used /fs*2 (=/Nyquist), which compressed SEF95 into ~[0.15, 0.45].
        # Dividing by 47 Hz (lowpass cutoff) yields ~[0.2, 0.85] for typical EEG.
        # NOTE: requires HDF5 rebuild if training data used the old normalization.
        if self.compute_sef:
            sef_hz = _sef95(pxx, freqs)
            feats.append(float(np.clip(sef_hz / 47.0, 0.0, 1.0)))

        # LZC
        if self.compute_lzc:
            feats.append(_lzc(x))

        # Multi-threshold BSR
        if self.compute_bsr:
            feats.extend(_multi_bsr(x, self.bsr_thresholds))

        # EMG 谱斜率（与 BatchProcessor._batch_spectral_slope 相同逻辑）
        if self.compute_slope:
            feats.append(_spectral_slope(pxx, freqs))

        # Gamma 污染比（与 BatchProcessor._batch_gamma_emg_ratio 相同逻辑）
        if self.compute_gamma_ratio:
            feats.append(_gamma_emg_ratio(pxx, freqs))

        # PAC: alpha-gamma modulation index (disabled by default)
        if self.compute_pac:
            feats.append(_pac_modulation_index(x, self.fs))

        # v12: advanced features
        if self.compute_sigma:
            feats.append(_sigma_power_ratio(pxx, freqs))

        if self.compute_slow:
            # Slow oscillation power (0.5 Hz Welch bin, covers 0.25-0.75 Hz)
            # Deep anesthesia biomarker — Steriade 1993
            # After Welch detrending DC is removed; 0.5 Hz bin is lowest non-DC
            so_mask = (freqs > 0.0) & (freqs <= 0.5)
            total_p = pxx.sum() + 1e-12
            feats.append(float(pxx[so_mask].sum() / total_p))

        if self.compute_zcr:
            feats.append(zero_crossing_rate(x, self.fs))

        if self.compute_hjorth_mob:
            feats.append(hjorth_mobility(x))

        if self.compute_hjorth_comp:
            feats.append(hjorth_complexity(x))

        return np.array(feats, dtype=np.float32)

    def process(self, ctx: EEGContext) -> EEGContext:
        per_channel = [self._channel_features(ctx.data[ch])
                       for ch in range(ctx.n_channels)]

        # Inter-channel: alpha asymmetry (Fp1 vs Fp2)
        if ctx.n_channels >= 2:
            def alpha_power(x):
                f, p = welch(x, fs=self.fs, nperseg=min(256, len(x)))
                mask = (f >= 8.0) & (f < 13.0)
                return p[mask].sum() + 1e-12

            a1 = alpha_power(ctx.data[0])
            a2 = alpha_power(ctx.data[1])
            asymmetry = float(np.log(a1) - np.log(a2))
        else:
            asymmetry = 0.0

        # SQI score as a feature (if already computed)
        sqi_feat = float(np.mean(ctx.sqi)) if ctx.sqi is not None else 1.0

        feature_vec = np.concatenate(per_channel + [[asymmetry, sqi_feat]])
        ctx.features = feature_vec
        ctx.artifacts["features"] = feature_vec.copy()
        return ctx

    @property
    def feats_per_channel(self) -> int:
        """Total features per channel (used to stride into feature vector)."""
        n = len(self.BAND_NAMES) + 1   # 5 bands + PE
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
        if self.compute_pac:
            n += 1
        # v12 advanced features
        if self.compute_sigma:       n += 1
        if self.compute_slow:        n += 1
        if self.compute_zcr:         n += 1
        if self.compute_hjorth_mob:  n += 1
        if self.compute_hjorth_comp: n += 1
        return n

    @property
    def total_feature_dim(self) -> int:
        """Total feature vector length (model feature_dim must match this)."""
        # n_ch × per-channel + 2 inter-channel (alpha asymmetry + mean SQI)
        # Caller is responsible for knowing n_channels; we can't know it here.
        # Use FeatureExtractor(cfg, fs).total_feature_dim_for(n_ch) instead.
        raise NotImplementedError("Use total_feature_dim_for(n_ch)")

    def total_feature_dim_for(self, n_channels: int) -> int:
        """Return total feature vector length for a given number of EEG channels."""
        return self.feats_per_channel * n_channels + 2   # +2: asymmetry + SQI

    def _feature_offset(self, name: str) -> int:
        """Return the per-channel index of a named feature (derived from config flags)."""
        offset = 5 + 1   # bands + PE
        if self.compute_sef:         offset += 1
        if self.compute_lzc:         offset += 1
        if self.compute_bsr:         offset += len(self.bsr_thresholds)
        if self.compute_slope:       offset += 1
        if self.compute_gamma_ratio: offset += 1
        if self.compute_pac:         offset += 1
        if name == "spectral_slope":     return offset - (1 if self.compute_slope else 0)
        if name == "gamma_emg_ratio":    return offset - (1 if self.compute_gamma_ratio else 0)
        if name == "pac":                return offset - (1 if self.compute_pac else 0)
        if self.compute_sigma:       offset += 1
        if name == "sigma_power":        return offset - 1
        if self.compute_slow:        offset += 1
        if name == "slow_oscillation":   return offset - 1
        if self.compute_zcr:         offset += 1
        if name == "zero_crossing_rate": return offset - 1
        if self.compute_hjorth_mob:  offset += 1
        if name == "hjorth_mobility":    return offset - 1
        if self.compute_hjorth_comp: offset += 1
        if name == "hjorth_complexity":  return offset - 1
        raise ValueError(f"Unknown feature name: {name}")

    _VALIDATE_FLAGS = {
        "spectral_slope":      "compute_slope",
        "gamma_emg_ratio":     "compute_gamma_ratio",
        "sigma_power":         "compute_sigma",
        "slow_oscillation":    "compute_slow",
        "zero_crossing_rate":  "compute_zcr",
        "hjorth_mobility":     "compute_hjorth_mob",
        "hjorth_complexity":   "compute_hjorth_comp",
    }

    def _validate_feature_range(self, ctx: EEGContext, name: str, lo: float, hi: float) -> None:
        flag = self._VALIDATE_FLAGS.get(name)
        if flag is None or not getattr(self, flag, False):
            return
        idx = self._feature_offset(name)
        stride = self.feats_per_channel
        for ch in range(ctx.n_channels):
            val = float(ctx.features[ch * stride + idx])
            if not (lo <= val <= hi):
                raise ValueError(f"Ch{ch} {name}={val:.3f} outside [{lo},{hi}].")

    def validate(self, ctx: EEGContext) -> None:
        assert ctx.features is not None, "FeatureExtractor did not set ctx.features"
        if np.isnan(ctx.features).any():
            raise ValueError("FeatureExtractor produced NaN features.")
        n_bands = len(self.BAND_NAMES)
        stride  = self.feats_per_channel
        for ch in range(ctx.n_channels):
            offset   = ch * stride
            band_sum = float(ctx.features[offset: offset + n_bands].sum())
            if not (0.3 <= band_sum <= 1.3):
                raise ValueError(
                    f"Ch{ch} band power sum={band_sum:.3f} not in [0.3, 1.3]. "
                    f"Relative power normalisation may have failed."
                )
        self._validate_feature_range(ctx, "spectral_slope", 0.0, 1.0)
        self._validate_feature_range(ctx, "sigma_power", 0.0, 1.0)
        self._validate_feature_range(ctx, "zero_crossing_rate", 0.0, 1.0)
