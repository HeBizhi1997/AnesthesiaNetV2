"""
VitalLoader — reads a .vital file, applies recording-level filtering,
per-patient robust amplitude normalisation, and extracts sliding windows.

Key design decisions vs. naive implementation:

1. Label-lag correction (label_lag_sec = 15 s)
   BIS monitors compute the BIS value using an internal long-window FFT +
   bispectral analysis that introduces ~15–20 s of processing latency.
   The raw EEG is real-time; the displayed BIS value is "stale".
   Fix: pair EEG window ending at time t  with the BIS label at t + lag.
   Equivalently: when iterating label time t, take EEG from
   [t - lag - window, t - lag].

2. Per-patient robust amplitude normalisation
   Individual EEG amplitude varies enormously across patients (genetics,
   electrode contact, skull thickness). Raw µV values fed into the CNN
   would cause the model to react to inter-patient amplitude rather than
   anaesthesia-induced spectral changes.
   Fix: compute a per-patient scale factor from the first `baseline_sec`
   seconds of the filtered recording (awake / pre-induction period, where
   BIS should be > 80). Scale = median(|x|) of baseline, computed per
   channel. Each window is divided by this scale → relative amplitude.

3. Windows written in strict chronological order
   Essential for LNN SequenceDataset which reads consecutive windows to
   build temporal sequences. The loop iterates t in ascending order so
   HDF5 row i always precedes row i+1 in time.

4. Vectorized batch processing (BatchProcessor)
   Instead of calling scipy.signal.welch N times (once per 4-second
   window), all windows are stacked into (N, n_ch, T) and processed in
   one pass using numpy FFT.  ~10-20x faster per file.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

import numpy as np
import h5py
import vitaldb
import yaml

from ..pipeline.steps.filters import recording_filter
from .batch_processor import BatchProcessor


class VitalLoader:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.eeg_tracks = cfg["eeg"]["channels"]
        self.fs = float(cfg["eeg"]["srate"])
        self.target_track = cfg["labels"]["target"]
        self.sqi_track = cfg["labels"]["sqi_track"]
        self.sr_track = cfg["labels"]["sr_track"]

        w = cfg["windowing"]
        self.window_sec = w["window_sec"]
        self.stride_sec = w["stride_sec"]
        self.label_lag_sec = int(w.get("label_lag_sec", 15))
        self.min_label = w["min_valid_label"]
        self.max_label = w["max_valid_label"]
        self.baseline_sec = int(w.get("baseline_sec", 60))
        self.window_samples = int(self.window_sec * self.fs)

        f = cfg["filters"]
        wv = cfg["wavelet"]
        self._filter_kwargs = dict(
            highpass=f["highpass_hz"],
            lowpass=f["lowpass_hz"],
            notch_freqs=f["notch_hz"],
            median_kernel_ms=f["median_kernel_ms"],
            wavelet=wv["wavelet"],
            wavelet_level=wv["level"],
            esu_energy_thresh=wv["esu_energy_thresh"],
        )
        self._batch_processor = BatchProcessor(cfg, fs=self.fs)

    # ------------------------------------------------------------------ #
    # Load raw tracks via vitaldb official API                            #
    # ------------------------------------------------------------------ #

    def _load_raw(self, path: str) -> Optional[Dict[str, np.ndarray]]:
        try:
            vf = vitaldb.VitalFile(path)
        except Exception as e:
            print(f"  [WARN] Cannot open {path}: {e}")
            return None

        eeg = vf.to_numpy(self.eeg_tracks, 1.0 / self.fs)  # (T_samp, n_ch)
        if eeg is None or eeg.shape[1] != len(self.eeg_tracks):
            print(f"  [WARN] Missing EEG tracks in {path}")
            return None

        label_arr = vf.to_numpy(
            [self.target_track, self.sqi_track, self.sr_track], 1.0
        )  # (T_sec, 3)
        if label_arr is None:
            print(f"  [WARN] Missing label tracks in {path}")
            return None

        return {
            "eeg": eeg.astype(np.float32),
            "labels": label_arr.astype(np.float32),
        }

    # ------------------------------------------------------------------ #
    # Per-patient robust amplitude scale                                   #
    # ------------------------------------------------------------------ #

    def _compute_baseline_scale(
        self, eeg_filtered: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        频带比例归一化（Band-Ratio Normalization）— v8 升级版

        临床问题：
          纯 MAD 归一化在以下场景会严重失准：
          1. 患者已在深度麻醉时开始录制（无清醒段）
          2. 大量 Delta 爆发或 ESU 干扰（高幅低频）拉偏整体 MAD
          3. 1/f² 物理定律：EEG 能量集中低频，Delta 振幅比 Alpha 大 5-10×，
             全频段 MAD 被低频主导，高频细节（Alpha/Beta）被压缩失真

        频带比例检测策略：
          1. 首先计算各频段能量比（Delta, Alpha+Beta）
          2. 若 Delta 比例 > delta_dominance_thresh（90%），则切换为：
             "Alpha+Beta 自适应增益补偿"：使用中高频段（8-30 Hz）的能量作为
             归一化基准，而非被低频主导的全频段 MAD。
             物理依据：Alpha/Beta 能量对于苏醒/维持状态更加稳定，
             在爆发抑制时虽然低，但不会像 Delta 一样被 ESU 严重污染。
          3. 若无法找到清醒段且无法做频带分析，保留 MAD 回退并警告

        参数（从 config 读取）：
          use_band_ratio        : 是否启用频带比例归一化（默认 True）
          delta_dominance_thresh: Delta 主导阈值（默认 0.90）
          band_ratio_alpha_beta : Alpha+Beta 频段 [Hz]（默认 [8, 30]）

        Returns scale (n_channels,) — divide each channel by its scale.
        """
        n_ch = eeg_filtered.shape[0]
        baseline_samp = int(self.baseline_sec * self.fs)
        scale = np.zeros(n_ch, dtype=np.float64)

        # 从 config 读取频带比例参数（向后兼容：默认启用）
        use_band_ratio = self.cfg.get("normalization", {}).get(
            "use_band_ratio", True)
        delta_thresh = self.cfg.get("normalization", {}).get(
            "delta_dominance_thresh", 0.90)
        ab_band = self.cfg.get("normalization", {}).get(
            "alpha_beta_band_hz", [8.0, 30.0])

        for ch in range(n_ch):
            seg = None
            fallback = False

            # ── 步骤 1：寻找清醒段（BIS > 80）─────────────────────────────
            search_end = min(int(self.baseline_sec * 3 * self.fs),
                             eeg_filtered.shape[1])
            bis_search = labels[: int(search_end / self.fs), 0]
            awake_mask = (~np.isnan(bis_search)) & (bis_search > 80)

            if awake_mask.sum() * self.fs >= baseline_samp // 2:
                awake_sec_idx = np.where(awake_mask)[0]
                start_samp = int(awake_sec_idx[0] * self.fs)
                end_samp   = min(start_samp + baseline_samp,
                                 eeg_filtered.shape[1])
                seg = eeg_filtered[ch, start_samp:end_samp]

            if seg is None or len(seg) < 64:
                seg = eeg_filtered[ch]
                fallback = True
                if ch == 0:
                    print(
                        f"  [WARN] 归一化回退: 无法找到足够的清醒段"
                        f"(BIS>80, >={self.baseline_sec//2}s). "
                        f"尝试频带比例归一化..."
                    )

            # ── 步骤 2：频带比例检测（Delta 主导检查）─────────────────────
            if use_band_ratio and len(seg) >= int(self.fs):
                freqs   = np.fft.rfftfreq(len(seg), d=1.0 / self.fs)
                psd     = np.abs(np.fft.rfft(seg - seg.mean())) ** 2
                total_p = psd.sum() + 1e-12

                delta_mask = (freqs >= 0.5) & (freqs < 4.0)
                ab_mask    = (freqs >= ab_band[0]) & (freqs < ab_band[1])

                delta_ratio = psd[delta_mask].sum() / total_p
                ab_power    = psd[ab_mask].sum()

                if delta_ratio > delta_thresh and ab_power > 0:
                    # Delta 主导（典型场景：麻醉后无清醒段 / ESU 严重干扰）
                    # 切换为 Alpha+Beta 能量作为归一化基准
                    # 物理意义：Alpha+Beta RMS ≈ σ of "清醒脑电特征成分"
                    ab_rms = np.sqrt(ab_power / max(ab_mask.sum(), 1))
                    # 将 Alpha+Beta RMS 转换为全频段等效σ：
                    # 经验系数 2.5（清醒 EEG 中 Delta 约占 Alpha×2.5 振幅）
                    equivalent_sigma = ab_rms * 2.5
                    scale[ch] = max(equivalent_sigma, 0.1)
                    if ch == 0 and fallback:
                        print(
                            f"  [INFO] 频带比例归一化: Delta={delta_ratio*100:.1f}% > "
                            f"{delta_thresh*100:.0f}%，切换至 Alpha+Beta 归一化 "
                            f"(ab_rms={ab_rms:.4f}, sigma_eq={equivalent_sigma:.4f})"
                        )
                    continue   # 跳过下方 MAD 计算

                elif delta_ratio > delta_thresh:
                    if ch == 0:
                        print(
                            f"  [WARN] Delta 主导({delta_ratio*100:.1f}%) 但 "
                            f"Alpha+Beta 能量为零，回退到 MAD"
                        )

            # ── 步骤 3：标准 MAD 归一化（清醒段或频带检测通过） ────────────
            if fallback and ch == 0:
                print(
                    f"  [WARN] 使用整段录制 MAD 归一化. "
                    f"若录制开始时患者已麻醉则此归一化可能不准确."
                )
            mad = np.median(np.abs(seg - np.median(seg)))
            scale[ch] = max(mad / 0.6745, 0.1)   # σ ≈ MAD / 0.6745 for Gaussian

        return scale.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Main entry: process one .vital file                                 #
    # ------------------------------------------------------------------ #

    def process_file(
        self,
        vital_path: str,
        out_h5: h5py.File,
        case_id: str,
    ) -> int:
        """
        Load, filter, normalise, and batch-extract features for one .vital file.

        Steps:
          1. Load raw EEG + labels via vitaldb
          2. Interpolate NaNs on full recording
          3. MNE FIR recording-level filter
          4. Per-patient robust amplitude normalisation
          5. Collect all valid windows into (N, n_ch, T) array
          6. BatchProcessor.compute() → sqi_arr (N, n_ch), feat_arr (N, n_feat)
          7. Write to HDF5
        """
        raw = self._load_raw(vital_path)
        if raw is None:
            return 0

        eeg_raw = raw["eeg"]      # (T_samp, n_ch)
        labels = raw["labels"]    # (T_sec, 3)  [BIS, SQI, SR]

        n_sec = labels.shape[0]
        n_samp = eeg_raw.shape[0]

        # ── 1. Interpolate NaNs in full recording ──────────────────────
        eeg_clean = eeg_raw.T.astype(np.float64)   # (n_ch, T_samp)
        for ch in range(eeg_clean.shape[0]):
            nans = np.isnan(eeg_clean[ch])
            if nans.all():
                print(f"  [WARN] Ch{ch} entirely NaN in {case_id}, skipping")
                return 0
            if nans.any():
                idx = np.arange(n_samp)
                eeg_clean[ch, nans] = np.interp(
                    idx[nans], idx[~nans], eeg_clean[ch, ~nans]
                )

        # ── 2. Recording-level MNE FIR filtering ──────────────────────
        try:
            eeg_filtered = recording_filter(
                eeg_clean, self.fs, **self._filter_kwargs, verbose=False
            )  # (n_ch, T_samp) float32
        except Exception as e:
            print(f"  [WARN] Filter failed for {case_id}: {e}")
            return 0

        # ── 3. Per-patient robust amplitude normalisation ──────────────
        scale = self._compute_baseline_scale(
            eeg_filtered.astype(np.float64), labels
        )  # (n_ch,)
        eeg_normalised = eeg_filtered / scale[:, np.newaxis]  # (n_ch, T_samp)

        # ── 4. Collect all valid windows (no per-window Python loop for feats) ──
        lag = self.label_lag_sec
        win = self.window_sec

        waves_list: List[np.ndarray] = []
        label_list: List[float] = []
        time_list: List[int] = []

        for t_label in range(lag + win, n_sec, self.stride_sec):
            bis_val = labels[t_label - 1, 0]
            if np.isnan(bis_val) or bis_val < self.min_label or bis_val > self.max_label:
                continue

            eeg_end_sec = t_label - lag
            sample_end = int(eeg_end_sec * self.fs)
            sample_start = sample_end - self.window_samples

            if sample_start < 0 or sample_end > n_samp:
                continue

            window = eeg_normalised[:, sample_start:sample_end]
            if np.isnan(window).any() or np.isinf(window).any():
                continue

            waves_list.append(window.astype(np.float32))
            label_list.append(float(bis_val))
            time_list.append(eeg_end_sec)

        if not label_list:
            return 0

        # ── 5. Vectorized batch SQI + feature extraction ──────────────
        waves_arr = np.stack(waves_list, axis=0)      # (N, n_ch, T)
        sqi_arr, feats_arr = self._batch_processor.compute(waves_arr)
        # sqi_arr  : (N, n_ch)
        # feats_arr: (N, n_feat)

        labels_arr = np.array(label_list, dtype=np.float32)
        times_arr  = np.array(time_list,  dtype=np.int32)

        # ── 6. Write to HDF5 (rows in chronological order) ────────────
        grp = out_h5.require_group(case_id)
        grp.create_dataset("waves",    data=waves_arr,
                           compression="gzip", compression_opts=4)
        grp.create_dataset("features", data=feats_arr,
                           compression="gzip", compression_opts=4)
        grp.create_dataset("sqi",      data=sqi_arr,
                           compression="gzip", compression_opts=4)
        grp.create_dataset("labels",   data=labels_arr,
                           compression="gzip", compression_opts=4)
        grp.create_dataset("times",    data=times_arr,
                           compression="gzip", compression_opts=4)
        grp.attrs["n_windows"]         = len(label_list)
        grp.attrs["fs"]                = self.fs
        grp.attrs["n_channels"]        = waves_arr.shape[1]
        grp.attrs["n_features"]        = feats_arr.shape[1]
        grp.attrs["label_lag_sec"]     = lag
        grp.attrs["scale_per_channel"] = scale.tolist()

        return len(label_list)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
