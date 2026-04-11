"""
test_inference.py -- End-to-end single-file inference + interactive HTML report.

Pipeline:
  1. Load raw EEG from .vital file (vitaldb)
  2. MNE FIR recording-level filter -> per-patient MAD normalisation
  3. BatchProcessor: SQI + 24-dim feature vector per window
  4. Streaming LNN inference (hidden state carried across full surgery)
  5. DSA (Density Spectral Array): scipy spectrogram -> log-power heatmap
  6. Band decomposition: butter bandpass per band, downsampled for display
  7. HTML report: BIS comparison | DSA + time slider | Band waves | Features | Stats

Usage:
  python scripts/test_inference.py
  python scripts/test_inference.py --vital raw_data/2000.vital
  python scripts/test_inference.py --vital raw_data/2000.vital \\
      --checkpoint outputs/checkpoints/best_model.pt \\
      --out outputs/reports/case_2000.html
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from scipy.signal import butter, sosfiltfilt, spectrogram as scipy_spectrogram
from tqdm import tqdm

import vitaldb

from src.pipeline.steps.filters import recording_filter
from src.data.batch_processor import BatchProcessor
from src.models.anesthesia_net import AnesthesiaNet
from src.models.anesthesia_net_v2 import AnesthesiaNetV2
from src.data.loader import load_config


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """Interpolate NaN in a 1-D array."""
    nans = np.isnan(arr)
    if nans.all():
        arr[:] = 0.0
        return arr
    if nans.any():
        idx = np.arange(len(arr))
        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def _mad_scale(seg: np.ndarray) -> float:
    mad = np.median(np.abs(seg - np.median(seg)))
    return max(float(mad / 0.6745), 0.1)


def _butter_band(lo: float, hi: float, fs: float = 128.0, order: int = 4):
    nyq = fs / 2.0
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


# ─────────────────────────────────────────────────────────────────────────────
# Main inference engine
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self, cfg: dict, checkpoint: str | None = None,
                 device: str = "auto"):
        self.cfg = cfg
        self.fs = float(cfg["eeg"]["srate"])
        self.window_sec = cfg["windowing"]["window_sec"]
        self.stride_sec = cfg["windowing"]["stride_sec"]
        self.lag = int(cfg["windowing"]["label_lag_sec"])
        self.win_samp = int(self.window_sec * self.fs)
        self.min_label = cfg["windowing"]["min_valid_label"]
        self.max_label = cfg["windowing"]["max_valid_label"]
        self.baseline_sec = int(cfg["windowing"].get("baseline_sec", 60))

        f = cfg["filters"]
        wv = cfg["wavelet"]
        self._filter_kwargs = dict(
            highpass=f["highpass_hz"], lowpass=f["lowpass_hz"],
            notch_freqs=f["notch_hz"], median_kernel_ms=f["median_kernel_ms"],
            wavelet=wv["wavelet"], wavelet_level=wv["level"],
            esu_energy_thresh=wv["esu_energy_thresh"],
        )
        self._bp = BatchProcessor(cfg, fs=self.fs)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.model_version = "v1"
        self.checkpoint_path = checkpoint
        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
            # Detect model version from checkpoint filename or config
            model_ver = cfg.get("training", {}).get("model_version", "v1")
            if "v2" in str(checkpoint) or model_ver == "v2":
                self.model = AnesthesiaNetV2.from_config(cfg)
                self.model_version = "v2"
            else:
                self.model = AnesthesiaNet.from_config(cfg)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            ep = ckpt.get("epoch", "?")
            mae = ckpt.get("val_mae", float("nan"))
            mv = "V2 (multi-task)" if self.model_version == "v2" else "V1"
            print(f"  Model {mv} loaded: epoch {ep}, val_MAE={mae:.2f} BIS")
        else:
            print("  No checkpoint found -- preprocessing-only mode.")

    # ── Load raw vital ────────────────────────────────────────────────────────

    def load_vital(self, path: str) -> dict | None:
        print(f"Loading {path} ...")
        try:
            vf = vitaldb.VitalFile(path)
        except Exception as e:
            print(f"  ERROR: {e}")
            return None

        eeg_tracks = self.cfg["eeg"]["channels"]
        lab_tracks = [self.cfg["labels"]["target"],
                      self.cfg["labels"]["sqi_track"],
                      self.cfg["labels"]["sr_track"]]

        eeg = vf.to_numpy(eeg_tracks, 1.0 / self.fs)   # (T_samp, n_ch)
        labels = vf.to_numpy(lab_tracks, 1.0)            # (T_sec, 3)

        if eeg is None or labels is None:
            print("  ERROR: required tracks missing")
            return None

        print(f"  Duration: {eeg.shape[0] / self.fs / 60:.1f} min  "
              f"({eeg.shape[0]} samples, {labels.shape[0]} label-seconds)")
        return {"eeg": eeg.astype(np.float32), "labels": labels.astype(np.float32)}

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, raw: dict) -> dict:
        eeg_raw = raw["eeg"]       # (T_samp, n_ch)
        labels  = raw["labels"]    # (T_sec, 3)
        n_samp, n_ch = eeg_raw.shape

        # Interpolate NaNs
        eeg_clean = eeg_raw.T.copy().astype(np.float64)  # (n_ch, T_samp)
        for ch in range(n_ch):
            _interp_nans(eeg_clean[ch])

        # MNE FIR filter on full recording
        print("  MNE FIR filtering ...")
        t0 = time.time()
        try:
            eeg_filt = recording_filter(
                eeg_clean, self.fs, **self._filter_kwargs, verbose=False
            ).astype(np.float64)   # (n_ch, T_samp)
        except Exception as e:
            print(f"  Filter failed ({e}), using raw signal")
            eeg_filt = eeg_clean
        print(f"  Filter done in {time.time()-t0:.1f}s")

        # Per-patient MAD normalisation
        scale = np.zeros(n_ch)
        for ch in range(n_ch):
            # Search for awake baseline (BIS > 80) in first 3×baseline_sec
            search_end = min(int(self.baseline_sec * 3 * self.fs), n_samp)
            bis_search = labels[:int(search_end / self.fs), 0]
            awake_mask = (~np.isnan(bis_search)) & (bis_search > 80)
            baseline_samp = int(self.baseline_sec * self.fs)
            seg = None
            if awake_mask.sum() * self.fs >= baseline_samp // 2:
                idx0 = int(np.where(awake_mask)[0][0] * self.fs)
                idx1 = min(idx0 + baseline_samp, n_samp)
                seg = eeg_filt[ch, idx0:idx1]
            if seg is None or len(seg) < 64:
                seg = eeg_filt[ch]
            scale[ch] = _mad_scale(seg)

        eeg_norm = eeg_filt / scale[:, np.newaxis]    # (n_ch, T_samp)

        return {
            "eeg_filtered": eeg_filt,   # (n_ch, T_samp) float64, physical units µV
            "eeg_norm":     eeg_norm,   # (n_ch, T_samp) float64, normalised
            "labels":       labels,
            "scale":        scale,
            "n_samp":       n_samp,
            "n_ch":         n_ch,
        }

    # ── DSA (Density Spectral Array) ─────────────────────────────────────────

    def compute_dsa(self, eeg_filt: np.ndarray, ch: int = 0,
                    max_freq: float = 30.0) -> dict:
        """Compute spectrogram for DSA display.  Returns dict with 2D power."""
        f, t, Sxx = scipy_spectrogram(
            eeg_filt[ch], fs=self.fs,
            nperseg=self.win_samp,           # 4-second window
            noverlap=self.win_samp - int(self.stride_sec * self.fs),
            scaling="density",
        )
        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
        mask = f <= max_freq
        return {
            "freqs":  f[mask].tolist(),
            "times":  t.tolist(),
            "power":  Sxx_db[mask, :].tolist(),   # list-of-lists [freq][time]
        }

    # ── Band decomposition ────────────────────────────────────────────────────

    def compute_band_waves(self, eeg_filt: np.ndarray,
                           ds: int = 16) -> dict:
        """Return downsampled band-filtered waveforms.  ds=16 -> 8 Hz display."""
        bands_cfg = self.cfg["features"]["bands"]
        band_defs = {
            "delta": bands_cfg["delta"],
            "theta": bands_cfg["theta"],
            "alpha": bands_cfg["alpha"],
            "beta":  bands_cfg["beta"],
            "gamma": bands_cfg["gamma"],
        }
        n_ch = eeg_filt.shape[0]
        out: dict = {"ds_hz": self.fs / ds}
        n_samp = eeg_filt.shape[1]
        t_full = np.arange(n_samp) / self.fs      # seconds
        out["time"] = t_full[::ds].tolist()

        for band, (lo, hi) in band_defs.items():
            sos = _butter_band(lo, hi, self.fs)
            for ch in range(n_ch):
                filtered = sosfiltfilt(sos, eeg_filt[ch])
                out[f"{band}_ch{ch+1}"] = filtered[::ds].tolist()

        return out

    # ── Window extraction + BatchProcessor ───────────────────────────────────

    def extract_windows(self, pre: dict) -> dict:
        """Slide windows, compute SQI + features.  Returns arrays in time order."""
        eeg_norm = pre["eeg_norm"]
        labels   = pre["labels"]
        n_samp, n_ch = pre["n_samp"], pre["n_ch"]
        lag = self.lag
        win = self.window_sec

        waves_list, feat_list, time_list, label_list = [], [], [], []

        n_sec = labels.shape[0]
        print(f"  Extracting windows (lag={lag}s, win={win}s, stride={self.stride_sec}s) ...")

        for t_label in range(lag + win, n_sec, self.stride_sec):
            bis_val = labels[t_label - 1, 0]
            if np.isnan(bis_val) or bis_val < self.min_label or bis_val > self.max_label:
                continue
            eeg_end_sec = t_label - lag
            s_end = int(eeg_end_sec * self.fs)
            s_start = s_end - self.win_samp
            if s_start < 0 or s_end > n_samp:
                continue
            w = eeg_norm[:, s_start:s_end]
            if np.isnan(w).any() or np.isinf(w).any():
                continue
            waves_list.append(w.astype(np.float32))
            label_list.append(float(bis_val))
            time_list.append(float(eeg_end_sec))

        if not waves_list:
            print("  No valid windows found.")
            return {}

        waves_arr = np.stack(waves_list, axis=0)     # (N, n_ch, win_samp)
        print(f"  Running BatchProcessor on {len(waves_arr)} windows ...")
        sqi_arr, feat_arr = self._bp.compute(waves_arr)

        return {
            "waves":    waves_arr,
            "features": feat_arr,
            "sqi":      sqi_arr,
            "labels":   np.array(label_list, dtype=np.float32),
            "times":    np.array(time_list,  dtype=np.float32),
        }

    # ── Streaming model inference ─────────────────────────────────────────────

    def run_inference(self, windows: dict, chunk: int | None = None) -> dict:
        """
        Pass full surgery through model with propagated hidden state.

        Returns dict with:
          pred_bis   : (N,) float32 -- predicted BIS [0,100]
          phases     : (N,) int8    -- predicted surgical phase (v2 only)
          stim       : (N,) float32 -- stimulation probability (v2 only)
        """
        if self.model is None:
            return {"pred_bis": None, "phases": None, "stim": None}

        if chunk is None:
            chunk = self.cfg["training"].get("seq_len", 10)

        N      = len(windows["labels"])
        waves  = windows["waves"]
        feats  = windows["features"]
        sqis   = windows["sqi"]
        h = None
        all_preds  = np.empty(N, dtype=np.float32)
        all_phases = np.zeros(N, dtype=np.int8)
        all_stim   = np.zeros(N, dtype=np.float32)

        import sys as _sys
        _use_tqdm = _sys.stdout.isatty()
        _iter = tqdm(range(0, N, chunk), desc="  Inference", leave=False) if _use_tqdm \
                else range(0, N, chunk)
        print(f"  Streaming inference ({self.model_version}) on {N} windows (chunk={chunk}) ...")

        with torch.no_grad():
            for i in _iter:
                j = min(i + chunk, N)
                actual = j - i
                wt = torch.from_numpy(waves[i:j]).unsqueeze(0).to(self.device)
                ft = torch.from_numpy(feats[i:j]).unsqueeze(0).to(self.device)
                st = torch.from_numpy(sqis[i:j]).unsqueeze(0).to(self.device)

                if self.model_version == "v2":
                    pred_bis_t, phase_logits, stim_logits, _corr, h = \
                        self.model(wt, ft, st, hx=h)
                    # pred_bis_t: (1, actual, 1) normalised [0,1]
                    all_preds[i:j] = pred_bis_t[0, :actual, 0].cpu().float().numpy() * 100.0
                    # phase: (1, actual, 4) -> argmax
                    all_phases[i:j] = phase_logits[0, :actual].argmax(-1).cpu().numpy().astype(np.int8)
                    # stim: (1, actual, 1) -> sigmoid prob
                    all_stim[i:j] = torch.sigmoid(stim_logits[0, :actual, 0]).cpu().float().numpy()
                else:
                    _, pred_seq, h = self.model(wt, ft, st, hx=h)
                    all_preds[i:j] = pred_seq[0, :actual, 0].cpu().float().numpy() * 100.0

                h = h.detach()

        return {
            "pred_bis": all_preds,
            "phases":   all_phases if self.model_version == "v2" else None,
            "stim":     all_stim   if self.model_version == "v2" else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_NAMES = ["Pre-Op", "Induction", "Maintenance", "Recovery"]

def _phase_stats(labels: np.ndarray, infer: dict) -> dict:
    pred   = infer.get("pred_bis")
    phases = infer.get("phases")   # (N,) int8 predicted phase, or None
    stim   = infer.get("stim")     # (N,) float32 stim prob, or None

    # BIS-range based zones (for legacy display)
    bis_induction   = labels >= 60
    bis_maintenance = (labels >= 40) & (labels < 60)
    bis_deep        = labels < 40
    total = len(labels)

    def mae(mask):
        if pred is None or mask.sum() == 0:
            return None
        return float(np.abs(pred[mask] - labels[mask]).mean())

    stats = {
        "n_total":       total,
        "n_induction":   int(bis_induction.sum()),
        "n_maintenance": int(bis_maintenance.sum()),
        "n_deep":        int(bis_deep.sum()),
        "pct_induction":   float(bis_induction.mean() * 100),
        "pct_maintenance": float(bis_maintenance.mean() * 100),
        "pct_deep":        float(bis_deep.mean() * 100),
        "bis_mean":  float(labels.mean()),
        "bis_std":   float(labels.std()),
        "bis_min":   float(labels.min()),
        "bis_max":   float(labels.max()),
        "mae_overall":     float(np.abs(pred - labels).mean()) if pred is not None else None,
        "mae_induction":   mae(bis_induction),
        "mae_maintenance": mae(bis_maintenance),
        "mae_deep":        mae(bis_deep),
        "pearson_r": float(np.corrcoef(pred, labels)[0, 1]) if pred is not None and len(labels) > 2 else None,
        "rmse": float(np.sqrt(((pred - labels) ** 2).mean())) if pred is not None else None,
        # V2 phase-specific stats
        "has_phases": False,
        "n_stim_events": 0,
    }

    if phases is not None:
        stats["has_phases"] = True
        phase_mae = {}
        phase_pct = {}
        for ph_id, ph_name in enumerate(_PHASE_NAMES):
            mask = (phases == ph_id)
            key = ph_name.lower().replace("-", "_")
            phase_pct[f"pct_{key}"] = float(mask.mean() * 100)
            phase_mae[f"mae_{key}"] = mae(mask)
        stats.update(phase_pct)
        stats.update(phase_mae)

    if stim is not None:
        stats["n_stim_events"] = int((stim > 0.5).sum())

    return stats


def _subsample(arr, target=4000):
    """Subsample an array to ~target points for JSON size."""
    if len(arr) <= target:
        return arr.tolist() if isinstance(arr, np.ndarray) else arr
    step = max(1, len(arr) // target)
    return arr[::step].tolist() if isinstance(arr, np.ndarray) else arr[::step]


# ─────────────────────────────────────────────────────────────────────────────
# HTML report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_html(case_id: str, data: dict, stats: dict, out_path: Path) -> None:
    print("  Generating HTML report ...")

    # ── JSON payload ──────────────────────────────────────────────────────────
    times  = data["times"]
    labels = data["labels"]
    pred   = data.get("pred_bis")
    phases = data.get("phases")     # (N,) int8 predicted phase, or None
    stim   = data.get("stim")       # (N,) float32 stim prob, or None
    sqi    = data["sqi"]            # (N, n_ch)
    feats  = data["features"]       # (N, 24)
    dsa    = data["dsa"]
    bands  = data["bands"]
    n_ch   = sqi.shape[1]

    # Subsample BIS/SQI/feature timeseries for plot (keep ≤4000 pts)
    t_sub     = _subsample(times, 4000)
    lab_sub   = _subsample(labels, 4000)
    pred_sub  = _subsample(pred, 4000) if pred is not None else None
    sqi1_sub  = _subsample(sqi[:, 0], 4000)
    sqi2_sub  = _subsample(sqi[:, 1] if n_ch > 1 else sqi[:, 0], 4000)

    # Feature trends (band powers ch1)
    feat_delta = _subsample(feats[:, 0], 4000)
    feat_theta = _subsample(feats[:, 1], 4000)
    feat_alpha = _subsample(feats[:, 2], 4000)
    feat_beta  = _subsample(feats[:, 3], 4000)
    feat_gamma = _subsample(feats[:, 4], 4000)
    feat_pe    = _subsample(feats[:, 5], 4000)

    # ── Spectral analysis: three methods to remove 1/f² physics ────────────────
    _bands_raw = np.clip(feats[:, :5].astype(np.float64), 1e-12, None)  # (N,5)
    _BAND_CENTERS = np.array([2.0, 6.0, 10.5, 21.5, 38.5])  # Hz centres
    _BAND_NAMES5  = ["delta", "theta", "alpha", "beta", "gamma"]
    N_win = _bands_raw.shape[0]

    # ── Identify baseline windows (pre-op: BIS>80 or phase==0 or first 3min) ──
    if phases is not None:
        _bl_mask = (phases == 0)   # PRE_OP predicted by model
    else:
        _bl_mask = labels > 80.0   # BIS-based fallback
    if _bl_mask.sum() < 30:        # too few → use first 3 minutes
        _bl_secs = min(180, int(times[-1])) if len(times) > 0 else 180
        _bl_mask = times <= _bl_secs
    _bl_mask = _bl_mask.astype(bool)

    _bl_mean = _bands_raw[_bl_mask].mean(axis=0) + 1e-12   # (5,) baseline mean
    _bl_std  = _bands_raw[_bl_mask].std(axis=0)  + 1e-12   # (5,) baseline std

    # ── Method 1: Baseline Fold Change ──────────────────────────────────────
    # fold_change[i, b] = current_power[b] / baseline_mean[b]
    # Reveals changes relative to patient's own awake state.
    _fold = _bands_raw / _bl_mean[np.newaxis, :]           # (N, 5)
    # Smooth each band with 30s causal MA
    def _causal_ma(arr1d, w=30):
        cs = np.cumsum(arr1d)
        out = np.empty_like(arr1d)
        for i in range(len(arr1d)):
            lo = max(0, i - w + 1)
            out[i] = (cs[i] - (cs[lo-1] if lo > 0 else 0.0)) / (i - lo + 1)
        return out
    _fold_sm = np.stack([_causal_ma(_fold[:, b]) for b in range(5)], axis=1)
    fc_delta = _subsample(_fold_sm[:, 0], 4000)
    fc_theta = _subsample(_fold_sm[:, 1], 4000)
    fc_alpha = _subsample(_fold_sm[:, 2], 4000)
    fc_beta  = _subsample(_fold_sm[:, 3], 4000)
    fc_gamma = _subsample(_fold_sm[:, 4], 4000)
    # Baseline stats for annotation
    fc_baseline = {b: float(_bl_mean[i]) for i, b in enumerate(_BAND_NAMES5)}

    # ── Method 2: FOOOF-style 1/f background removal ────────────────────────
    # Per window: fit log-linear model to log(freq) vs log(power) across 5 bands.
    # Residuals = log(power) - fitted → oscillatory component free of 1/f slope.
    _log_f  = np.log10(_BAND_CENTERS)           # (5,) fixed
    _log_p  = np.log10(_bands_raw)              # (N, 5)
    # OLS in log-log space: slope = Cov(log_f, log_p) / Var(log_f)
    _lf_c   = _log_f - _log_f.mean()
    _lp_c   = _log_p - _log_p.mean(axis=1, keepdims=True)
    _slope  = (_lp_c * _lf_c).sum(axis=1) / (_lf_c ** 2).sum()  # (N,)
    _inter  = _log_p.mean(axis=1) - _slope * _log_f.mean()       # (N,)
    # Fitted background at each band
    _fitted = _inter[:, np.newaxis] + _slope[:, np.newaxis] * _log_f  # (N, 5)
    # Residual: deviation above/below the 1/f background (in log10 units)
    _resid  = _log_p - _fitted                                    # (N, 5)
    # Smooth residuals
    _resid_sm = np.stack([_causal_ma(_resid[:, b]) for b in range(5)], axis=1)
    ff_delta = _subsample(_resid_sm[:, 0], 4000)
    ff_theta = _subsample(_resid_sm[:, 1], 4000)
    ff_alpha = _subsample(_resid_sm[:, 2], 4000)
    ff_beta  = _subsample(_resid_sm[:, 3], 4000)
    ff_gamma = _subsample(_resid_sm[:, 4], 4000)

    # ── Method 3: Dynamic Z-score per band ──────────────────────────────────
    # Z[i, b] = (power[i,b] - baseline_mean[b]) / baseline_std[b]
    # Reveals statistical significance of current state vs patient's own baseline.
    _zscore  = (_bands_raw - _bl_mean[np.newaxis, :]) / _bl_std[np.newaxis, :]
    _zscore_sm = np.stack([_causal_ma(_zscore[:, b]) for b in range(5)], axis=1)
    # For heatmap: subsample time axis, keep 5 bands
    _ZS_TARGET = 2000
    _zs_step = max(1, N_win // _ZS_TARGET)
    zs_t     = times[::_zs_step].tolist()
    zs_mat   = _zscore_sm[::_zs_step, :].T.tolist()   # (5, T_sub)

    # ── Alpha/Delta Power Ratio (ADPR) ───────────────────────────────────────
    _adpr = np.log10(_bands_raw[:, 2] / _bands_raw[:, 0])
    adpr_sub = _subsample(_causal_ma(_adpr, 30), 4000)

    # ── Raw log-normalised % (kept for backward compat display) ─────────────
    _bands_log = np.log10(_bands_raw)
    _bands_pos = _bands_log - _bands_log.min(axis=1, keepdims=True)
    _band_sum  = np.where(_bands_pos.sum(axis=1, keepdims=True) == 0, 1.0,
                          _bands_pos.sum(axis=1, keepdims=True))
    _bands_pct = _bands_pos / _band_sum * 100.0
    bp_delta = _subsample(_bands_pct[:, 0], 4000)
    bp_theta = _subsample(_bands_pct[:, 1], 4000)
    bp_alpha = _subsample(_bands_pct[:, 2], 4000)
    bp_beta  = _subsample(_bands_pct[:, 3], 4000)
    bp_gamma = _subsample(_bands_pct[:, 4], 4000)

    # Band waves -- already downsampled in compute_band_waves
    btime = bands["time"]
    # Subsample band waves to ≤8000 pts
    _bs = lambda k: _subsample(np.array(bands[k]), 8000)
    b_delta = _bs("delta_ch1")
    b_theta = _bs("theta_ch1")
    b_alpha = _bs("alpha_ch1")
    b_beta  = _bs("beta_ch1")
    b_gamma = _bs("gamma_ch1")
    bt_sub  = _subsample(np.array(btime), 8000)

    duration_min = float(times[-1] / 60) if len(times) > 0 else 0

    # Phase and stim data (v2 model only)
    phase_sub = _subsample(phases, 4000) if phases is not None else None
    stim_sub  = _subsample(stim,   4000) if stim   is not None else None

    js_data = json.dumps({
        "case_id":    case_id,
        "dur_min":    duration_min,
        "stats":      stats,
        "t":          t_sub,
        "label":      lab_sub,
        "pred":       pred_sub,
        "phases":     phase_sub,
        "stim":       stim_sub,
        "sqi1":       sqi1_sub,
        "sqi2":       sqi2_sub,
        "fp_delta":   feat_delta,
        "fp_theta":   feat_theta,
        "fp_alpha":   feat_alpha,
        "fp_beta":    feat_beta,
        "fp_gamma":   feat_gamma,
        "fp_pe":      feat_pe,
        "bp_delta":   bp_delta,
        "bp_theta":   bp_theta,
        "bp_alpha":   bp_alpha,
        "bp_beta":    bp_beta,
        "bp_gamma":   bp_gamma,
        "adpr":       adpr_sub,
        "fc_delta":   fc_delta,
        "fc_theta":   fc_theta,
        "fc_alpha":   fc_alpha,
        "fc_beta":    fc_beta,
        "fc_gamma":   fc_gamma,
        "fc_baseline": fc_baseline,
        "ff_delta":   ff_delta,
        "ff_theta":   ff_theta,
        "ff_alpha":   ff_alpha,
        "ff_beta":    ff_beta,
        "ff_gamma":   ff_gamma,
        "zs_t":       zs_t,
        "zs_mat":     zs_mat,
        "bt":         bt_sub,
        "b_delta":    b_delta,
        "b_theta":    b_theta,
        "b_alpha":    b_alpha,
        "b_beta":     b_beta,
        "b_gamma":    b_gamma,
        "dsa_times":  dsa["times"],
        "dsa_freqs":  dsa["freqs"],
        "dsa_power":  dsa["power"],
    }, ensure_ascii=False, allow_nan=False)

    # ── HTML ──────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OR Monitor -- {case_id}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #030b14;
    --panel:    #071425;
    --border:   #0d2a45;
    --accent:   #00d4ff;
    --accent2:  #00ff9d;
    --amber:    #ffb300;
    --red:      #ff3d5a;
    --muted:    #304560;
    --text:     #a8c4d8;
    --texthi:   #e0f0ff;
    --mono:     'JetBrains Mono', monospace;
    --head:     'Chakra Petch', sans-serif;
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ background:var(--bg); color:var(--text); font-family:var(--mono); min-height:100vh; }}

  /* ── Scanline overlay ─────────────────────────────── */
  body::before {{
    content:''; position:fixed; inset:0; pointer-events:none; z-index:999;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.08) 2px,rgba(0,0,0,.08) 4px);
  }}

  /* ── Header ──────────────────────────────────────── */
  .hdr {{
    border-bottom:1px solid var(--border);
    padding:16px 28px;
    display:flex; align-items:center; justify-content:space-between;
    background:linear-gradient(90deg, rgba(0,212,255,.06), transparent);
    position:sticky; top:0; z-index:100; backdrop-filter:blur(8px);
  }}
  .hdr-left {{ display:flex; align-items:center; gap:20px; }}
  .pulse-dot {{
    width:10px; height:10px; border-radius:50%; background:var(--accent2);
    box-shadow:0 0 8px var(--accent2); animation:pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.3}} }}
  .hdr-title {{ font-family:var(--head); font-size:1.1rem; font-weight:700; color:var(--texthi); letter-spacing:.08em; }}
  .hdr-sub {{ font-size:.7rem; color:var(--muted); margin-top:2px; }}
  .hdr-stats {{ display:flex; gap:24px; }}
  .hstat {{ text-align:center; }}
  .hstat-val {{ font-family:var(--head); font-size:1.4rem; font-weight:700; }}
  .hstat-lbl {{ font-size:.6rem; color:var(--muted); letter-spacing:.1em; text-transform:uppercase; }}

  /* ── Nav tabs ─────────────────────────────────────── */
  .nav {{ display:flex; gap:0; border-bottom:1px solid var(--border); background:var(--panel); padding:0 24px; position:sticky; top:57px; z-index:90; }}
  .nav-tab {{ padding:10px 18px; font-family:var(--head); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; cursor:pointer; color:var(--muted); border-bottom:2px solid transparent; transition:all .2s; user-select:none; }}
  .nav-tab:hover {{ color:var(--accent); }}
  .nav-tab.active {{ color:var(--accent); border-bottom-color:var(--accent); }}

  /* ── Sections ─────────────────────────────────────── */
  .section {{ display:none; padding:24px 24px 0; animation:fadeIn .25s; }}
  .section.active {{ display:block; }}
  @keyframes fadeIn {{ from{{opacity:0;transform:translateY(6px)}} to{{opacity:1;transform:none}} }}

  /* ── Panel card ───────────────────────────────────── */
  .card {{
    background:var(--panel); border:1px solid var(--border); border-radius:6px;
    padding:18px 20px; margin-bottom:20px;
    box-shadow:0 4px 24px rgba(0,0,0,.4);
  }}
  .card-title {{ font-family:var(--head); font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:14px; display:flex; align-items:center; gap:8px; }}
  .card-title::before {{ content:''; width:3px; height:12px; background:var(--accent); display:inline-block; border-radius:2px; }}

  /* ── DSA slider ───────────────────────────────────── */
  .dsa-wrap {{ position:relative; }}
  .slider-row {{ display:flex; align-items:center; gap:14px; margin-top:12px; }}
  .slider-row label {{ font-size:.68rem; color:var(--muted); white-space:nowrap; }}
  #dsa-slider {{
    flex:1; -webkit-appearance:none; appearance:none; height:3px;
    background:linear-gradient(90deg, var(--accent) var(--pct,0%), var(--border) var(--pct,0%));
    border-radius:2px; outline:none; cursor:pointer;
  }}
  #dsa-slider::-webkit-slider-thumb {{
    -webkit-appearance:none; width:14px; height:14px; border-radius:50%;
    background:var(--accent); box-shadow:0 0 8px var(--accent); cursor:pointer;
  }}
  #dsa-time-lbl {{ font-family:var(--head); font-size:.85rem; color:var(--accent); width:60px; text-align:right; }}

  /* ── Stats grid ───────────────────────────────────── */
  .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; }}
  .stat-card {{
    background:rgba(0,0,0,.25); border:1px solid var(--border); border-radius:5px;
    padding:14px 16px;
  }}
  .stat-card .sv {{ font-family:var(--head); font-size:1.6rem; font-weight:700; color:var(--texthi); }}
  .stat-card .sl {{ font-size:.6rem; color:var(--muted); letter-spacing:.1em; text-transform:uppercase; margin-top:4px; }}
  .stat-card .sd {{ font-size:.7rem; color:var(--text); margin-top:6px; border-top:1px solid var(--border); padding-top:6px; }}

  /* ── Phase bar ────────────────────────────────────── */
  .phase-bar {{ display:flex; height:12px; border-radius:3px; overflow:hidden; margin:12px 0 6px; }}
  .ph-0 {{ background:#6366f1; }}
  .ph-1 {{ background:#ffb300; }}
  .ph-2 {{ background:#00d4ff; }}
  .ph-3 {{ background:#00ff9d; }}
  .ph-i {{ background:#ffb300; }}
  .ph-m {{ background:#00d4ff; }}
  .ph-d {{ background:#ff3d5a; }}
  .phase-legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:.65rem; }}
  .pl-dot {{ display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:5px; }}
  /* stim event badge */
  .stim-badge {{ display:inline-block; padding:3px 10px; border-radius:3px; font-size:.68rem;
    background:rgba(255,61,90,.15); border:1px solid rgba(255,61,90,.4); color:#ff3d5a; margin-bottom:14px; }}

  /* ── BIS badge ────────────────────────────────────── */
  .bis-badges {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }}
  .bis-badge {{ padding:6px 14px; border-radius:3px; font-family:var(--head); font-size:.75rem; font-weight:600; letter-spacing:.05em; }}
  .bb-awake {{ background:rgba(255,179,0,.15); border:1px solid rgba(255,179,0,.4); color:var(--amber); }}
  .bb-adequate {{ background:rgba(0,212,255,.1); border:1px solid rgba(0,212,255,.3); color:var(--accent); }}
  .bb-deep {{ background:rgba(255,61,90,.12); border:1px solid rgba(255,61,90,.35); color:var(--red); }}

  /* ── Scrollable chart wrapper ─────────────────────── */
  .chart-scroll {{ overflow:hidden; }}
  .plot {{ width:100%; }}

  /* Responsive */
  @media (max-width:768px) {{ .hdr-stats {{ display:none; }} .stats-grid {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>

<!-- ── Header ─────────────────────────────────────────────────────── -->
<header class="hdr">
  <div class="hdr-left">
    <div class="pulse-dot"></div>
    <div>
      <div class="hdr-title">ANESTHESIA DEPTH MONITOR &nbsp;/&nbsp; {case_id}</div>
      <div class="hdr-sub">AnesthesiaNet &middot; GRU-LNN &middot; Multi-task v2 &middot; Inference Report</div>
    </div>
  </div>
  <div class="hdr-stats" id="hdr-stats"></div>
</header>

<!-- ── Nav ────────────────────────────────────────────────────────── -->
<nav class="nav" id="nav">
  <div class="nav-tab active" data-sec="sec-bis">BIS Timeline</div>
  <div class="nav-tab" data-sec="sec-phases">Phases</div>
  <div class="nav-tab" data-sec="sec-dsa">DSA</div>
  <div class="nav-tab" data-sec="sec-bands">EEG Bands</div>
  <div class="nav-tab" data-sec="sec-sqi">SQI</div>
  <div class="nav-tab" data-sec="sec-features">Features</div>
  <div class="nav-tab" data-sec="sec-analysis">Analysis</div>
</nav>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- Section 1: BIS Timeline                                          -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<section class="section active" id="sec-bis">
  <div class="card">
    <div class="card-title">BIS -- Predicted vs Ground Truth</div>
    <div id="bis-badges" class="bis-badges"></div>
    <div id="plot-bis" class="plot" style="height:340px"></div>
  </div>
</section>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- Section 2: Surgical Phases (v2 model)                            -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<section class="section" id="sec-phases">
  <div class="card">
    <div class="card-title">Predicted Surgical Phase Timeline</div>
    <div id="stim-badge-wrap"></div>
    <div id="plot-phase-timeline" class="plot" style="height:260px"></div>
  </div>
  <div class="card">
    <div class="card-title">BIS with Phase Background + Stimulation Events</div>
    <div id="plot-bis-phases" class="plot" style="height:340px"></div>
  </div>
  <div class="card">
    <div class="card-title">Stimulation Event Probability</div>
    <div id="plot-stim" class="plot" style="height:220px"></div>
  </div>
</section>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- Section 3: DSA                                                   -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<section class="section" id="sec-dsa">
  <div class="card">
    <div class="card-title">Density Spectral Array (Fp1 -- full recording)</div>
    <div id="plot-dsa" class="plot" style="height:320px"></div>
    <div class="slider-row">
      <label>CURSOR</label>
      <input type="range" id="dsa-slider" min="0" max="100" value="0">
      <span id="dsa-time-lbl">0 s</span>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Frequency Spectrum at Cursor</div>
    <div id="plot-spectrum" class="plot" style="height:260px"></div>
  </div>
</section>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- Section 3: EEG Bands                                             -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<section class="section" id="sec-bands">
  <div class="card">
    <div class="card-title">EEG Band Decomposition (Fp1, downsampled)</div>
    <div id="plot-bands" class="plot" style="height:520px"></div>
  </div>
</section>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- Section 4: SQI                                                   -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<section class="section" id="sec-sqi">
  <div class="card">
    <div class="card-title">Signal Quality Index -- Pipeline Output (not BIS monitor SQI)</div>
    <div id="plot-sqi" class="plot" style="height:280px"></div>
  </div>
</section>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- Section 5: Features                                              -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<section class="section" id="sec-features">
  <div class="card">
    <div class="card-title">Relative Band Powers (Fp1) + Permutation Entropy</div>
    <div id="plot-features" class="plot" style="height:360px"></div>
  </div>
  <div class="card">
    <div class="card-title">Method 1 -- Baseline Fold Change (relative to pre-op awake state)</div>
    <div id="fc-alert" style="margin-bottom:10px"></div>
    <div id="plot-fold-change" class="plot" style="height:320px"></div>
  </div>
  <div class="card">
    <div class="card-title">Method 2 -- FOOOF: 1/f Background Removed (oscillatory component only)</div>
    <div id="plot-fooof" class="plot" style="height:280px"></div>
  </div>
  <div class="card">
    <div class="card-title">Method 3 -- Dynamic Z-score Heatmap (deviation from baseline, per band)</div>
    <div id="plot-zscore" class="plot" style="height:260px"></div>
  </div>
  <div class="card">
    <div class="card-title">Alpha/Delta Power Ratio (ADPR) -- Standard Clinical Indicator</div>
    <div id="plot-adpr" class="plot" style="height:220px"></div>
  </div>
  <div class="card">
    <div class="card-title">Reference: Raw Log-normalised Band % (1/f corrected stacked area)</div>
    <div id="plot-band-pct" class="plot" style="height:320px"></div>
  </div>
</section>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- Section 6: Analysis                                              -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<section class="section" id="sec-analysis">
  <div class="card">
    <div class="card-title">Surgery Phase Distribution</div>
    <div id="phase-bar-wrap"></div>
    <div id="plot-phase" class="plot" style="height:280px"></div>
  </div>
  <div class="card">
    <div class="card-title">Prediction Accuracy</div>
    <div id="plot-scatter" class="plot" style="height:320px"></div>
  </div>
  <div class="card">
    <div class="card-title">Key Metrics</div>
    <div id="stats-grid" class="stats-grid"></div>
  </div>
</section>

<!-- ══════════════════════════════════════════════════════════════════ -->
<script>
// ── Embedded data ────────────────────────────────────────────────────
const D = {js_data};

// ── Plotly shared theme ──────────────────────────────────────────────
const BG  = '#030b14';
const PAN = '#071425';
const GRD = '#0d2a45';
const TXT = '#a8c4d8';
const TXTH = '#e0f0ff';
const CYAN = '#00d4ff';
const GREEN= '#00ff9d';
const AMBE = '#ffb300';
const RED  = '#ff3d5a';
const MUTE = '#304560';

const LAYOUT_BASE = {{
  paper_bgcolor: PAN,
  plot_bgcolor:  PAN,
  font:  {{ family:"'JetBrains Mono',monospace", size:11, color:TXT }},
  xaxis: {{ gridcolor:GRD, linecolor:GRD, zerolinecolor:GRD, tickfont:{{size:10}} }},
  yaxis: {{ gridcolor:GRD, linecolor:GRD, zerolinecolor:GRD, tickfont:{{size:10}} }},
  margin: {{ t:20, b:50, l:55, r:20 }},
  legend: {{ bgcolor:'rgba(0,0,0,0)', font:{{size:10}} }},
  hovermode: 'x unified',
}};

function layout(extra={{}}) {{ return Object.assign({{}}, LAYOUT_BASE, extra); }}

// ── BIS phase background shapes ─────────────────────────────────────
function bisShapes(tArr, lArr) {{
  if (!tArr || !lArr) return [];
  const shapes = [];
  const tMin = tArr[0], tMax = tArr[tArr.length-1];
  // Coloured zones
  shapes.push({{ type:'rect', xref:'x', yref:'paper', x0:tMin, x1:tMax, y0:0, y1:1,
    fillcolor:'rgba(0,212,255,0.04)', line:{{width:0}}, layer:'below' }});
  // Reference lines
  [40,60].forEach(y => shapes.push({{
    type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:y, y1:y,
    line:{{color:MUTE, width:1, dash:'dot'}}
  }}));
  return shapes;
}}

// ── Header stats ─────────────────────────────────────────────────────
function buildHeader() {{
  const s = D.stats;
  const el = document.getElementById('hdr-stats');
  const items = [
    ['BIS','mean',  s.bis_mean?.toFixed(1) ?? '--', CYAN],
    ['DURATION', 'min', D.dur_min?.toFixed(0) ?? '--', GREEN],
    ['WINDOWS', 'n', s.n_total?.toLocaleString() ?? '--', TXTH],
  ];
  if (s.mae_overall != null)
    items.push(['MAE', 'BIS pts', s.mae_overall?.toFixed(2) ?? '--', AMBE]);

  el.innerHTML = items.map(([l,u,v,c]) =>
    `<div class="hstat">
       <div class="hstat-val" style="color:${{c}}">${{v}}</div>
       <div class="hstat-lbl">${{l}} / ${{u}}</div>
     </div>`).join('');
}}

// ── Phase helpers ────────────────────────────────────────────────────
const PHASE_COLORS = ['#6366f1','#ffb300','#00d4ff','#00ff9d'];
const PHASE_NAMES  = ['Pre-Op','Induction','Maintenance','Recovery'];
const PHASE_ALPHA  = [0.10, 0.15, 0.07, 0.13];

// Convert hex color + alpha to rgba string
function hexToRgba(hex, a) {{
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

function phaseShapes2(tArr, phArr) {{
  if (!phArr || !tArr) return [];
  const shapes = [];
  let cur = phArr[0], t0 = tArr[0];
  for (let i = 1; i <= tArr.length-1; i++) {{
    if (phArr[i] !== cur) {{
      shapes.push({{
        type:'rect', xref:'x', yref:'paper',
        x0:t0, x1:tArr[i], y0:0, y1:1,
        fillcolor: hexToRgba(PHASE_COLORS[cur] || PHASE_COLORS[2], PHASE_ALPHA[cur] ?? 0.08),
        line:{{width:0}}, layer:'below',
      }});
      cur = phArr[i]; t0 = tArr[i];
    }}
  }}
  // last span
  shapes.push({{
    type:'rect', xref:'x', yref:'paper',
    x0:t0, x1:tArr[tArr.length-1], y0:0, y1:1,
    fillcolor: hexToRgba(PHASE_COLORS[cur] || PHASE_COLORS[2], PHASE_ALPHA[cur] ?? 0.08),
    line:{{width:0}}, layer:'below',
  }});
  return shapes;
}}

// ── BIS Timeline ─────────────────────────────────────────────────────
function buildBIS() {{
  const t = D.t, lbl = D.label, prd = D.pred;
  const traces = [];

  traces.push({{ x:t, y:lbl, mode:'lines', name:'True BIS',
    line:{{color:AMBE, width:1.8}}, opacity:0.9 }});

  if (prd)
    traces.push({{ x:t, y:prd, mode:'lines', name:'Predicted BIS',
      line:{{color:CYAN, width:2}}, opacity:0.95 }});

  // Stimulation events as vertical markers
  if (D.stim && D.pred) {{
    const stim_t = [], stim_v = [];
    D.stim.forEach((s,i) => {{ if(s > 0.5) {{ stim_t.push(D.t[i]); stim_v.push(D.pred[i] ?? D.label[i]); }} }});
    if (stim_t.length > 0)
      traces.push({{ x:stim_t, y:stim_v, mode:'markers', name:'Stimulation',
        marker:{{color:RED, size:7, symbol:'triangle-up', line:{{color:'#ff3d5a',width:1}}}},
        hovertemplate:'Stim @ %{{x:.0f}}s<extra></extra>' }});
  }}

  const s = D.stats;
  const badges = document.getElementById('bis-badges');
  if (s.has_phases) {{
    badges.innerHTML = `
      <div class="bis-badge" style="background:rgba(99,102,241,.15);border-color:rgba(99,102,241,.4);color:#6366f1">PRE-OP&nbsp;&nbsp;${{s.pct_pre_op?.toFixed(0)??0}}%</div>
      <div class="bis-badge bb-awake">INDUCTION&nbsp;&nbsp;${{s.pct_induction?.toFixed(0)??0}}%</div>
      <div class="bis-badge bb-adequate">MAINTENANCE&nbsp;&nbsp;${{s.pct_maintenance?.toFixed(0)??0}}%</div>
      <div class="bis-badge" style="background:rgba(0,255,157,.1);border-color:rgba(0,255,157,.3);color:#00ff9d">RECOVERY&nbsp;&nbsp;${{s.pct_recovery?.toFixed(0)??0}}%</div>
    `;
  }} else {{
    badges.innerHTML = `
      <div class="bis-badge bb-awake">INDUCTION&nbsp;&nbsp;${{s.pct_induction?.toFixed(0)}}%</div>
      <div class="bis-badge bb-adequate">MAINTENANCE&nbsp;&nbsp;${{s.pct_maintenance?.toFixed(0)}}%</div>
      <div class="bis-badge bb-deep">DEEP&nbsp;&nbsp;${{s.pct_deep?.toFixed(0)}}%</div>
    `;
  }}

  const extraShapes = D.phases ? phaseShapes2(t, D.phases) : bisShapes(t, lbl);

  Plotly.newPlot('plot-bis', traces, layout({{
    xaxis:{{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis:{{ title:'BIS', range:[0,105], ...LAYOUT_BASE.yaxis }},
    shapes: [...extraShapes,
      {{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:60,y1:60,line:{{color:MUTE,width:1,dash:'dot'}}}},
      {{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:40,y1:40,line:{{color:MUTE,width:1,dash:'dot'}}}},
    ],
    annotations:[
      {{xref:'paper',yref:'y',x:0,y:60,text:'60',showarrow:false,font:{{size:9,color:MUTE}},xanchor:'left'}},
      {{xref:'paper',yref:'y',x:0,y:40,text:'40',showarrow:false,font:{{size:9,color:MUTE}},xanchor:'left'}},
    ],
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── Phases Section ───────────────────────────────────────────────────
function buildPhases() {{
  const t = D.t;
  const ph = D.phases;
  const stim = D.stim;

  if (!ph) {{
    ['plot-phase-timeline','plot-bis-phases','plot-stim'].forEach(id => {{
      document.getElementById(id).innerHTML =
        '<p style="text-align:center;color:#304560;padding:40px;font-size:.8rem">Phase predictions require v2 model checkpoint</p>';
    }});
    return;
  }}

  // Stimulation event count badge
  const s = D.stats;
  document.getElementById('stim-badge-wrap').innerHTML =
    `<span class="stim-badge">STIMULATION EVENTS DETECTED: ${{s.n_stim_events ?? 0}}</span>`;

  // ── Phase timeline (scatter with color by phase) ──────────────────
  const phTraces = PHASE_NAMES.map((name, pid) => {{
    const mask_t = t.filter((_,i) => ph[i] === pid);
    const mask_y = t.map((_,i) => ph[i] === pid ? pid : null);
    return {{
      x: t, y: t.map((_,i) => ph[i] === pid ? pid : null),
      mode:'markers', name:name,
      marker:{{color:PHASE_COLORS[pid], size:4, symbol:'square'}},
      hovertemplate:name+' @ %{{x:.0f}}s<extra></extra>',
      connectgaps:false,
    }};
  }});

  Plotly.newPlot('plot-phase-timeline', phTraces, layout({{
    xaxis:{{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis:{{ tickvals:[0,1,2,3], ticktext:PHASE_NAMES, range:[-0.5,3.5], ...LAYOUT_BASE.yaxis }},
    showlegend:true,
    legend:{{ x:1.01, y:0.5, xanchor:'left' }},
    margin:{{...LAYOUT_BASE.margin, r:10}},
  }}), {{responsive:true, displayModeBar:false}});

  // ── BIS + phase background + stim markers ────────────────────────
  const bisTraces = [
    {{ x:t, y:D.label, mode:'lines', name:'True BIS', line:{{color:AMBE, width:1.8}}, opacity:0.9 }},
  ];
  if (D.pred)
    bisTraces.push({{ x:t, y:D.pred, mode:'lines', name:'Predicted BIS', line:{{color:CYAN,width:2}}, opacity:0.95 }});
  if (stim) {{
    const st_t=[], st_v=[];
    stim.forEach((sv,i) => {{ if(sv>0.5){{st_t.push(t[i]);st_v.push(D.pred?D.pred[i]:D.label[i]);}} }});
    if(st_t.length)
      bisTraces.push({{ x:st_t, y:st_v, mode:'markers', name:'Stimulation',
        marker:{{color:RED, size:8, symbol:'triangle-up'}},
        hovertemplate:'Stim @ %{{x:.0f}}s<extra></extra>' }});
  }}

  Plotly.newPlot('plot-bis-phases', bisTraces, layout({{
    xaxis:{{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis:{{ title:'BIS', range:[0,105], ...LAYOUT_BASE.yaxis }},
    shapes:[
      ...phaseShapes2(t, ph),
      {{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:60,y1:60,line:{{color:MUTE,width:1,dash:'dot'}}}},
      {{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:40,y1:40,line:{{color:MUTE,width:1,dash:'dot'}}}},
    ],
  }}), {{responsive:true, displayModeBar:false}});

  // ── Stimulation probability ───────────────────────────────────────
  if (stim) {{
    Plotly.newPlot('plot-stim', [
      {{ x:t, y:stim, mode:'lines', name:'Stim Prob',
         line:{{color:RED, width:1.5}}, fill:'tozeroy', fillcolor:'rgba(255,61,90,0.08)' }},
      {{ x:t, y:t.map(()=>0.5), mode:'lines', name:'threshold',
         line:{{color:MUTE, width:1, dash:'dash'}}, showlegend:false }},
    ], layout({{
      xaxis:{{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
      yaxis:{{ title:'P(stim)', range:[-0.02,1.05], ...LAYOUT_BASE.yaxis }},
    }}), {{responsive:true, displayModeBar:false}});
  }} else {{
    document.getElementById('plot-stim').innerHTML =
      '<p style="text-align:center;color:#304560;padding:30px;font-size:.8rem">No stimulation data</p>';
  }}
}}

// ── DSA ──────────────────────────────────────────────────────────────
let dsaCursorT = D.dsa_times[0] ?? 0;

function buildDSA() {{
  const t  = D.dsa_times;
  const f  = D.dsa_freqs;
  const z  = D.dsa_power;    // [freq_idx][time_idx]

  // Transpose to [time][freq] for Plotly (z[i][j] = power at time i, freq j)
  const zT = t.map((_,ti) => f.map((_,fi) => z[fi][ti]));

  const heatTrace = [{{
    type:'heatmap', x:t, y:f, z:zT,
    colorscale:'Inferno',
    zmin:-30, zmax:15,
    colorbar:{{ title:'dB', thickness:10, len:0.8,
      tickfont:{{size:9,color:TXT}}, titlefont:{{size:9,color:TXT}} }},
    hovertemplate:'Time: %{{x:.0f}}s<br>Freq: %{{y:.1f}}Hz<br>Power: %{{z:.1f}}dB<extra></extra>',
  }}];

  // Cursor vertical line
  const cursorShape = () => [{{
    type:'line', xref:'x', yref:'paper',
    x0:dsaCursorT, x1:dsaCursorT, y0:0, y1:1,
    line:{{color:CYAN, width:2}},
  }}];

  Plotly.newPlot('plot-dsa', heatTrace, layout({{
    xaxis:{{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis:{{ title:'Frequency (Hz)', ...LAYOUT_BASE.yaxis }},
    shapes: cursorShape(),
    margin:{{...LAYOUT_BASE.margin, r:80}},
  }}), {{responsive:true, displayModeBar:false}});

  // Click on DSA to move cursor
  document.getElementById('plot-dsa').on('plotly_click', ev => {{
    if (!ev.points.length) return;
    dsaCursorT = ev.points[0].x;
    updateDSACursor();
  }});

  buildSpectrum();
  buildDSASlider();
}}

function updateDSACursor() {{
  const t = D.dsa_times;
  const slider = document.getElementById('dsa-slider');
  const pct = ((dsaCursorT - t[0]) / (t[t.length-1] - t[0])) * 100;
  slider.value = pct;
  slider.style.setProperty('--pct', pct + '%');
  document.getElementById('dsa-time-lbl').textContent = Math.round(dsaCursorT) + 's';
  Plotly.relayout('plot-dsa', {{ shapes: [{{
    type:'line', xref:'x', yref:'paper',
    x0:dsaCursorT, x1:dsaCursorT, y0:0, y1:1,
    line:{{color:CYAN, width:2}},
  }}]}});
  updateSpectrum();
}}

function buildSpectrum() {{
  updateSpectrum();
}}

function updateSpectrum() {{
  const t  = D.dsa_times;
  const f  = D.dsa_freqs;
  const z  = D.dsa_power;
  // Find nearest time index
  let ti = 0;
  let minD = Infinity;
  t.forEach((tv,i) => {{ const d=Math.abs(tv-dsaCursorT); if(d<minD){{minD=d;ti=i;}} }});
  const spec = f.map((_,fi) => z[fi][ti]);

  Plotly.react('plot-spectrum', [{{
    x:f, y:spec, type:'scatter', mode:'lines', fill:'tozeroy',
    line:{{color:CYAN, width:2}},
    fillcolor:'rgba(0,212,255,0.08)',
    hovertemplate:'%{{x:.1f}}Hz: %{{y:.1f}}dB<extra></extra>',
  }}], layout({{
    xaxis:{{ title:'Frequency (Hz)', ...LAYOUT_BASE.xaxis }},
    yaxis:{{ title:'Power (dB)', ...LAYOUT_BASE.yaxis }},
    annotations:[{{
      xref:'paper',yref:'paper',x:0.01,y:0.97,
      text:`t = ${{Math.round(dsaCursorT)}}s`,
      showarrow:false, font:{{size:11,color:AMBE}}, xanchor:'left', yanchor:'top',
    }}],
  }}), {{responsive:true, displayModeBar:false}});
}}

function buildDSASlider() {{
  const slider = document.getElementById('dsa-slider');
  const t = D.dsa_times;
  slider.addEventListener('input', () => {{
    const pct = Number(slider.value);
    slider.style.setProperty('--pct', pct + '%');
    dsaCursorT = t[0] + (pct / 100) * (t[t.length-1] - t[0]);
    document.getElementById('dsa-time-lbl').textContent = Math.round(dsaCursorT) + 's';
    updateDSACursor();
  }});
}}

// ── EEG Band Decomposition ───────────────────────────────────────────
function buildBands() {{
  const t = D.bt;
  const COLORS = {{ delta:CYAN, theta:GREEN, alpha:AMBE, beta:'#c084fc', gamma:RED }};
  const OFFSETS = {{ delta:0, theta:3, alpha:6, beta:9, gamma:12 }};
  const LABELS  = {{ delta:'δ Delta (0.5–4Hz)', theta:'θ Theta (4–8Hz)',
                     alpha:'α Alpha (8–13Hz)',  beta:'β Beta (13–30Hz)',
                     gamma:'γ Gamma (30–47Hz)' }};

  const bands = ['gamma','beta','alpha','theta','delta'];
  const traces = [];
  bands.forEach((b,bi) => {{
    const raw = D[`b_${{b}}`];
    const off  = OFFSETS[b];
    const y    = raw.map(v => v + off);
    traces.push({{
      x:t, y, mode:'lines', name:LABELS[b],
      line:{{color:COLORS[b], width:1.2}},
      hovertemplate:`${{b}}: %{{customdata:.3f}}<br>t=%{{x:.0f}}s<extra></extra>`,
      customdata:raw,
    }});
  }});

  const tickvals = Object.values(OFFSETS);
  const ticktext = bands.slice().reverse().map(b => LABELS[b]);

  Plotly.newPlot('plot-bands', traces, layout({{
    xaxis:{{ title:'Time (s)', ...LAYOUT_BASE.xaxis,
             rangeslider:{{bgcolor:PAN, bordercolor:GRD}} }},
    yaxis:{{ tickvals, ticktext, gridcolor:GRD, linecolor:GRD, zerolinecolor:GRD }},
    legend:{{ x:1.01, y:0.5, xanchor:'left' }},
    margin:{{...LAYOUT_BASE.margin, r:10}},
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── SQI Timeline ─────────────────────────────────────────────────────
function buildSQI() {{
  const traces = [
    {{ x:D.t, y:D.sqi1, mode:'lines', name:'SQI Fp1',
       line:{{color:CYAN, width:1.8}}, fill:'tozeroy', fillcolor:'rgba(0,212,255,0.06)' }},
    {{ x:D.t, y:D.sqi2, mode:'lines', name:'SQI Fp2',
       line:{{color:GREEN, width:1.8}}, fill:'tozeroy', fillcolor:'rgba(0,255,157,0.05)' }},
  ];

  Plotly.newPlot('plot-sqi', traces, layout({{
    xaxis:{{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis:{{ title:'SQI', range:[-0.05,1.1], ...LAYOUT_BASE.yaxis }},
    shapes:[{{
      type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:0.5, y1:0.5,
      line:{{color:RED, width:1, dash:'dash'}}
    }}],
    annotations:[{{xref:'paper',yref:'y',x:0,y:0.5,text:'threshold 0.5',
      showarrow:false,font:{{size:9,color:RED}},xanchor:'left',yanchor:'bottom'}}],
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── Feature Trends ───────────────────────────────────────────────────
function buildFeatures() {{
  const COLS = [CYAN,'#c084fc',GREEN,AMBE,RED,'#94a3b8'];
  const names = ['δ Delta','θ Theta','α Alpha','β Beta','γ Gamma','PE'];
  const keys  = ['fp_delta','fp_theta','fp_alpha','fp_beta','fp_gamma','fp_pe'];
  const traces = keys.map((k,i) => ({{
    x:D.t, y:D[k], mode:'lines', name:names[i],
    line:{{color:COLS[i], width:1.5}}, opacity:0.9,
  }}));

  Plotly.newPlot('plot-features', traces, layout({{
    xaxis:{{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis:{{ title:'Relative Power / PE', range:[-0.02,1.02], ...LAYOUT_BASE.yaxis }},
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── Shared band constants ─────────────────────────────────────────────
const B_COLS  = [CYAN, '#c084fc', GREEN, AMBE, RED];
const B_NAMES = ['delta (0.5-4Hz)','theta (4-8Hz)','alpha (8-13Hz)','beta (13-30Hz)','gamma (30-47Hz)'];
const B_GRK   = ['\u03b4','\u03b8','\u03b1','\u03b2','\u03b3'];
const B_KEYS_FC = ['fc_delta','fc_theta','fc_alpha','fc_beta','fc_gamma'];
const B_KEYS_FF = ['ff_delta','ff_theta','ff_alpha','ff_beta','ff_gamma'];
const B_KEYS_BP = ['bp_delta','bp_theta','bp_alpha','bp_beta','bp_gamma'];

// ── Method 1: Baseline Fold Change ───────────────────────────────────
function buildFoldChange() {{
  const traces = B_KEYS_FC.map((k, i) => ({{
    x: D.t, y: D[k],
    name: B_GRK[i] + ' ' + B_NAMES[i],
    mode: 'lines',
    line: {{ color:B_COLS[i], width: i===2 ? 2.5 : 1.5 }},
    opacity: i===2 ? 1.0 : 0.75,
    hovertemplate: B_GRK[i]+': %{{y:.2f}}x baseline<br>t=%{{x:.0f}}s<extra></extra>',
  }}));

  // Auto-alert: detect alpha > 2x baseline
  const alpha_fc = D.fc_alpha;
  let alert_t = null, peak_fc = 0;
  if (alpha_fc) {{
    alpha_fc.forEach((v, i) => {{
      if (v > peak_fc) {{ peak_fc = v; alert_t = D.t[i]; }}
    }});
  }}
  const alertEl = document.getElementById('fc-alert');
  if (peak_fc > 2.0) {{
    alertEl.innerHTML = `<div class="stim-badge" style="color:#00ff9d;border-color:rgba(0,255,157,.4);background:rgba(0,255,157,.1)">
      ALERT: alpha peak = ${{peak_fc.toFixed(2)}}x baseline at t=${{Math.round(alert_t)}}s
      -- Typical propofol anesthesia induction signature detected
    </div>`;
  }} else if (peak_fc > 1.5) {{
    alertEl.innerHTML = `<div class="stim-badge">alpha peak = ${{peak_fc.toFixed(2)}}x baseline</div>`;
  }}

  const shapes = [
    {{ type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:1, y1:1,
       line:{{color:MUTE, width:1, dash:'dash'}} }},
    {{ type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:2, y1:2,
       line:{{color:'#00ff9d', width:1, dash:'dot'}} }},
  ];

  Plotly.newPlot('plot-fold-change', traces, layout({{
    xaxis: {{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis: {{ title:'Fold Change vs Baseline', ...LAYOUT_BASE.yaxis,
              type:'log', tickformat:'.1f' }},
    hovermode: 'x unified',
    legend: {{ x:1.01, y:0.5, xanchor:'left' }},
    margin: {{...LAYOUT_BASE.margin, r:10}},
    shapes,
    annotations: [
      {{ xref:'paper', yref:'y', x:0, y:1, text:'baseline (1x)',
         showarrow:false, font:{{size:9,color:MUTE}}, xanchor:'left', yanchor:'bottom' }},
      {{ xref:'paper', yref:'y', x:0, y:2, text:'2x threshold',
         showarrow:false, font:{{size:9,color:GREEN}}, xanchor:'left', yanchor:'bottom' }},
    ],
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── Method 2: FOOOF oscillatory residuals ────────────────────────────
function buildFOOOF() {{
  const traces = B_KEYS_FF.map((k, i) => ({{
    x: D.t, y: D[k],
    name: B_GRK[i] + ' ' + B_NAMES[i],
    mode: 'lines',
    line: {{ color:B_COLS[i], width: i===2 ? 2.5 : 1.5 }},
    opacity: i===2 ? 1.0 : 0.72,
    fill: i===2 ? 'tozeroy' : 'none',
    fillcolor: i===2 ? 'rgba(0,255,157,0.08)' : undefined,
    hovertemplate: B_GRK[i]+': %{{y:.3f}} log-residual<br>t=%{{x:.0f}}s<extra></extra>',
  }}));

  Plotly.newPlot('plot-fooof', traces, layout({{
    xaxis: {{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis: {{ title:'Oscillatory Power (log10 residual)', ...LAYOUT_BASE.yaxis }},
    hovermode: 'x unified',
    legend: {{ x:1.01, y:0.5, xanchor:'left' }},
    margin: {{...LAYOUT_BASE.margin, r:10}},
    shapes: [
      {{ type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:0, y1:0,
         line:{{color:MUTE, width:1, dash:'dash'}} }},
    ],
    annotations: [
      {{ xref:'paper', yref:'paper', x:0.01, y:0.97,
         text:'Above 0: band stronger than 1/f background | Below 0: suppressed',
         showarrow:false, font:{{size:9,color:MUTE}}, xanchor:'left', yanchor:'top' }},
    ],
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── Method 3: Z-score heatmap ────────────────────────────────────────
function buildZscore() {{
  const bandLabels = B_GRK.map((g,i) => g+' '+B_NAMES[i].split(' ')[0]);
  const zMat = D.zs_mat;   // (5, T_sub)

  // Clamp display range to [-4, 4] sigma
  const ZMIN = -4, ZMAX = 4;
  const zClamped = zMat.map(row => row.map(v => Math.max(ZMIN, Math.min(ZMAX, v))));

  Plotly.newPlot('plot-zscore', [{{
    type: 'heatmap',
    x: D.zs_t,
    y: bandLabels,
    z: zClamped,
    colorscale: [
      [0,    '#00d4ff'],   // deep negative = delta dominant (deep anesthesia)
      [0.35, '#071425'],   // neutral
      [0.5,  '#1a1a2e'],   // baseline (z=0)
      [0.65, '#071425'],
      [1,    '#ff3d5a'],   // deep positive = band far above baseline
    ],
    zmin: ZMIN, zmax: ZMAX,
    colorbar: {{
      title: 'Z-score', thickness:10, len:0.8,
      tickfont:{{size:9,color:TXT}}, titlefont:{{size:9,color:TXT}},
    }},
    hovertemplate: '%{{y}}<br>t=%{{x:.0f}}s<br>Z=%{{z:.2f}}<extra></extra>',
  }}], layout({{
    xaxis: {{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis: {{ ...LAYOUT_BASE.yaxis, autorange:'reversed' }},
    margin: {{...LAYOUT_BASE.margin, r:80}},
    annotations: [
      {{ xref:'paper', yref:'paper', x:0.01, y:0.03,
         text:'Red = significantly above baseline | Blue = suppressed',
         showarrow:false, font:{{size:9,color:MUTE}}, xanchor:'left', yanchor:'bottom' }},
    ],
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── ADPR timeline ────────────────────────────────────────────────────
function buildADPR() {{
  const adpr = D.adpr;
  if (!adpr) return;
  const adprMin = Math.min.apply(null, adpr) - 0.05;
  const adprMax = Math.max.apply(null, adpr) + 0.05;

  Plotly.newPlot('plot-adpr', [
    {{ x:D.t, y:adpr, mode:'lines', name:'log ADPR (30s MA)',
       line:{{color:GREEN, width:2}},
       hovertemplate:'log(\u03b1/\u03b4): %{{y:.3f}}<br>t=%{{x:.0f}}s<extra></extra>' }},
  ], layout({{
    xaxis: {{ title:'Time (s)', ...LAYOUT_BASE.xaxis }},
    yaxis: {{ title:'log\u2081\u2080 (\u03b1/\u03b4)', ...LAYOUT_BASE.yaxis }},
    shapes: [
      {{ type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:0, y1:0,
         line:{{color:MUTE, width:1, dash:'dash'}} }},
      {{ type:'rect', xref:'paper', yref:'y', x0:0, x1:1, y0:0, y1:adprMax,
         fillcolor:'rgba(0,255,157,0.04)', line:{{width:0}}, layer:'below' }},
      {{ type:'rect', xref:'paper', yref:'y', x0:0, x1:1, y0:adprMin, y1:0,
         fillcolor:'rgba(0,212,255,0.04)', line:{{width:0}}, layer:'below' }},
    ],
    annotations: [
      {{ xref:'paper', yref:'paper', x:0.01, y:0.97,
         text:'\u03b1 > \u03b4 (lighter / propofol signature)',
         showarrow:false, font:{{size:9,color:GREEN}}, xanchor:'left', yanchor:'top' }},
      {{ xref:'paper', yref:'paper', x:0.01, y:0.06,
         text:'\u03b4 > \u03b1 (deeper anesthesia / slow-wave)',
         showarrow:false, font:{{size:9,color:CYAN}}, xanchor:'left', yanchor:'bottom' }},
    ],
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── Reference: Raw log-normalised stacked area ────────────────────────
function buildBandPct() {{
  const traces = B_KEYS_BP.map((k, i) => ({{
    x: D.t, y: D[k],
    name: B_GRK[i] + ' ' + B_NAMES[i],
    mode: 'lines',
    stackgroup: 'one',
    groupnorm: 'percent',
    fillcolor: hexToRgba(B_COLS[i], 0.60),
    line: {{ color: B_COLS[i], width: 0.5 }},
    hovertemplate: B_GRK[i]+': %{{y:.1f}}%<br>t=%{{x:.0f}}s<extra></extra>',
  }}));

  Plotly.newPlot('plot-band-pct', traces, layout({{
    xaxis: {{ title:'Time (s)', ...LAYOUT_BASE.xaxis,
              rangeslider:{{bgcolor:PAN, bordercolor:GRD}} }},
    yaxis: {{ title:'Band Power (log-norm %)', range:[0,100],
              ticksuffix:'%', ...LAYOUT_BASE.yaxis }},
    hovermode: 'x unified',
    legend: {{ x:1.01, y:0.5, xanchor:'left' }},
    margin: {{...LAYOUT_BASE.margin, r:10}},
    annotations: [{{
      xref:'paper', yref:'paper', x:0.01, y:0.97,
      text:'log10 transform applied before normalisation -- corrects 1/f\u00b2 physics',
      showarrow:false, font:{{size:9,color:MUTE}}, xanchor:'left', yanchor:'top',
    }}],
  }}), {{responsive:true, displayModeBar:false}});
}}

// ── Analysis Section ──────────────────────────────────────────────────
function buildAnalysis() {{
  const s = D.stats;

  // Phase distribution bar
  const phWrap = document.getElementById('phase-bar-wrap');
  if (s.has_phases) {{
    const tp = (s.pct_pre_op??0).toFixed(0);
    const ti = (s.pct_induction??0).toFixed(0);
    const tm = (s.pct_maintenance??0).toFixed(0);
    const tr = (s.pct_recovery??0).toFixed(0);
    phWrap.innerHTML = `
      <div class="phase-bar">
        <div class="ph-0" style="width:${{tp}}%" title="Pre-Op ${{tp}}%"></div>
        <div class="ph-1" style="width:${{ti}}%" title="Induction ${{ti}}%"></div>
        <div class="ph-2" style="width:${{tm}}%" title="Maintenance ${{tm}}%"></div>
        <div class="ph-3" style="width:${{tr}}%" title="Recovery ${{tr}}%"></div>
      </div>
      <div class="phase-legend">
        <span><span class="pl-dot" style="background:#6366f1"></span>Pre-Op ${{tp}}%</span>
        <span><span class="pl-dot" style="background:#ffb300"></span>Induction ${{ti}}%</span>
        <span><span class="pl-dot" style="background:#00d4ff"></span>Maintenance ${{tm}}%</span>
        <span><span class="pl-dot" style="background:#00ff9d"></span>Recovery ${{tr}}%</span>
      </div>`;
  }} else {{
    const ti = s.pct_induction?.toFixed(0) ?? 0;
    const tm = s.pct_maintenance?.toFixed(0) ?? 0;
    const td = s.pct_deep?.toFixed(0) ?? 0;
    phWrap.innerHTML = `
      <div class="phase-bar">
        <div class="ph-i" style="width:${{ti}}%"></div>
        <div class="ph-m" style="width:${{tm}}%"></div>
        <div class="ph-d" style="width:${{td}}%"></div>
      </div>
      <div class="phase-legend">
        <span><span class="pl-dot" style="background:#ffb300"></span>Induction ${{ti}}%</span>
        <span><span class="pl-dot" style="background:#00d4ff"></span>Maintenance ${{tm}}%</span>
        <span><span class="pl-dot" style="background:#ff3d5a"></span>Deep ${{td}}%</span>
      </div>`;
  }}

  // Phase MAE bar chart
  const phaseLabels = s.has_phases
    ? ['Pre-Op','Induction','Maintenance','Recovery']
    : ['Induction','Maintenance','Deep'];
  const phaseMAEs = s.has_phases
    ? [s.mae_pre_op, s.mae_induction, s.mae_maintenance, s.mae_recovery]
    : [s.mae_induction, s.mae_maintenance, s.mae_deep];
  const phaseColors = s.has_phases ? ['#6366f1',AMBE,CYAN,GREEN] : [AMBE,CYAN,RED];
  const hasMAE = phaseMAEs.some(v => v != null);

  Plotly.newPlot('plot-phase', [{{
    type:'bar', orientation:'h',
    x: hasMAE ? phaseMAEs.map(v => v ?? 0) : phaseLabels.map(()=>0),
    y: phaseLabels,
    marker:{{ color:phaseColors }},
    text: hasMAE ? phaseMAEs.map(v => v != null ? v.toFixed(2) : 'N/A') : phaseLabels.map(()=>'N/A'),
    textposition:'outside', textfont:{{color:TXTH, size:11}},
    hovertemplate:'MAE: %{{x:.2f}} BIS<extra></extra>',
  }}], layout({{
    xaxis:{{ title:'MAE (BIS points)', ...LAYOUT_BASE.xaxis }},
    yaxis:{{ ...LAYOUT_BASE.yaxis }},
    margin:{{...LAYOUT_BASE.margin, l:110}},
    showlegend:false,
  }}), {{responsive:true, displayModeBar:false}});

  // Scatter: pred vs true
  const hasPred = D.pred != null;
  if (hasPred) {{
    const minV = Math.min(Math.min.apply(null,D.label), Math.min.apply(null,D.pred)) - 2;
    const maxV = Math.max(Math.max.apply(null,D.label), Math.max.apply(null,D.pred)) + 2;
    Plotly.newPlot('plot-scatter', [
      {{ x:D.label, y:D.pred, mode:'markers', name:'windows',
         marker:{{color:CYAN, size:3, opacity:0.4}},
         hovertemplate:'True: %{{x:.0f}}<br>Pred: %{{y:.0f}}<extra></extra>' }},
      {{ x:[minV,maxV], y:[minV,maxV], mode:'lines', name:'ideal',
         line:{{color:AMBE, width:1.5, dash:'dot'}}, showlegend:false }},
    ], layout({{
      xaxis:{{ title:'True BIS', range:[minV,maxV], ...LAYOUT_BASE.xaxis }},
      yaxis:{{ title:'Predicted BIS', range:[minV,maxV], ...LAYOUT_BASE.yaxis }},
      annotations:[{{xref:'paper',yref:'paper',x:0.02,y:0.97,
        text:`r = ${{s.pearson_r?.toFixed(3) ?? 'N/A'}}`,
        showarrow:false,font:{{size:12,color:GREEN}},xanchor:'left',yanchor:'top'}}],
    }}), {{responsive:true, displayModeBar:false}});
  }} else {{
    document.getElementById('plot-scatter').innerHTML =
      '<p style="text-align:center;color:#304560;padding:40px;font-size:.8rem">No model predictions (checkpoint not found)</p>';
  }}

  // Key metrics grid
  const grid = document.getElementById('stats-grid');
  const fmt = v => v != null ? v.toFixed(2)+' BIS' : 'N/A';
  const metrics = [
    ['BIS Mean+SD',     s.bis_mean?.toFixed(1)+' +/- '+s.bis_std?.toFixed(1), `Range: ${{s.bis_min?.toFixed(0)}}-${{s.bis_max?.toFixed(0)}}`, CYAN],
    ['Duration',        D.dur_min?.toFixed(1)+' min', `${{s.n_total?.toLocaleString()}} windows`, GREEN],
    ['MAE Overall',     fmt(s.mae_overall), 'Prediction error', AMBE],
    ['RMSE',            s.rmse != null ? s.rmse.toFixed(2)+' BIS' : 'N/A', 'Root mean sq. error', CYAN],
    ['Pearson r',       s.pearson_r != null ? s.pearson_r.toFixed(4) : 'N/A', 'Correlation', GREEN],
  ];
  if (s.has_phases) {{
    metrics.push(['MAE Pre-Op',      fmt(s.mae_pre_op),      'Phase 0: awake', '#6366f1']);
    metrics.push(['MAE Induction',   fmt(s.mae_induction),   'Phase 1: drug onset', AMBE]);
    metrics.push(['MAE Maintenance', fmt(s.mae_maintenance),  'Phase 2: stable anesthesia', CYAN]);
    metrics.push(['MAE Recovery',    fmt(s.mae_recovery),     'Phase 3: emergence', GREEN]);
    metrics.push(['Stim Events',     String(s.n_stim_events??0), 'BIS rises during maintenance', RED]);
  }} else {{
    metrics.push(['MAE Induction',   fmt(s.mae_induction),   'BIS >= 60', AMBE]);
    metrics.push(['MAE Maintenance', fmt(s.mae_maintenance),  'BIS 40-60', CYAN]);
    metrics.push(['MAE Deep',        fmt(s.mae_deep),         'BIS < 40', RED]);
  }}
  grid.innerHTML = metrics.map(([l,v,d,c]) => `
    <div class="stat-card">
      <div class="sv" style="color:${{c}}">${{v}}</div>
      <div class="sl">${{l}}</div>
      <div class="sd">${{d}}</div>
    </div>`).join('');
}}

// ── Tab navigation ────────────────────────────────────────────────────
const BUILT = {{}};
function showSection(id) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`[data-sec="${{id}}"]`).classList.add('active');

  if (!BUILT[id]) {{
    BUILT[id] = true;
    if (id==='sec-bis')      buildBIS();
    if (id==='sec-phases')   buildPhases();
    if (id==='sec-dsa')      buildDSA();
    if (id==='sec-bands')    buildBands();
    if (id==='sec-sqi')      buildSQI();
    if (id==='sec-features') {{ buildFeatures(); buildFoldChange(); buildFOOOF(); buildZscore(); buildADPR(); }}
    if (id==='sec-analysis') buildAnalysis();
  }}
}}

document.querySelectorAll('.nav-tab').forEach(tab => {{
  tab.addEventListener('click', () => showSection(tab.dataset.sec));
}});

// ── Init ──────────────────────────────────────────────────────────────
buildHeader();
showSection('sec-bis');
</script>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Report written -> {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_checkpoint() -> tuple[str | None, str | None]:
    """Auto-detect best checkpoint and its config in priority order v6>v5>v4>v2>v1."""
    candidates = [
        ("outputs/checkpoints/v6/best_model_v2.pt", "configs/pipeline_v6.yaml"),
        ("outputs/checkpoints/v5/best_model_v2.pt", "configs/pipeline_v5.yaml"),
        ("outputs/checkpoints/v4/best_model.pt",    "configs/pipeline_v4.yaml"),
        ("outputs/checkpoints/v3/best_model.pt",    "configs/pipeline_v3.yaml"),
        ("outputs/checkpoints/v2/best_model.pt",    "configs/pipeline_v2.yaml"),
        ("outputs/checkpoints/best_model.pt",       "configs/pipeline_v1.yaml"),
        ("outputs/checkpoints/v1/best_model.pt",    "configs/pipeline_v1.yaml"),
    ]
    for ckpt, cfg_path in candidates:
        if Path(ckpt).exists():
            return ckpt, cfg_path
    return None, "configs/pipeline_v6.yaml"


def main():
    # Auto-detect best checkpoint before parsing args
    _auto_ckpt, _auto_cfg = _find_best_checkpoint()

    parser = argparse.ArgumentParser(description="Single-file inference + HTML report")
    parser.add_argument("--vital",      default=None,
                        help="Path to .vital file (default: first in raw_data/)")
    parser.add_argument("--config",     default=_auto_cfg,
                        help=f"Config yaml (default: auto-detected -> {_auto_cfg})")
    parser.add_argument("--checkpoint", default=_auto_ckpt,
                        help=f"Checkpoint .pt path (default: auto-detected -> {_auto_ckpt})")
    parser.add_argument("--out",        default=None,
                        help="Output HTML path (default: outputs/reports/<case>.html)")
    parser.add_argument("--dsa-ch",     type=int, default=0,
                        help="Channel index for DSA (0=Fp1, 1=Fp2)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve vital file
    if args.vital:
        vital_path = Path(args.vital)
    else:
        raw_dir = Path(cfg["paths"]["raw_data"])
        vitals = sorted(raw_dir.glob("*.vital"))
        if not vitals:
            print(f"No .vital files found in {raw_dir}"); sys.exit(1)
        vital_path = vitals[0]

    case_id = vital_path.stem
    out_path = Path(args.out) if args.out else Path("outputs/reports") / f"{case_id}.html"

    print(f"\n{'='*60}")
    print(f"  Case: {case_id}")
    print(f"  Vital: {vital_path}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    engine = InferenceEngine(cfg, checkpoint=args.checkpoint)

    # 1. Load
    raw = engine.load_vital(str(vital_path))
    if raw is None: sys.exit(1)

    # 2. Preprocess
    print("Preprocessing ...")
    pre = engine.preprocess(raw)

    # 3. DSA
    print("Computing DSA ...")
    dsa = engine.compute_dsa(pre["eeg_filtered"], ch=args.dsa_ch)

    # 4. Band waves
    print("Computing band decomposition ...")
    bands = engine.compute_band_waves(pre["eeg_filtered"])

    # 5. Windows + features/SQI
    print("Extracting windows & features ...")
    windows = engine.extract_windows(pre)
    if not windows:
        print("ERROR: no valid windows extracted."); sys.exit(1)

    # 6. Model inference
    print("Running model inference ...")
    infer = engine.run_inference(windows)
    pred_bis = infer["pred_bis"]
    phases   = infer["phases"]
    stim     = infer["stim"]

    # 7. Stats
    stats = _phase_stats(windows["labels"], infer)
    print(f"\n{'-'*50}")
    print(f"  Windows   : {stats['n_total']:,}")
    print(f"  BIS mean  : {stats['bis_mean']:.1f} +/- {stats['bis_std']:.1f}")
    if stats.get("has_phases"):
        print(f"  Phases: Pre-Op {stats.get('pct_pre_op',0):.0f}%  "
              f"Induction {stats.get('pct_induction',0):.0f}%  "
              f"Maintenance {stats.get('pct_maintenance',0):.0f}%  "
              f"Recovery {stats.get('pct_recovery',0):.0f}%")
        print(f"  Stim events detected: {stats.get('n_stim_events',0)}")
    else:
        print(f"  Phase dist: Induction {stats['pct_induction']:.0f}%  "
              f"Maintenance {stats['pct_maintenance']:.0f}%  "
              f"Deep {stats['pct_deep']:.0f}%")
    if stats['mae_overall'] is not None:
        print(f"  MAE       : {stats['mae_overall']:.2f} BIS  (r={stats['pearson_r']:.4f})")
    print(f"{'-'*50}\n")

    # 8. HTML report
    generate_html(
        case_id=case_id,
        data={
            "times":    windows["times"],
            "labels":   windows["labels"],
            "pred_bis": pred_bis,
            "phases":   phases,
            "stim":     stim,
            "sqi":      windows["sqi"],
            "features": windows["features"],
            "dsa":      dsa,
            "bands":    bands,
        },
        stats=stats,
        out_path=out_path,
    )
    print(f"Done.  Open in browser: file:///{out_path.resolve()}")


if __name__ == "__main__":
    main()
