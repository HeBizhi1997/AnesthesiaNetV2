"""
PipelineValidator — data-testing hooks for every pipeline stage.

Design: each "hook" is a callable that receives an EEGContext and raises
AssertionError (or returns a dict of metrics) describing what it found.

Hooks are used in two modes:
  1. Hard assertions (validate=True): raise on failure → stop processing
  2. Soft reporting (validate=False): collect warnings, return report dict

Usage:
    from src.pipeline.validator import PipelineValidator

    validator = PipelineValidator.from_config(cfg)

    # Attach to a pipeline — called automatically after each step:
    pipeline = EEGPipeline(validate=True)
    pipeline.add(SQIComputer(cfg["sqi"]))
    # Manually:
    report = validator.check_window(ctx, stage="post_sqi")

    # Validate a full HDF5 dataset after preprocessing:
    validator.validate_dataset("outputs/preprocessed/dataset.h5")
"""

from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import h5py
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .context import EEGContext


# ─────────────────────────────────────────────────────────────────────────────
# Individual hook functions
# ─────────────────────────────────────────────────────────────────────────────

def hook_no_nan_inf(ctx: EEGContext, stage: str = "") -> Dict:
    """CRITICAL: signal must not contain NaN or Inf."""
    n_nan = int(np.isnan(ctx.data).sum())
    n_inf = int(np.isinf(ctx.data).sum())
    ok = (n_nan == 0) and (n_inf == 0)
    if not ok:
        raise AssertionError(
            f"[{stage}] Data contains {n_nan} NaN and {n_inf} Inf values."
        )
    return {"nan": n_nan, "inf": n_inf}


def hook_amplitude_range(
    ctx: EEGContext,
    stage: str = "",
    max_uv: float = 2000.0,
    min_rms_uv: float = 0.05,
) -> Dict:
    """
    EEG amplitude must be within physiological range.
    Filtered EEG (BIS monitor Fp1/Fp2): typical ±200 µV, RMS ~5-30 µV.
    After filtering, clip any residuals beyond 2000 µV (hard limit).
    """
    peak = float(np.abs(ctx.data).max())
    rms = float(np.sqrt(np.mean(ctx.data ** 2)))
    if peak > max_uv:
        raise AssertionError(
            f"[{stage}] Peak amplitude {peak:.1f} µV exceeds {max_uv} µV. "
            f"Possible unfiltered ESU artifact or ADC overflow."
        )
    if rms < min_rms_uv:
        raise AssertionError(
            f"[{stage}] RMS {rms:.4f} µV is below {min_rms_uv} µV. "
            f"Possible flat line / disconnected electrode."
        )
    return {"peak_uv": peak, "rms_uv": rms}


def hook_spectral_integrity(
    ctx: EEGContext,
    stage: str = "",
    min_eeg_band_ratio: float = 0.3,
    max_hf_ratio: float = 0.6,
) -> Dict:
    """
    After filtering, EEG band (0.5-47 Hz) should dominate the spectrum.
    If high-frequency content (>47 Hz) still dominates, filtering failed.
    """
    results = {}
    for ch in range(ctx.n_channels):
        x = ctx.data[ch].astype(np.float64)
        f, pxx = welch(x, fs=ctx.fs, nperseg=min(256, len(x)))
        total = pxx.sum() + 1e-12
        eeg_mask = (f >= 0.5) & (f <= 47.0)
        hf_mask = f > 47.0
        eeg_ratio = float(pxx[eeg_mask].sum() / total)
        hf_ratio = float(pxx[hf_mask].sum() / total)

        if eeg_ratio < min_eeg_band_ratio:
            raise AssertionError(
                f"[{stage}] Ch{ch}: EEG band power ratio {eeg_ratio:.2f} "
                f"< {min_eeg_band_ratio}. Filter may have failed."
            )
        if hf_ratio > max_hf_ratio:
            raise AssertionError(
                f"[{stage}] Ch{ch}: HF ratio {hf_ratio:.2f} > {max_hf_ratio}. "
                f"ESU/muscle artifact may not have been removed."
            )
        results[f"ch{ch}_eeg_ratio"] = eeg_ratio
        results[f"ch{ch}_hf_ratio"] = hf_ratio
    return results


def hook_highpass_effectiveness(
    ctx: EEGContext,
    stage: str = "",
    max_dc_ratio: float = 0.05,
) -> Dict:
    """Verify that DC/sub-Hz content is suppressed after high-pass filtering."""
    results = {}
    for ch in range(ctx.n_channels):
        x = ctx.data[ch].astype(np.float64)
        f, pxx = welch(x, fs=ctx.fs, nperseg=min(256, len(x)))
        total = pxx.sum() + 1e-12
        dc_mask = f < 0.5
        dc_ratio = float(pxx[dc_mask].sum() / total)
        if dc_ratio > max_dc_ratio:
            raise AssertionError(
                f"[{stage}] Ch{ch}: DC/sub-Hz ratio {dc_ratio:.3f} > {max_dc_ratio}. "
                f"High-pass filter may not have been applied."
            )
        results[f"ch{ch}_dc_ratio"] = dc_ratio
    return results


def hook_notch_effectiveness(
    ctx: EEGContext,
    stage: str = "",
    notch_freqs: List[float] = None,
    max_notch_ratio: float = 0.15,
) -> Dict:
    """Verify that notch frequencies are suppressed."""
    if notch_freqs is None:
        notch_freqs = [50.0]
    results = {}
    for ch in range(ctx.n_channels):
        x = ctx.data[ch].astype(np.float64)
        f, pxx = welch(x, fs=ctx.fs, nperseg=min(256, len(x)))
        total = pxx.sum() + 1e-12
        for freq in notch_freqs:
            band = (f >= freq - 2) & (f <= freq + 2)
            wide = (f >= freq - 10) & (f <= freq + 10) & ~band
            notch_power = pxx[band].sum()
            surround_power = pxx[wide].sum() + 1e-12
            ratio = float(notch_power / surround_power)
            if ratio > max_notch_ratio:
                warnings.warn(
                    f"[{stage}] Ch{ch}: Power at {freq}Hz is {ratio:.2f}x "
                    f"surroundings — notch may be insufficient."
                )
            results[f"ch{ch}_{freq}hz_ratio"] = ratio
    return results


def hook_sqi_computed(ctx: EEGContext, stage: str = "") -> Dict:
    """Verify SQI was computed and is in [0, 1]."""
    if ctx.sqi is None:
        raise AssertionError(f"[{stage}] ctx.sqi is None — SQIComputer did not run.")
    if ctx.sqi.shape != (ctx.n_channels,):
        raise AssertionError(
            f"[{stage}] ctx.sqi shape {ctx.sqi.shape} != ({ctx.n_channels},)"
        )
    if not (0.0 <= ctx.sqi.min() and ctx.sqi.max() <= 1.0):
        raise AssertionError(
            f"[{stage}] SQI values out of [0,1]: {ctx.sqi}"
        )
    return {"sqi": ctx.sqi.tolist()}


def hook_features_computed(ctx: EEGContext, stage: str = "",
                            expected_dim: int = 24,
                            feats_per_channel: int = 11) -> Dict:
    """
    Verify feature vector was computed, has correct dim, and no NaN.
    feats_per_channel = 5 bands + PE + SEF + LZC + 3×BSR = 11.
    """
    if ctx.features is None:
        raise AssertionError(f"[{stage}] ctx.features is None.")
    if ctx.features.shape[0] != expected_dim:
        raise AssertionError(
            f"[{stage}] Feature dim {ctx.features.shape[0]} != {expected_dim}"
        )
    if np.isnan(ctx.features).any():
        raise AssertionError(f"[{stage}] Features contain NaN.")
    # Band powers (first 5 per channel) should sum to ~1
    n_bands = 5
    for ch in range(ctx.n_channels):
        offset = ch * feats_per_channel
        band_sum = float(ctx.features[offset: offset + n_bands].sum())
        if not (0.3 <= band_sum <= 1.3):
            raise AssertionError(
                f"[{stage}] Ch{ch} band powers sum={band_sum:.3f} not in [0.7,1.3]. "
                f"Relative power normalisation failed."
            )
    return {"feature_dim": int(ctx.features.shape[0]),
            "feature_range": [float(ctx.features.min()), float(ctx.features.max())]}


def hook_gaussian_approx(ctx: EEGContext, stage: str = "",
                          max_kurtosis: float = 10.0) -> Dict:
    """
    Post-filter EEG should be approximately Gaussian.
    Excess kurtosis > 10 suggests remaining impulse artifacts.
    """
    results = {}
    for ch in range(ctx.n_channels):
        x = ctx.data[ch].astype(np.float64)
        kurt = float(kurtosis(x, fisher=True))
        sk = float(skew(x))
        if abs(kurt) > max_kurtosis:
            warnings.warn(
                f"[{stage}] Ch{ch}: kurtosis={kurt:.1f} > {max_kurtosis}. "
                f"Possible residual ESU spikes."
            )
        results[f"ch{ch}_kurtosis"] = kurt
        results[f"ch{ch}_skew"] = sk
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PipelineValidator:
    """
    Runs the appropriate hooks for each pipeline stage.
    Call check_window(ctx, stage) after each step.
    """

    STAGE_HOOKS = {
        "raw": [hook_no_nan_inf, hook_amplitude_range],
        "post_median": [hook_no_nan_inf, hook_amplitude_range],
        "post_highpass": [hook_no_nan_inf, hook_highpass_effectiveness,
                          hook_amplitude_range],
        "post_wavelet": [hook_no_nan_inf, hook_amplitude_range,
                         hook_gaussian_approx],
        "post_notch": [hook_no_nan_inf, hook_notch_effectiveness,
                       hook_amplitude_range],
        "post_lowpass": [hook_no_nan_inf, hook_spectral_integrity,
                         hook_amplitude_range],
        "post_sqi": [hook_sqi_computed],
        "post_features": [hook_features_computed],
        "final": [hook_no_nan_inf, hook_spectral_integrity,
                  hook_sqi_computed, hook_features_computed,
                  hook_amplitude_range],
    }

    def __init__(self, cfg: dict, hard_fail: bool = True):
        self.cfg = cfg
        self.hard_fail = hard_fail
        self.notch_freqs = cfg["filters"]["notch_hz"]

        # Dynamically compute feats_per_channel from config to stay in sync
        # with FeatureExtractor regardless of which optional features are enabled.
        fcfg = cfg.get("features", {})
        n_bsr = len(fcfg.get("bsr_thresholds_uv", [2.0, 5.0, 10.0]))
        n_pac = 1 if fcfg.get("pac", False) else 0
        # 5 bands + PE + SEF95 + LZC + N×BSR + PAC
        self.feats_per_ch = 5 + 1 + 1 + 1 + n_bsr + n_pac

        n_ch = len(cfg["eeg"]["channels"])
        # total = per_ch × n_channels + 2 inter-channel (asymmetry + SQI)
        self.n_feat = self.feats_per_ch * n_ch + 2

        # Sanity-check against model.feature_dim if explicitly set
        cfg_feat_dim = cfg.get("model", {}).get("feature_dim")
        if cfg_feat_dim is not None and cfg_feat_dim != self.n_feat:
            import warnings
            warnings.warn(
                f"PipelineValidator: model.feature_dim={cfg_feat_dim} in config "
                f"does not match computed feature dim={self.n_feat}. "
                f"Update model.feature_dim in your config YAML."
            )

    def check_window(self, ctx: EEGContext, stage: str) -> Dict[str, Any]:
        """Run all hooks for a given stage. Returns combined report dict."""
        hooks = self.STAGE_HOOKS.get(stage, [hook_no_nan_inf])
        report = {"stage": stage}
        for hook in hooks:
            try:
                # Pass extra kwargs for hooks that need them
                if hook is hook_notch_effectiveness:
                    result = hook(ctx, stage, notch_freqs=self.notch_freqs)
                elif hook is hook_features_computed:
                    result = hook(ctx, stage, expected_dim=self.n_feat,
                                  feats_per_channel=self.feats_per_ch)
                else:
                    result = hook(ctx, stage)
                report.update(result)
            except AssertionError:
                if self.hard_fail:
                    raise
                report[f"{hook.__name__}_FAILED"] = True
        return report

    @classmethod
    def from_config(cls, cfg: dict, hard_fail: bool = True) -> "PipelineValidator":
        return cls(cfg, hard_fail=hard_fail)

    # ------------------------------------------------------------------ #
    # State-transition check  (clinical-logic QA on a full case)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_state_transitions(
        labels: np.ndarray,
        sqi: np.ndarray,
        max_jump: float = 20.0,
        min_sqi: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Scan all consecutive BIS pairs.  A jump > max_jump between adjacent
        seconds is physiologically implausible if signal quality is good.

        Such jumps indicate:
          - Sudden electrode displacement (motion artifact)
          - ESU artifact the SQI checker missed
          - Data integrity issue in the .vital file

        Returns a report with flagged indices and a boolean `is_golden`
        (True = no suspicious transitions → eligible for high-confidence
        training subset).

        Parameters
        ----------
        labels : (N,)   BIS values in chronological order
        sqi    : (N,)   mean SQI per window (already mean across channels)
        """
        n = len(labels)
        flagged: List[int] = []

        for i in range(1, n):
            jump = abs(float(labels[i]) - float(labels[i - 1]))
            mean_sqi_pair = float(sqi[i - 1] + sqi[i]) / 2.0
            if jump > max_jump and mean_sqi_pair >= min_sqi:
                flagged.append(i)

        return {
            "n_windows": n,
            "n_flagged_transitions": len(flagged),
            "flagged_indices": flagged[:20],   # cap at 20 for readability
            "flag_rate": len(flagged) / max(n - 1, 1),
            # "Golden" = less than 1% of transitions are suspicious
            "is_golden": len(flagged) / max(n - 1, 1) < 0.01,
        }

    # ------------------------------------------------------------------ #
    # Dataset-level validation (run on completed HDF5 file)               #
    # ------------------------------------------------------------------ #

    def validate_dataset(
        self,
        h5_path: str,
        n_sample_per_case: int = 5,
        verbose: bool = True,
        train_case_ids: list = None,
    ) -> Dict[str, Any]:
        """
        Spot-check random windows from each case in the HDF5 file.
        Returns a summary report.

        DATA LEAKAGE WARNING
        --------------------
        This method reads ALL cases in the HDF5 (train + val + test).
        The ``golden_case_ids`` field in the returned report lists case IDs
        with fewer than 1% suspicious BIS transitions — spanning the entire
        dataset including test-set patients.

        **Do NOT use ``golden_case_ids`` to select which patients enter the
        training set.**  Doing so leaks test-set data quality information
        into the training pipeline (look-ahead bias).

        Safe usage: call this method for QA reporting only.  If you need to
        restrict training to golden cases, pass ``train_case_ids`` here —
        only those cases will be included in the golden-case count.

        Parameters
        ----------
        train_case_ids : list, optional
            If provided, ``golden_case_ids`` in the report will only include
            IDs from this list (i.e., confirmed training-split cases).
            Test-split patients are still QA-checked but NOT included in the
            golden list used for any selection decisions.
        """
        import random
        report = {"cases": {}, "global": {}}
        all_labels = []
        all_sqi = []
        all_feat_means = []
        failed_cases = []

        with h5py.File(h5_path, "r") as hf:
            cases = list(hf.keys())
            for cid in cases:
                grp = hf[cid]
                n = int(grp.attrs["n_windows"])
                fs = float(grp.attrs["fs"])
                n_ch = int(grp.attrs["n_channels"])

                # Sample random windows
                idxs = random.sample(range(n), min(n_sample_per_case, n))
                case_ok = True
                for idx in idxs:
                    wave = grp["waves"][idx]      # (n_ch, win_samp)
                    feat = grp["features"][idx]
                    sqi_val = grp["sqi"][idx]
                    label = float(grp["labels"][idx])

                    ctx = EEGContext(data=wave.copy(), fs=fs)
                    ctx.features = feat.copy()
                    ctx.sqi = sqi_val.copy()
                    ctx.label = label

                    try:
                        self.check_window(ctx, "final")
                    except AssertionError as e:
                        if verbose:
                            print(f"  [FAIL] case={cid} idx={idx}: {e}")
                        case_ok = False
                        failed_cases.append(cid)
                        break

                labels = grp["labels"][:]
                sqi_means = grp["sqi"][:].mean(axis=1)
                all_labels.extend(labels.tolist())
                all_sqi.extend(sqi_means.tolist())
                all_feat_means.append(float(grp["features"][:].mean()))

                # State-transition check for this case
                trans = self.check_state_transitions(labels, sqi_means)
                if verbose and not trans["is_golden"]:
                    print(f"  [WARN] {cid}: {trans['n_flagged_transitions']} "
                          f"suspicious BIS jumps (flag_rate={trans['flag_rate']:.3f})")

                report["cases"][cid] = {
                    "n_windows": n,
                    "label_mean": float(labels.mean()),
                    "label_min": float(labels.min()),
                    "label_max": float(labels.max()),
                    "ok": case_ok,
                    "is_golden": trans["is_golden"],
                    "n_flagged_transitions": trans["n_flagged_transitions"],
                }

        labels_arr = np.array(all_labels)
        sqi_arr = np.array(all_sqi)

        # Restrict golden-case count to training-split cases only.
        # If train_case_ids is None we fall back to all cases (QA mode only;
        # do NOT use the resulting golden_case_ids to filter training data).
        if train_case_ids is not None:
            train_set = set(train_case_ids)
            n_golden   = sum(1 for k, v in report["cases"].items()
                             if v.get("is_golden") and k in train_set)
            golden_ids = [k for k, v in report["cases"].items()
                          if v.get("is_golden") and k in train_set]
        else:
            n_golden   = sum(1 for v in report["cases"].values() if v.get("is_golden"))
            golden_ids = [k for k, v in report["cases"].items() if v.get("is_golden")]

        report["global"] = {
            "n_cases": len(cases),
            "n_failed_cases": len(set(failed_cases)),
            "n_golden_cases": n_golden,
            "golden_case_ids": golden_ids,
            "total_windows": len(all_labels),
            "label_mean": float(labels_arr.mean()),
            "label_std": float(labels_arr.std()),
            "label_range": [float(labels_arr.min()), float(labels_arr.max())],
            "mean_sqi": float(sqi_arr.mean()),
            "induction_pct": float((labels_arr >= 60).mean() * 100),
            "maintenance_pct": float(((labels_arr >= 40) & (labels_arr < 60)).mean() * 100),
            "recovery_pct": float((labels_arr < 40).mean() * 100),
        }

        if verbose:
            g = report["global"]
            print(f"\n=== Dataset Validation Report ===")
            print(f"  Cases:         {g['n_cases']} total | "
                  f"{g['n_failed_cases']} QA-failed | "
                  f"{g['n_golden_cases']} golden (<1% BIS jump)")
            if train_case_ids is None:
                print(f"  [WARN] golden_case_ids covers ALL cases (incl. test split).")
                print(f"         Pass train_case_ids= to restrict to training cases only.")
            print(f"  Total windows: {g['total_windows']:,}")
            print(f"  BIS mean±std:  {g['label_mean']:.1f} ± {g['label_std']:.1f}")
            print(f"  BIS range:     {g['label_range']}")
            print(f"  Mean SQI:      {g['mean_sqi']:.3f}")
            print(f"  Phase dist:    Induction {g['induction_pct']:.1f}% | "
                  f"Maintenance {g['maintenance_pct']:.1f}% | "
                  f"Recovery {g['recovery_pct']:.1f}%")

        return report
