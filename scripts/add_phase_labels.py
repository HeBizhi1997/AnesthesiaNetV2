"""
add_phase_labels.py — Post-process dataset.h5 to add phase and stimulation labels.

Problem: loader.py:process_file() never writes 'phases' or 'stim_events' to HDF5.
Training uses all-maintenance defaults (phase=2, stim=0) for every window, making
the phase classification and stim detection tasks meaningless.

Solution: Infer phase labels from BIS trajectory using clinical rules, and detect
stimulation events as sudden BIS rises during maintenance.

Phase definitions:
  0 = pre_op     : BIS high (>80), before induction. Short initial period.
  1 = induction  : BIS dropping from >80 to maintenance level. 2-10 minutes.
  2 = maintenance: BIS stable at 30-70. Majority of surgery.
  3 = recovery   : BIS rising from maintenance back to >75. End of surgery.

Stimulation event:
  BIS rises >= STIM_THRESH points above recent 60s median during maintenance.
  A 45-second window around the rise peak is labeled as stim=1.

Usage:
    python scripts/add_phase_labels.py
    python scripts/add_phase_labels.py --h5 outputs/preprocessed/dataset.h5
    python scripts/add_phase_labels.py --overwrite   # recompute existing labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────────────────────────────────────
# Clinical thresholds
# ──────────────────────────────────────────────────────────────────────────────

INDUCTION_THRESH   = 75    # BIS below this sustained → past induction
MAINT_THRESH       = 65    # BIS below this for 120+ s → in maintenance
RECOVERY_THRESH    = 72    # BIS above this sustained at end → recovery
PREOP_MIN_BIS      = 80    # BIS must be above this to label as pre-op
STIM_DELTA         = 8.0   # BIS rise above 60s median to count as stimulation
STIM_WINDOW_SEC    = 45    # seconds marked around stim peak
SMOOTH_SEC         = 60    # uniform moving average for phase detection (seconds)
MIN_SUSTAINED_SEC  = 90    # minimum seconds of sustained BIS for phase assignment


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple causal moving average — no scipy dependency."""
    if window <= 1:
        return arr.copy()
    out = np.empty_like(arr)
    cumsum = np.cumsum(arr)
    out[0] = arr[0]
    for i in range(1, min(window, len(arr))):
        out[i] = cumsum[i] / (i + 1)
    out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    for i in range(min(window, len(arr)) - 1, 0, -1):
        if i < len(arr):
            pass  # already handled above
    return out


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values."""
    out = arr.copy()
    nans = np.isnan(out) | (out < 0) | (out > 100)
    if nans.all():
        out[:] = 50.0
        return out
    if not nans.any():
        return out
    idx = np.arange(len(out))
    out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])
    return out


def infer_phases(bis_arr: np.ndarray) -> np.ndarray:
    """
    Infer surgical phase labels from per-window BIS values.

    Args:
        bis_arr: (N,) BIS values in [10, 100] or NaN. One value per window
                 (stride_sec = 1 second per window in the HDF5).

    Returns:
        phases: (N,) int64 array with values 0–3.
    """
    N = len(bis_arr)
    phases = np.full(N, 2, dtype=np.int64)  # default: maintenance

    if N < 120:
        return phases

    bis = _fill_nan(bis_arr.astype(np.float64))
    bis_s = _smooth(bis, SMOOTH_SEC)  # smooth for robust threshold crossings

    # No deep anesthesia reached → probably awake recording or incomplete
    if bis_s.min() > 70:
        return phases

    # ── Step 1: Find induction point ─────────────────────────────────────────
    # Induction = first time BIS drops below INDUCTION_THRESH and stays for
    # at least MIN_SUSTAINED_SEC seconds.
    below_ind = bis_s < INDUCTION_THRESH

    induction_start_idx = None  # index where BIS first sustainedly drops
    for i in range(N):
        if below_ind[i]:
            sustained_end = min(i + MIN_SUSTAINED_SEC, N)
            if below_ind[i:sustained_end].mean() >= 0.7:
                induction_start_idx = i
                break

    if induction_start_idx is None:
        return phases

    # ── Step 2: Find pre-op end / induction start ────────────────────────────
    # Walk back from induction_start_idx to find last time BIS was above PREOP_MIN_BIS
    preop_last = 0
    for i in range(induction_start_idx, -1, -1):
        if bis_s[i] >= PREOP_MIN_BIS:
            preop_last = i
            break

    # Only label pre-op if there's a meaningful high-BIS period (>= 30s)
    if preop_last >= 30:
        phases[:preop_last + 1] = 0          # pre_op
        phases[preop_last + 1: induction_start_idx + 1] = 1  # induction transition

    # Label the induction drop zone
    # Induction = from preop_last to when BIS first stabilises below MAINT_THRESH
    below_maint = bis_s < MAINT_THRESH
    maint_start_idx = induction_start_idx  # default: maintenance starts at induction
    for i in range(induction_start_idx, min(induction_start_idx + 600, N)):
        sustained_end = min(i + MIN_SUSTAINED_SEC, N)
        if below_maint[i:sustained_end].mean() >= 0.8:
            maint_start_idx = i
            break

    if maint_start_idx > preop_last + 1:
        phases[preop_last + 1: maint_start_idx] = 1   # induction

    # Maintenance is default (2) for everything between maint_start and recovery

    # ── Step 3: Find recovery ─────────────────────────────────────────────────
    # Recovery = sustained BIS rise above RECOVERY_THRESH at the END of recording
    # Search from the last window backward
    above_rec = bis_s > RECOVERY_THRESH

    recovery_start_idx = None
    for i in range(N - 1, max(maint_start_idx, N - 7200), -1):
        if not above_rec[i]:
            # How long does it stay above from here to end?
            break
        recovery_start_idx = i

    if recovery_start_idx is not None:
        recovery_dur = N - recovery_start_idx
        # Only label recovery if:
        # 1. Duration >= MIN_SUSTAINED_SEC
        # 2. BIS at the start of recovery is lower than at end (genuinely rising)
        if (recovery_dur >= MIN_SUSTAINED_SEC and
                bis_s[recovery_start_idx] < bis_s[min(N - 1, recovery_start_idx + 60)] - 3):
            phases[recovery_start_idx:] = 3

    return phases


def detect_stim_events(bis_arr: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """
    Detect stimulation events: sudden BIS rises during maintenance (phase=2).

    A stimulation event is defined as: BIS rises >= STIM_DELTA points above the
    60-second running median during a maintenance window.
    A 45-second window around the peak of each rise is marked as stim=1.

    Args:
        bis_arr: (N,) BIS values.
        phases:  (N,) phase labels.

    Returns:
        stim: (N,) float32 array with values 0.0 or 1.0.
    """
    N = len(bis_arr)
    stim = np.zeros(N, dtype=np.float32)

    if N < 120:
        return stim

    bis = _fill_nan(bis_arr.astype(np.float64))

    BASELINE_SEC = 60
    HALF_WINDOW  = STIM_WINDOW_SEC // 2

    for i in range(BASELINE_SEC, N):
        if phases[i] != 2:  # only during maintenance
            continue

        # 60-second rolling median (causal: look backward only)
        baseline = np.median(bis[max(0, i - BASELINE_SEC): i])

        if bis[i] - baseline >= STIM_DELTA:
            # Mark window around this rise
            s = max(0, i - HALF_WINDOW)
            e = min(N, i + HALF_WINDOW)
            stim[s:e] = 1.0

    return stim


def process_h5(h5_path: str, overwrite: bool = False, verbose: bool = True) -> None:
    """
    Read each case from the HDF5, infer phase and stim labels, and write them back.

    Skips cases that already have 'phases' unless overwrite=True.
    """
    with h5py.File(h5_path, "a") as f:
        case_ids = list(f.keys())

    n_cases      = len(case_ids)
    n_skipped    = 0
    n_processed  = 0
    phase_counts = np.zeros(4, dtype=np.int64)
    stim_total   = 0

    for idx, cid in enumerate(case_ids):
        with h5py.File(h5_path, "a") as f:
            grp = f[cid]

            if "phases" in grp and not overwrite:
                n_skipped += 1
                continue

            bis_arr = grp["labels"][:]    # (N,)  per-window BIS

            phases = infer_phases(bis_arr)
            stim   = detect_stim_events(bis_arr, phases)

            # Delete old if overwriting
            for key in ("phases", "stim_events"):
                if key in grp:
                    del grp[key]

            grp.create_dataset("phases",
                               data=phases, dtype=np.int64,
                               compression="gzip", compression_opts=1)
            grp.create_dataset("stim_events",
                               data=stim,   dtype=np.float32,
                               compression="gzip", compression_opts=1)

        phase_counts += np.bincount(phases, minlength=4)
        stim_total   += int(stim.sum())
        n_processed  += 1

        if verbose and (idx + 1) % 100 == 0:
            print(f"  [{idx+1:4d}/{n_cases}] processed={n_processed} "
                  f"skipped={n_skipped}")

    total_windows = phase_counts.sum()
    print(f"\nDone.  processed={n_processed}, skipped={n_skipped}")
    if total_windows > 0:
        labels = ["pre_op", "induction", "maintenance", "recovery"]
        print(f"Phase distribution ({total_windows:,} windows):")
        for i, (label, cnt) in enumerate(zip(labels, phase_counts)):
            pct = 100.0 * cnt / total_windows
            print(f"  {i} {label:12s}: {cnt:8,}  ({pct:5.2f}%)")
        stim_rate = 100.0 * stim_total / total_windows
        print(f"Stim events:       {stim_total:8,}  ({stim_rate:5.2f}%)")

        # Sanity check
        ind_pct = 100.0 * phase_counts[1] / total_windows
        rec_pct = 100.0 * phase_counts[3] / total_windows
        if ind_pct < 0.05:
            print("[WARN] Induction < 0.05% — phase inference may have failed.")
        if rec_pct < 0.05:
            print("[WARN] Recovery < 0.05% — many recordings may end at maintenance.")
        if stim_rate < 0.1:
            print("[WARN] Stim rate < 0.1% — stim detection threshold may be too high.")


def main():
    parser = argparse.ArgumentParser(
        description="Add phase/stim labels to existing dataset.h5"
    )
    parser.add_argument("--h5",        default="outputs/preprocessed/dataset.h5")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if phases already present")
    parser.add_argument("--quiet",     action="store_true")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found. Run preprocess_data.py first.")
        sys.exit(1)

    print(f"Adding phase/stim labels to: {h5_path}")
    if args.overwrite:
        print("  --overwrite: recomputing all cases")

    process_h5(str(h5_path), overwrite=args.overwrite, verbose=not args.quiet)


if __name__ == "__main__":
    main()
