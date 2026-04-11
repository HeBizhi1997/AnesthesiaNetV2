"""
phase_labeler.py - Automatic surgical phase and stimulation event labeling.

Phase definitions (clinical):
  0  PRE_OP      : Patient awake, BIS > 80, before drug administration
  1  INDUCTION   : BIS dropping from > 80 toward < 60 (drug takes effect)
  2  MAINTENANCE : BIS stable in 40-65, the bulk of surgery (>95% of time)
  3  RECOVERY    : BIS rising from < 60 back toward > 70 (drug clearance)

Stimulation events (during maintenance):
  Any BIS rise > STIM_RISE_THRESH points within STIM_WINDOW_SEC seconds.
  Caused by: electrocautery, repositioning, intubation, suturing, incision.

Label derivation is fully algorithmic from the BIS timeseries — no manual
annotation required. Applied to the HDF5 dataset at build time.
"""

from __future__ import annotations
import numpy as np

# ── Phase constants ────────────────────────────────────────────────────────────
PRE_OP      = 0
INDUCTION   = 1
MAINTENANCE = 2
RECOVERY    = 3
PHASE_NAMES = ["pre_op", "induction", "maintenance", "recovery"]

# ── Tunable thresholds ─────────────────────────────────────────────────────────
BIS_AWAKE         = 80.0   # BIS >= this → awake
BIS_ANESTHETIZED  = 65.0   # BIS <= this → under anesthesia
BIS_DEEP          = 40.0   # BIS <= this → deep anesthesia

# Induction: BIS must drop at least this much total
INDUCTION_DROP_MIN = 20.0
# Smoothing window for phase detection (seconds = windows at 1Hz)
SMOOTH_WIN = 30

# Stimulation: BIS rise within a rolling window
STIM_RISE_THRESH  = 6.0    # BIS points rise in STIM_WINDOW_SEC
STIM_WINDOW_SEC   = 60     # rolling window (seconds/windows at 1Hz)
STIM_COOLDOWN_SEC = 90     # min gap between two events


def smooth(arr: np.ndarray, win: int) -> np.ndarray:
    """Causal moving average (no future leakage, no boundary artifact).
    Each output[i] = mean(arr[max(0,i-win+1) : i+1])."""
    out = np.empty_like(arr, dtype=np.float64)
    cs = np.cumsum(arr, dtype=np.float64)
    for i in range(len(arr)):
        lo = max(0, i - win + 1)
        out[i] = (cs[i] - (cs[lo - 1] if lo > 0 else 0.0)) / (i - lo + 1)
    return out


def label_phases(bis: np.ndarray) -> np.ndarray:
    """
    Assign a phase label (0-3) to each window in a BIS timeseries.

    Parameters
    ----------
    bis : (N,) float array of BIS values [0, 100]

    Returns
    -------
    phases : (N,) uint8 array with values in {0,1,2,3}
    """
    N = len(bis)
    phases = np.full(N, MAINTENANCE, dtype=np.uint8)

    bis_s = smooth(bis, SMOOTH_WIN)

    # ── 1. Find the induction window ──────────────────────────────────────────
    # Induction starts at the last window where smoothed BIS > BIS_AWAKE
    # before the first sustained anesthetic period.
    # Induction ends when BIS_S first drops below BIS_ANESTHETIZED.

    induction_start = -1
    induction_end   = -1

    # Find where BIS first drops below anesthesia threshold sustainedly
    for i in range(N):
        if bis_s[i] <= BIS_ANESTHETIZED:
            induction_end = i
            break

    if induction_end > 0:
        # Walk backwards to find start of BIS drop
        for i in range(induction_end, -1, -1):
            if bis_s[i] >= BIS_AWAKE:
                induction_start = i
                break

        if induction_start < 0:
            induction_start = 0  # BIS never reached awake level → take from start

        # Mark PRE_OP: before induction_start where BIS was high
        phases[:induction_start] = PRE_OP
        # Mark INDUCTION: from induction_start to induction_end
        if induction_end > induction_start:
            phases[induction_start:induction_end] = INDUCTION

    # ── 2. Find the recovery window ──────────────────────────────────────────
    # Recovery starts when BIS begins sustained rise after maintenance,
    # heading back toward awake. We look from the end.

    recovery_start = -1

    # Find the last window where BIS_S was below anesthesia threshold
    last_under = N - 1
    for i in range(N - 1, -1, -1):
        if bis_s[i] <= BIS_ANESTHETIZED:
            last_under = i
            break

    # Walk forward from last_under to find sustained BIS rise
    if last_under < N - 1:
        # Check if BIS actually rises significantly after last_under
        tail = bis_s[last_under:]
        if tail.max() - tail.min() >= INDUCTION_DROP_MIN:
            # Find start of the rise: where BIS begins increasing consistently
            # Use gradient: recovery starts when 30s-smoothed gradient > 0
            grad = np.gradient(bis_s)
            grad_s = smooth(grad, SMOOTH_WIN)
            for i in range(last_under, N):
                if grad_s[i] > 0.05:   # rising by > 0.05 BIS/sec
                    recovery_start = i
                    break

    if recovery_start > 0 and recovery_start > induction_end:
        phases[recovery_start:] = RECOVERY

    # ── 3. Maintenance is everything between induction_end and recovery_start ─
    maint_start = max(0, induction_end)
    maint_end   = recovery_start if recovery_start > 0 else N
    phases[maint_start:maint_end] = MAINTENANCE

    # Patch: any window in "maintenance" with BIS > BIS_AWAKE is actually
    # still pre-op (late recording start or awake patient mid-surgery)
    awake_during_maint = (phases == MAINTENANCE) & (bis > BIS_AWAKE)
    phases[awake_during_maint] = PRE_OP

    return phases


def label_stimulation(bis: np.ndarray,
                       phases: np.ndarray) -> np.ndarray:
    """
    Detect stimulation events: rapid BIS increases during maintenance.

    Returns binary array (1 = stimulation event detected at this window).
    Only labels windows in MAINTENANCE phase.
    """
    N = len(bis)
    stim = np.zeros(N, dtype=np.float32)
    last_event = -STIM_COOLDOWN_SEC

    for i in range(STIM_WINDOW_SEC, N):
        if phases[i] != MAINTENANCE:
            continue
        if i - last_event < STIM_COOLDOWN_SEC:
            continue
        window = bis[i - STIM_WINDOW_SEC: i]
        rise = bis[i] - window.min()
        if rise >= STIM_RISE_THRESH:
            stim[i] = 1.0
            last_event = i

    return stim


def compute_phase_weights(phases: np.ndarray) -> np.ndarray:
    """
    Per-sample class weights for phase classification loss.
    Inverse frequency weighting to handle extreme class imbalance.
    """
    counts = np.bincount(phases.astype(int), minlength=4).astype(float)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights /= weights.sum()
    return weights[phases.astype(int)]


def augment_hdf5_with_labels(h5_path: str, verbose: bool = True) -> None:
    """
    Add 'phases' and 'stim_events' datasets to every case in the HDF5 file.
    Idempotent: skips cases that already have both datasets.
    """
    import h5py

    with h5py.File(h5_path, "a") as f:
        case_ids = list(f.keys())
        n_done = 0
        phase_counts = np.zeros(4, dtype=np.int64)

        for cid in case_ids:
            grp = f[cid]
            if "phases" in grp and "stim_events" in grp:
                # Already labeled — accumulate stats only
                phase_counts += np.bincount(grp["phases"][:].astype(int), minlength=4)
                continue

            bis = grp["labels"][:].astype(np.float32)
            phases = label_phases(bis)
            stim   = label_stimulation(bis, phases)

            grp.create_dataset("phases",      data=phases, compression="gzip")
            grp.create_dataset("stim_events", data=stim,   compression="gzip")
            phase_counts += np.bincount(phases.astype(int), minlength=4)
            n_done += 1

        if verbose:
            total = phase_counts.sum()
            print(f"Labeled {n_done} new cases  "
                  f"(total windows: {total:,})")
            for i, name in enumerate(PHASE_NAMES):
                print(f"  {name:12s}: {phase_counts[i]:8,}  "
                      f"({phase_counts[i]/total*100:.1f}%)")
