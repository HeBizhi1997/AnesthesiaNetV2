"""
Two Dataset classes for anaesthesia depth modelling:

EEGDataset      — single-window dataset. Flat index over all windows.
                  Used for simple baselines or ablation studies.
                  Shuffling is safe.

SequenceDataset — consecutive-window sequences for LNN training.
                  Each sample is a sequence of seq_len consecutive windows
                  from the SAME patient, in chronological order.
                  The LNN hidden state h(t) is propagated across the sequence,
                  allowing it to learn temporal dynamics (induction hysteresis,
                  burst suppression, recovery lag).

                  Shuffling policy:
                    - NEVER shuffle within a sequence (breaks temporal order)
                    - Shuffle sequence START POINTS across cases (fine)
                  → Use shuffle=True in DataLoader; the Dataset guarantees
                    each returned item is already a valid temporal sequence.

                  cache_in_memory=True (default):
                    Loads all HDF5 data into RAM at __init__ time.
                    __getitem__ becomes pure numpy slicing (no I/O).
                    ~3 GB for 750k windows @ 128Hz/4s/2ch — well within
                    the 205 GB system RAM.  Eliminates the gzip-decompression
                    bottleneck that starved the GPU with only 4 DataLoader workers.

build_datasets() returns SequenceDataset by default (recommended for LNN).
Pass use_sequences=False to get EEGDataset for debugging.
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Single-window Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        case_ids: List[str],
        augment: bool = False,
        noise_std: float = 0.1,
    ):
        self.h5_path = h5_path
        self.augment = augment
        self.noise_std = noise_std
        self._index: List[Tuple[str, int]] = []
        with h5py.File(h5_path, "r") as f:
            for cid in case_ids:
                if cid not in f:
                    continue
                n = int(f[cid].attrs["n_windows"])
                self._index.extend([(cid, i) for i in range(n)])
        self._h5: Optional[h5py.File] = None

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, win_idx = self._index[idx]
        grp = self._open()[case_id]
        wave = grp["waves"][win_idx]
        feat = grp["features"][win_idx]
        sqi = grp["sqi"][win_idx]
        label = float(grp["labels"][win_idx])
        if self.augment:
            wave, feat = _augment(wave, feat, self.noise_std)
        return {
            "wave": torch.from_numpy(wave.astype(np.float32)),
            "features": torch.from_numpy(feat.astype(np.float32)),
            "sqi": torch.from_numpy(sqi.astype(np.float32)),
            "label": torch.tensor(label / 100.0, dtype=torch.float32),
            "label_raw": torch.tensor(label, dtype=torch.float32),
        }

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Dataset  (primary dataset for LNN training)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Each item is a contiguous temporal sequence of `seq_len` windows from
    one patient, followed by the label of the LAST window in the sequence.

    Index construction:
      For each case, valid start positions are all i where [i, i+seq_len)
      are consecutive in time (i.e., times[i+1] == times[i] + 1 second).
      A gap in the `times` array (e.g., due to skipped invalid labels)
      breaks the sequence; the start position is invalidated.

    Returns per sample:
      "wave"      : (seq_len, n_ch, win_samp)
      "features"  : (seq_len, n_feat)
      "sqi"       : (seq_len, n_ch)
      "label"     : scalar — BIS of last window, normalised [0,1]
      "label_raw" : scalar — BIS of last window, raw [0,100]
      "label_seq" : (seq_len,) — all labels in the sequence (for monotonic loss)

    Performance note (cache_in_memory=True):
      With gzip-compressed HDF5, each __getitem__ would decompress 4 arrays
      from disk.  On a 24-core machine with num_workers=8, multiple workers
      contend on the same HDF5 file handle, serialising I/O.
      Loading all data into RAM at init (cache_in_memory=True) replaces
      decompressed HDF5 reads with plain numpy slices — effectively free.
    """

    def __init__(
        self,
        h5_path: str,
        case_ids: List[str],
        seq_len: int = 10,
        seq_stride: Optional[int] = None,
        augment: bool = False,
        noise_std: float = 0.1,
        cache_in_memory: bool = True,
        min_seq_std: float = 0.0,
    ):
        """
        seq_stride  : step between consecutive sequence start indices.
            None / 1  : every window is a valid start (default, overlapping).
            seq_len   : non-overlapping chunks — good for long seq_len (e.g. 300)
                        to keep epoch size manageable.
        min_seq_std : minimum BIS std (in raw BIS units, 0-100) within a sequence.
            Sequences with BIS std < min_seq_std are skipped (flat/low-value).
            0.0 = no filtering (default).  Recommended: 3.0-5.0 for seq_len=300.
        """
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.seq_stride = seq_stride if seq_stride is not None else 1
        self.augment = augment
        self.noise_std = noise_std
        self.cache_in_memory = cache_in_memory
        self.min_seq_std = min_seq_std

        # Per-case numpy arrays (populated if cache_in_memory=True)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Build valid sequence start indices
        self._index: List[Tuple[str, int]] = []
        # Store last-window BIS for each sequence (for weighted sampling)
        self._seq_bis: List[float] = []

        with h5py.File(h5_path, "r") as f:
            for cid in case_ids:
                if cid not in f:
                    continue
                grp = f[cid]
                n = int(grp.attrs["n_windows"])
                if n < seq_len:
                    continue

                # Optionally load all arrays into RAM
                if cache_in_memory:
                    entry = {
                        "waves":    grp["waves"][:].astype(np.float32),
                        "features": grp["features"][:].astype(np.float32),
                        "sqi":      grp["sqi"][:].astype(np.float32),
                        "labels":   grp["labels"][:].astype(np.float32),
                    }
                    # Load multi-task labels if present in HDF5
                    if "phases" in grp:
                        entry["phases"]      = grp["phases"][:].astype(np.int64)
                        entry["stim_events"] = grp["stim_events"][:].astype(np.float32)
                    self._cache[cid] = entry

                labels_arr = (self._cache[cid]["labels"]
                              if cache_in_memory else grp["labels"][:])

                # `times` stores the EEG-end second for each window.
                # Consecutive windows should differ by stride_sec = 1.
                if "times" in grp:
                    times = grp["times"][:]
                    diffs = np.diff(times)
                    for i in range(0, n - seq_len + 1, self.seq_stride):
                        if not np.all(diffs[i: i + seq_len - 1] == 1):
                            continue
                        seg = labels_arr[i: i + seq_len]
                        if min_seq_std > 0 and seg.std() < min_seq_std:
                            continue
                        self._index.append((cid, i))
                        self._seq_bis.append(float(seg[-1]))
                else:
                    # No times array (old format) → assume all consecutive
                    for i in range(0, n - seq_len + 1, self.seq_stride):
                        seg = labels_arr[i: i + seq_len]
                        if min_seq_std > 0 and seg.std() < min_seq_std:
                            continue
                        self._index.append((cid, i))
                        self._seq_bis.append(float(seg[-1]))

        self._h5: Optional[h5py.File] = None
        # If all data is in _cache, no file handle is needed at runtime.
        # Setting this flag lets train.py set num_workers=0 automatically,
        # avoiding the Windows-spawn pickle-over-pipe OSError for large datasets.
        self.is_cached = cache_in_memory

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self._index)

    def boost_induction_sequences(self, factor: int) -> None:
        """Repeat sequences containing induction windows (phase==1) by `factor` times.

        Call this on the training dataset after construction to compensate for
        the severe under-representation of induction data (~0.3% of windows).
        Requires cache_in_memory=True and phases in the HDF5.

        factor=10 means each induction-containing sequence appears 11x total
        (1 original + 10 extra), raising induction from 0.3% toward ~3%.
        Only sequences where at least one window is phase==1 are boosted.
        """
        if factor <= 0:
            return
        if not self.cache_in_memory:
            print("boost_induction_sequences: requires cache_in_memory=True — skipped")
            return
        # Guard against accidental double-boosting (e.g. called twice on same dataset)
        if getattr(self, "_boost_applied", False):
            print("boost_induction_sequences: already applied, skipping to prevent "
                  "index/weight mismatch")
            return
        self._boost_applied = True

        extra_idx: list = []
        extra_bis: list = []
        for cid, start in self._index:
            c = self._cache[cid]
            if "phases" not in c:
                continue
            seg_phases = c["phases"][start: start + self.seq_len]
            if np.any(seg_phases == 1):    # contains induction
                bis_val = float(c["labels"][start + self.seq_len - 1])
                extra_idx.extend([(cid, start)] * factor)
                extra_bis.extend([bis_val] * factor)

        if extra_idx:
            # Extend both _index and _seq_bis atomically — they must stay in sync
            self._index   = self._index   + extra_idx
            self._seq_bis = self._seq_bis + extra_bis
            assert len(self._index) == len(self._seq_bis), (
                f"_index/seq_bis length mismatch: {len(self._index)} vs {len(self._seq_bis)}")
            print(f"Induction boost x{factor}: added {len(extra_idx):,} sequences "
                  f"(total {len(self._index):,})")
        else:
            print("boost_induction_sequences: no induction sequences found in dataset")

    def get_sample_weights(self, n_bins: int = 10) -> np.ndarray:
        """Return per-sample weights for WeightedRandomSampler.

        Divides the BIS range [0, 100] into n_bins equal bins and assigns
        each sequence a weight inversely proportional to its bin frequency.
        This balances the BIS distribution across batches so induction /
        emergence sequences (BIS 60-100, rare) receive the same expected
        gradient contribution as maintenance sequences (BIS 30-50, common).
        """
        bis = np.array(self._seq_bis, dtype=np.float32)
        bin_edges = np.linspace(0, 100, n_bins + 1)
        bin_idx = np.digitize(bis, bin_edges[1:-1])  # 0 .. n_bins-1
        counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float32)
        counts = np.maximum(counts, 1)  # avoid div-by-zero for empty bins
        bin_weights = 1.0 / counts
        sample_weights = bin_weights[bin_idx]
        return sample_weights

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, start = self._index[idx]
        end = start + self.seq_len

        if self.cache_in_memory:
            # Pure numpy slice — no I/O, no decompression
            c = self._cache[case_id]
            waves  = c["waves"][start:end].copy()
            feats  = c["features"][start:end].copy()
            sqis   = c["sqi"][start:end].copy()
            labels = c["labels"][start:end].copy()
            phases = c["phases"][start:end].copy()      if "phases"      in c else None
            stims  = c["stim_events"][start:end].copy() if "stim_events" in c else None
        else:
            grp    = self._open()[case_id]
            waves  = grp["waves"][start:end]
            feats  = grp["features"][start:end]
            sqis   = grp["sqi"][start:end]
            labels = grp["labels"][start:end]
            phases = grp["phases"][start:end]      if "phases"      in grp else None
            stims  = grp["stim_events"][start:end] if "stim_events" in grp else None

        if self.augment:
            noise = np.random.randn(*waves.shape).astype(np.float32) * self.noise_std
            scale = np.float32(np.random.uniform(0.85, 1.15))
            waves = waves * scale + noise
            if waves.shape[1] == 2 and np.random.rand() < 0.5:
                waves = waves[:, [1, 0], :]    # channel swap (all steps)

        last_label = float(labels[-1])
        out = {
            "wave":      torch.from_numpy(waves),
            "features":  torch.from_numpy(feats),
            "sqi":       torch.from_numpy(sqis),
            "label":     torch.tensor(last_label / 100.0, dtype=torch.float32),
            "label_raw": torch.tensor(last_label, dtype=torch.float32),
            "label_seq": torch.from_numpy(labels / 100.0),
        }
        if phases is not None:
            out["phases"]      = torch.from_numpy(phases.astype(np.int64))
            out["stim_events"] = torch.from_numpy(stims.astype(np.float32))
        return out

    def __del__(self):
        # Guard against AttributeError when __init__ raised before _h5 was set
        # (e.g. Windows multiprocessing pickle failure)
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _augment(
    wave: np.ndarray, feat: np.ndarray, noise_std: float
) -> Tuple[np.ndarray, np.ndarray]:
    wave = wave + np.random.randn(*wave.shape).astype(np.float32) * noise_std
    wave = wave * np.float32(np.random.uniform(0.85, 1.15))
    if wave.shape[0] == 2 and np.random.rand() < 0.5:
        wave = wave[[1, 0], :]
    return wave, feat


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def _filter_cases_by_std(
    h5_path: str,
    case_ids: List[str],
    pct_low: float = 0.0,
    pct_high: float = 100.0,
) -> List[str]:
    """Keep only cases whose BIS std falls in [pct_low, pct_high] percentile.

    This implements the document's "remove bottom/top X%" data quality step:
    - Bottom X% (low std): degenerate cases — BIS flat, equipment issue,
      recording started during stable deep anesthesia only.
    - Top X% (high std): potential artifact cases — BIS jumps wildly.
    - Middle 80%: normal surgery curves with induction + maintenance + emergence.

    Applied at CASE level (not sequence level) so that all temporal sequences
    from a retained case are kept, including flat maintenance-phase segments
    which teach the model to produce stable outputs for stable inputs.
    """
    if pct_low == 0.0 and pct_high == 100.0:
        return case_ids
    with h5py.File(h5_path, "r") as f:
        stds = np.array([f[cid]["labels"][:].std() for cid in case_ids],
                        dtype=np.float32)
    lo = np.percentile(stds, pct_low)
    hi = np.percentile(stds, pct_high)
    kept = [cid for cid, s in zip(case_ids, stds) if lo <= s <= hi]
    n_removed = len(case_ids) - len(kept)
    if n_removed > 0:
        print(f"Case filter (BIS std p{pct_low:.0f}-p{pct_high:.0f}, "
              f"{lo:.1f}-{hi:.1f}): kept {len(kept)}/{len(case_ids)} "
              f"(removed {n_removed})")
    return kept


def build_datasets(
    h5_path: str,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seq_len: int = 10,
    seq_stride: Optional[int] = None,
    seed: int = 42,
    noise_std: float = 0.05,
    use_sequences: bool = True,
    cache_in_memory: bool = True,
    min_seq_std: float = 0.0,
    case_std_pct_low: float = 0.0,
    case_std_pct_high: float = 100.0,
    induction_boost: int = 0,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Patient-level split (no data leakage across train/val/test).

    Returns (train_ds, val_ds, test_ds).

    case_std_pct_low / case_std_pct_high : percentile bounds for case-level
        BIS std filtering (document: "remove bottom/top X%").
        Default 0/100 = no filtering.  Recommended: 10/90 to remove degenerate
        cases (flat signal or extreme variability).
    min_seq_std : sequence-level BIS std threshold (raw BIS units).
        0.0 = no filtering (recommended; see note in SequenceDataset).
    induction_boost : repeat induction-containing sequences this many extra times
        in the training set.  0 = disabled (default).  Recommended: 10-20 to
        raise induction from 0.3% toward 3-6% of training steps.
        Applied only to the training dataset, not val/test.
    """
    with h5py.File(h5_path, "r") as f:
        all_cases = sorted(f.keys())

    # Case-level quality filter (before split, so filtering is consistent)
    all_cases = _filter_cases_by_std(
        h5_path, all_cases, case_std_pct_low, case_std_pct_high
    )

    rng = random.Random(seed)
    rng.shuffle(all_cases)

    n = len(all_cases)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    test_ids  = all_cases[:n_test]
    val_ids   = all_cases[n_test: n_test + n_val]
    train_ids = all_cases[n_test + n_val:]

    print(f"Patient split — train: {len(train_ids)}, "
          f"val: {len(val_ids)}, test: {len(test_ids)}")

    if use_sequences:
        Cls = SequenceDataset
        kw_train = dict(seq_len=seq_len, seq_stride=seq_stride, augment=True,
                        noise_std=noise_std, cache_in_memory=cache_in_memory,
                        min_seq_std=min_seq_std)
        kw_eval  = dict(seq_len=seq_len, seq_stride=seq_stride, augment=False,
                        cache_in_memory=cache_in_memory, min_seq_std=0.0)
    else:
        Cls = EEGDataset
        kw_train = dict(augment=True, noise_std=noise_std)
        kw_eval  = dict(augment=False)

    if cache_in_memory:
        print(f"Loading dataset into RAM (cache_in_memory=True)…")

    train_ds = Cls(h5_path, train_ids, **kw_train)
    val_ds   = Cls(h5_path, val_ids,   **kw_eval)
    test_ds  = Cls(h5_path, test_ids,  **kw_eval)

    # Induction oversampling: repeat sequences that include induction phase
    if use_sequences and induction_boost > 0:
        train_ds.boost_induction_sequences(induction_boost)

    print(f"  train sequences: {len(train_ds):,}, "
          f"val: {len(val_ds):,}, test: {len(test_ds):,}")

    return train_ds, val_ds, test_ds
