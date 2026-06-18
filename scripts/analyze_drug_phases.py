"""
Validate drug-derived phase labels (P0b) against the BIS signal they're meant to
replace — read-only, no GPU, reads only small 1-D per-case arrays (not waves).

For each phase we want a clinically sensible BIS distribution:
  pre_op     : high BIS (awake, ~85-98)
  induction  : spanning high→low (transition)
  maintenance: 40-60 (surgical anesthesia)
  recovery   : RISING and ending high (emergence) — if its BIS looks like
               maintenance, the labeler is over-marking the slow-decay tail.

Also reports drug-vs-BIS phase agreement and the corrected sqrt-inv-freq class
weights for the *drug* distribution.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np

H5 = sys.argv[1] if len(sys.argv) > 1 else "outputs/preprocessed/dataset_v3_v13.h5"
NAMES = ["pre_op", "induction", "maintenance", "recovery"]

bis_by_phase = {k: [] for k in range(4)}      # drug-phase → list of BIS values
bis_end_by_rec = []                            # recovery: (start_bis, end_bis) per case
agree = np.zeros((4, 4), dtype=np.int64)       # confusion drug(row) vs BIS(col)
n_cases = 0
rec_slope_pos = 0                              # recovery segments that actually rise
rec_total = 0

with h5py.File(H5, "r") as f:
    cids = list(f.keys())
    for cid in cids:
        g = f[cid]
        if "phases_drug" not in g:
            continue
        ph_d = g["phases_drug"][:].astype(int)
        bis  = g["labels"][:].astype(np.float32)
        ph_b = g["phases"][:].astype(int) if "phases" in g else None
        n = min(len(ph_d), len(bis))
        ph_d, bis = ph_d[:n], bis[:n]
        n_cases += 1

        for p in range(4):
            m = ph_d == p
            if m.any():
                # subsample to keep memory modest
                vals = bis[m]
                if len(vals) > 2000:
                    vals = vals[np.linspace(0, len(vals) - 1, 2000).astype(int)]
                bis_by_phase[p].extend(vals.tolist())

        # recovery slope check (does BIS rise across the recovery segment?)
        rm = ph_d == 3
        if rm.any():
            idx = np.where(rm)[0]
            seg = bis[idx[0]:idx[-1] + 1]
            if len(seg) >= 10:
                rec_total += 1
                if seg[-len(seg)//4:].mean() > seg[:len(seg)//4].mean() + 2:
                    rec_slope_pos += 1
                bis_end_by_rec.append((float(seg[:len(seg)//4].mean()),
                                       float(seg[-len(seg)//4:].mean())))

        if ph_b is not None:
            ph_b = ph_b[:n]
            for a in range(4):
                for b in range(4):
                    agree[a, b] += int(((ph_d == a) & (ph_b == b)).sum())

print(f"cases analyzed: {n_cases}\n")

print("Per drug-phase BIS distribution (true BIS value):")
print(f"  {'phase':<12} {'n':>10} {'mean':>6} {'p10':>5} {'p50':>5} {'p90':>5}")
total = sum(len(v) for v in bis_by_phase.values())
freqs = []
for p in range(4):
    v = np.array(bis_by_phase[p])
    freqs.append(len(v))
    if len(v):
        print(f"  {NAMES[p]:<12} {len(v):>10,} {v.mean():>6.1f} "
              f"{np.percentile(v,10):>5.0f} {np.percentile(v,50):>5.0f} {np.percentile(v,90):>5.0f}")
    else:
        print(f"  {NAMES[p]:<12} {0:>10}  (none)")

print(f"\nRecovery sanity: {rec_slope_pos}/{rec_total} recovery segments actually RISE "
      f"(>2 BIS from first→last quartile)")
if bis_end_by_rec:
    arr = np.array(bis_end_by_rec)
    print(f"  recovery start-BIS mean={arr[:,0].mean():.1f}  end-BIS mean={arr[:,1].mean():.1f}")

print("\nDrug(row) vs BIS(col) phase agreement (counts):")
print(f"  {'':<12}" + "".join(f"{n:>12}" for n in NAMES))
for a in range(4):
    row = agree[a]
    print(f"  {NAMES[a]:<12}" + "".join(f"{c:>12,}" for c in row))
diag = np.trace(agree); tot = agree.sum()
print(f"  overall drug==BIS agreement: {diag/max(tot,1)*100:.1f}%")

# corrected class weights for the drug distribution
fr = np.array(freqs, dtype=np.float64); fr = fr / fr.sum()
w = (1.0 / np.clip(fr, 1e-6, None)) ** 0.5
w = w / w.sum() * 4.0
print(f"\nDrug-phase frequencies: {[round(x,4) for x in fr.tolist()]}")
print(f"Corrected sqrt-inv-freq weights (vs old [0.78,2.15,0.12,0.96]):")
print(f"  {[round(x,3) for x in w.tolist()]}")
