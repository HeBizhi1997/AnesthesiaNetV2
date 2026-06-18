"""
Validate the P2 heteroscedastic (Laplace) uncertainty head on held-out val cases.
CPU-only, no GPU contention with a running training job.

Loads the ep10 checkpoint (pre-Phase-3 EEG+stim+uncertainty model), runs EEG-only
inference on a handful of true validation cases, and checks whether the predicted
Laplace scale b = exp(log_b) is actually useful:

  1. Does b correlate with |error|?  (Spearman)
  2. Calibration: empirical MAE binned by predicted b  (should increase monotonically)
  3. Interval coverage vs Laplace theory:  P(|e|<b)=63.2%, <2b=86.5%, <3b=95.0%
  4. Per-phase mean b  (induction/recovery should be HIGHER than maintenance)

Usage: python scripts/analyze_uncertainty.py [n_val_cases]
"""
from __future__ import annotations
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch

from src.models.anesthesia_net_v3 import AnesthesiaNetV3
from src.data.dataset import _filter_cases_by_std
from src.data.dataset_v3 import SequenceDatasetV3

H5   = "outputs/preprocessed/dataset_v3_v13.h5"
N_VAL_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CKPT = sys.argv[2] if len(sys.argv) > 2 else "outputs/checkpoints/v16/best_model_v3.pt"
NAMES = ["pre_op", "induction", "maintenance", "recovery"]
torch.manual_seed(0)

# ── reproduce the exact build_datasets_v3 patient split (seed=42) ──────────────
with h5py.File(H5, "r") as f:
    all_cases = sorted(f.keys())
all_cases = _filter_cases_by_std(H5, all_cases, 10.0, 90.0)
rng = random.Random(42); rng.shuffle(all_cases)
n = len(all_cases); n_test = max(1, int(n*0.10)); n_val = max(1, int(n*0.15))
val_ids = all_cases[n_test:n_test + n_val][:N_VAL_CASES]
print(f"val cases used: {val_ids}")

# ── load model (ep10) on CPU ──────────────────────────────────────────────────
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = AnesthesiaNetV3.from_config(cfg)
model.load_state_dict(ck["model_state_dict"], strict=True)
model.eval()
print(f"loaded ep{ck['epoch']}  val_mae={ck['val_mae']:.2f}  "
      f"uncertainty={model.bis_uncertainty}  distill={model.distill_mode}")
assert model.logvar_head is not None, "checkpoint has no uncertainty head"

ds = SequenceDatasetV3(H5, val_ids, seq_len=300, seq_stride=150,
                       augment=False, cache_in_memory=True, phase_source="drug")
print(f"val sequences: {len(ds)}")

preds, labels, bs, phases, sqis = [], [], [], [], []
with torch.no_grad():
    for i in range(len(ds)):
        s = ds[i]
        out = model(s["wave"].unsqueeze(0), s["features"].unsqueeze(0),
                    s["sqi"].unsqueeze(0))
        preds.append(out["pred_bis"][0, :, 0].numpy() * 100.0)
        bs.append(np.exp(out["pred_logvar"][0, :, 0].numpy()) * 100.0)  # Laplace b in BIS
        labels.append(s["label_seq"].numpy() * 100.0)
        phases.append(s["phases"].numpy())
        sqis.append(s["sqi"].mean(-1).numpy())

pred = np.concatenate(preds); lab = np.concatenate(labels)
b = np.concatenate(bs); ph = np.concatenate(phases); sqi = np.concatenate(sqis)
ok = sqi > 0.5
pred, lab, b, ph = pred[ok], lab[ok], b[ok], ph[ok]
err = np.abs(pred - lab)
print(f"\nvalid timesteps: {len(err):,}   overall per-step MAE={err.mean():.2f}   "
      f"mean b={b.mean():.2f}")

# 1. correlation b vs |error|
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx*ry).sum() / (np.sqrt((rx**2).sum()*(ry**2).sum()) + 1e-9))
print(f"\n[1] Spearman(b, |error|) = {spearman(b, err):.3f}  (want >0  → uncertainty tracks error)")

# 2. calibration: MAE binned by predicted b (quartiles)
print("\n[2] Empirical MAE binned by predicted uncertainty b (want monotone ↑):")
qs = np.quantile(b, [0, .25, .5, .75, 1.0])
for j in range(4):
    m = (b >= qs[j]) & (b <= qs[j+1] if j == 3 else b < qs[j+1])
    if m.any():
        print(f"  b∈[{qs[j]:5.1f},{qs[j+1]:5.1f}]  n={m.sum():>7,}  "
              f"MAE={err[m].mean():5.2f}  mean_b={b[m].mean():5.2f}")

# 3. interval coverage vs Laplace theory
print("\n[3] Interval coverage (Laplace theory in parens):")
for k, theo in [(1, 0.632), (2, 0.865), (3, 0.950)]:
    cov = float((err <= k*b).mean())
    print(f"  P(|e| <= {k}b) = {cov*100:5.1f}%   (theory {theo*100:.1f}%)")

# 4. per-phase mean b and MAE
print("\n[4] Per-phase mean b and MAE (induction/recovery should have higher b):")
for p in range(4):
    m = ph == p
    if m.sum() > 50:
        print(f"  {NAMES[p]:<12} n={m.sum():>7,}  mean_b={b[m].mean():5.2f}  MAE={err[m].mean():5.2f}")
