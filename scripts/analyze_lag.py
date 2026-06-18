"""
Direct test of the BIS-lag / anticipatory hypothesis (motivated by the recovery
over-confidence found in analyze_uncertainty.py).

For each phase, compute MAE between pred(t) and label(t+k) over a sweep of k.
If the optimal k>0 (esp. in recovery), the model's prediction LEADS the lagged BIS
label — meaning (a) the recovery over-confidence is the model being right about the
near future, and (b) anticipatory training (bis_anticipate_steps=k) should help.

CPU-only. Usage: python scripts/analyze_lag.py [n_val_cases]
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
CKPT = "outputs/checkpoints/v16/best_model_v3.pt"
N_VAL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
KS = [0, 3, 5, 8, 10, 15, 20, 25, 30]          # lag steps (≈ seconds, stride 1s)
NAMES = ["pre_op", "induction", "maintenance", "recovery"]
torch.manual_seed(0)

with h5py.File(H5, "r") as f:
    all_cases = sorted(f.keys())
all_cases = _filter_cases_by_std(H5, all_cases, 10.0, 90.0)
rng = random.Random(42); rng.shuffle(all_cases)
n = len(all_cases); n_test = max(1, int(n*0.10)); n_val = max(1, int(n*0.15))
val_ids = all_cases[n_test:n_test + n_val][:N_VAL]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = AnesthesiaNetV3.from_config(ck["cfg"])
model.load_state_dict(ck["model_state_dict"], strict=True); model.eval()
print(f"ep{ck['epoch']}  val_mae={ck['val_mae']:.2f}   lag sweep k={KS}")

ds = SequenceDatasetV3(H5, val_ids, seq_len=300, seq_stride=150,
                       augment=False, cache_in_memory=True, phase_source="drug")

# accumulate sum|err| and count per (phase, k)
serr = {p: {k: 0.0 for k in KS} for p in range(4)}
cnt  = {p: {k: 0   for k in KS} for p in range(4)}
serr_all = {k: 0.0 for k in KS}; cnt_all = {k: 0 for k in KS}

with torch.no_grad():
    for i in range(len(ds)):
        s = ds[i]
        out = model(s["wave"].unsqueeze(0), s["features"].unsqueeze(0), s["sqi"].unsqueeze(0))
        pred = out["pred_bis"][0, :, 0].numpy() * 100.0
        lab  = s["label_seq"].numpy() * 100.0
        ph   = s["phases"].numpy()
        sqi  = s["sqi"].mean(-1).numpy()
        T = len(pred)
        for k in KS:
            if T - k < 2: continue
            e = np.abs(pred[:T-k] - lab[k:])          # pred(t) vs label(t+k)
            m = sqi[:T-k] > 0.5
            pk = ph[:T-k]
            for p in range(4):
                sel = m & (pk == p)
                serr[p][k] += float(e[sel].sum()); cnt[p][k] += int(sel.sum())
            serr_all[k] += float(e[m].sum()); cnt_all[k] += int(m.sum())

def best(d_sum, d_cnt):
    maes = {k: (d_sum[k]/d_cnt[k] if d_cnt[k] else np.nan) for k in KS}
    bk = min((k for k in KS if d_cnt[k]), key=lambda k: maes[k])
    return maes, bk

print("\nMAE( pred(t), label(t+k) ) vs lag k  --  optimal k>0 means model leads BIS:")
header = "  phase        " + "".join(f"k={k:>2}  " for k in KS) + " best_k"
print(header)
maes, bk = best(serr_all, cnt_all)
print("  ALL          " + "".join(f"{maes[k]:5.2f} " for k in KS) + f"   {bk}")
for p in range(4):
    if sum(cnt[p].values()) < 200: continue
    maes, bk = best(serr[p], cnt[p])
    row = "".join((f"{maes[k]:5.2f} " if cnt[p][k] else "  -   ") for k in KS)
    print(f"  {NAMES[p]:<12} " + row + f"   {bk}")
print("\n(k in steps ~ seconds; stride 1s. label_lag_sec=15 already applied upstream.)")
