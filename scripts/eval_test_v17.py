"""
Definitive held-out TEST-set evaluation of a v3 checkpoint (default: v17 best).
The test split was never used for model selection, so this is the unbiased number.
Matches trainer_v3.val_epoch methodology: 15-step causal smoothing for BIS MAE.

Usage: python scripts/eval_test_v17.py [ckpt] [batch]
"""
from __future__ import annotations
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.anesthesia_net_v3 import AnesthesiaNetV3
from src.data.dataset import _filter_cases_by_std
from src.data.dataset_v3 import SequenceDatasetV3
from src.training.trainer_v3 import (
    _causal_rolling_mean, _auroc_numpy, _spearman_corr, _prediction_probability_pk)

H5    = "outputs/preprocessed/dataset_v3_v13.h5"
CKPT  = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/v17/best_model_v3.pt"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NAMES = ["pre_op", "induction", "maintenance", "recovery"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ── reproduce the exact TEST split (first n_test after seed-42 shuffle) ────────
with h5py.File(H5, "r") as f:
    all_cases = sorted(f.keys())
all_cases = _filter_cases_by_std(H5, all_cases, 10.0, 90.0)
rng = random.Random(42); rng.shuffle(all_cases)
n = len(all_cases); n_test = max(1, int(n*0.10))
test_ids = all_cases[:n_test]
print(f"TEST cases: {len(test_ids)} (held out, never used for selection)")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = AnesthesiaNetV3.from_config(ck["cfg"])
model.load_state_dict(ck["model_state_dict"], strict=True)
model.eval().to(dev)
print(f"ckpt: {CKPT}  ep{ck['epoch']}  saved_val_mae={ck['val_mae']:.2f}  "
      f"uncertainty={model.bis_uncertainty}")

ds = SequenceDatasetV3(H5, test_ids, seq_len=300, seq_stride=150,
                       augment=False, cache_in_memory=True, phase_source="drug")
dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
print(f"TEST sequences: {len(ds)}")

P_s, L, PH, B = [], [], [], []          # smoothed pred, label, phase, b (per step)
P_raw = []
pred_ph_all, true_ph_all = [], []
stim_sc, stim_lb = [], []
amp = torch.cuda.is_available()
with torch.no_grad():
    for batch in dl:
        w = batch["wave"].to(dev); fe = batch["features"].to(dev); sq = batch["sqi"].to(dev)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=amp):
            out = model(w, fe, sq)
        pred = out["pred_bis"][..., 0].float().cpu().numpy()       # (B,T)
        lab  = batch["label_seq"].numpy()                          # (B,T)
        ph   = batch["phases"].numpy()
        for b in range(pred.shape[0]):
            P_s.append(_causal_rolling_mean(pred[b], 15) * 100.0)
            P_raw.append(pred[b] * 100.0)
            L.append(lab[b] * 100.0); PH.append(ph[b])
        pred_ph_all.append(out["phase_logits"].argmax(-1).cpu().numpy().ravel())
        true_ph_all.append(batch["phases"].numpy().ravel())
        stim_sc.append(torch.sigmoid(out["stim_logits"][..., 0]).float().cpu().numpy().ravel())
        stim_lb.append(batch["stim_cv"].numpy().ravel())
        if model.bis_uncertainty and "pred_logvar" in out:
            for b in range(pred.shape[0]):
                B.append(np.exp(out["pred_logvar"][b, :, 0].float().cpu().numpy()) * 100.0)

ps = np.concatenate(P_s); lab = np.concatenate(L); ph = np.concatenate(PH)
praw = np.concatenate(P_raw)

mae = float(np.abs(ps - lab).mean())
print(f"\n=== TEST RESULTS ({len(ps):,} timesteps) ===")
print(f"BIS MAE (15-step smoothed) : {mae:.2f}")
print(f"BIS MAE (raw per-step)     : {float(np.abs(praw-lab).mean()):.2f}")
for p in range(4):
    m = ph == p
    if m.sum() >= 500:
        print(f"  {NAMES[p]:<12} MAE={np.abs(ps[m]-lab[m]).mean():5.2f}  (n={m.sum():,})")

pph = np.concatenate(pred_ph_all); tph = np.concatenate(true_ph_all)
print(f"Phase accuracy             : {(pph==tph).mean()*100:.1f}%")
ss = np.concatenate(stim_sc); sl = np.concatenate(stim_lb)
print(f"Stim AUROC                 : {_auroc_numpy(ss, sl):.3f}")
print(f"Spearman (trend)           : {_spearman_corr(ps, lab):.3f}")
print(f"Pk (prediction probability): {_prediction_probability_pk(ps, lab):.3f}")

if B:
    b = np.concatenate(B); err = np.abs(praw - lab)
    print(f"\nUncertainty: mean b={b.mean():.2f}  "
          f"P(|e|<2b)={(err<=2*b).mean()*100:.1f}% (theory 86.5%)  "
          f"Spearman(b,|e|)={_spearman_corr(b,err):.3f}")
