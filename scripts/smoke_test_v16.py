"""
Smoke test for v16 changes (P0-P3), CPU-only, synthetic data:

  P0a  trend metrics: _spearman_corr, _prediction_probability_pk
  P0b  drug-derived phases: label_phases_from_drug
  P1   response distillation: distill_mode="response" → bis_teacher / mask_resp / L_resp
  P2   uncertainty: bis_uncertainty=True → pred_logvar + Laplace NLL; anticipatory lag
  P3   inference bias calibration: BISPredictor._apply_bias_calibration

Run: python scripts/smoke_test_v16.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.models.anesthesia_net_v3 import AnesthesiaNetV3
from src.training.loss_v3 import MultiTaskLossV3
from src.training.trainer_v3 import _spearman_corr, _prediction_probability_pk
from src.data.phase_labeler import label_phases_from_drug

torch.manual_seed(0)
np.random.seed(0)

FEAT_DIM = 38
B, T = 4, 24


def banner(s): print(f"\n=== {s} ===")


# ── P0a: trend metrics ────────────────────────────────────────────────────────
banner("P0a trend metrics")
x = np.linspace(0, 1, 500)
y = x + np.random.randn(500) * 0.05
sp = _spearman_corr(y, x)
pk = _prediction_probability_pk(y, x)
print(f"  Spearman(noisy monotone) = {sp:.3f} (expect >0.9)")
print(f"  Pk(noisy monotone)       = {pk:.3f} (expect >0.9)")
assert sp > 0.9 and pk > 0.9
# reversed → Pk < 0.5
pk_rev = _prediction_probability_pk(-y, x)
print(f"  Pk(reversed)             = {pk_rev:.3f} (expect <0.5)")
assert pk_rev < 0.5
print("  OK")


# ── P0b: drug-derived phases ──────────────────────────────────────────────────
banner("P0b drug-derived phases")
N = 400
prop = np.zeros(N)
prop[30:80] = np.linspace(0, 5, 50)     # induction ramp
prop[80:330] = 5.0                       # maintenance plateau
prop[330:] = np.linspace(5, 0, N - 330)  # recovery decay
drug_ce = np.zeros((N, 6), dtype=np.float32)
drug_ce[:, 0] = prop
mask_drug = np.ones(N, dtype=np.float32)
phases = label_phases_from_drug(drug_ce, mask_drug)
assert phases is not None, "drug phases should be derivable"
present = set(np.unique(phases).tolist())
print(f"  phases present = {sorted(present)} (0=pre 1=ind 2=maint 3=rec)")
print(f"  counts = {np.bincount(phases, minlength=4).tolist()}")
assert {0, 1, 2, 3}.issubset(present), "expect all four phases on synthetic ramp/plateau/decay"
# insufficient drug → None (fallback path)
assert label_phases_from_drug(np.zeros((N, 6), np.float32), np.zeros(N, np.float32)) is None
print("  OK")


# ── build v16 model + loss ────────────────────────────────────────────────────
banner("Build v16 model (response distill + uncertainty)")
model = AnesthesiaNetV3(
    n_eeg_channels=2, window_samples=512, feature_dim=FEAT_DIM,
    d_model=128, gru_layers=2, gru_dropout=0.15, lnn_backend="gru",
    bsr_layer=True, grad_checkpoint=False, bis_head="shared", norm_type="batch",
    enable_multimodal=True, distill_mode="response", bis_uncertainty=True,
    pk_hidden=64, vital_hidden=64,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"  params = {n_params:,}  distill={model.distill_mode}  unc={model.bis_uncertainty}")
assert model.distill is None, "response mode must NOT build BYOL CrossModalDistillation"
assert model.logvar_head is not None

criterion = MultiTaskLossV3(
    lambda_bis=2.0, lambda_phase=0.10, lambda_stim=0.15,
    lambda_pkd=0.4, lambda_vitald=0.4, lambda_resp=0.3,
    lambda_distill_pk=0.0, lambda_distill_vital=0.0, lambda_trans=0.2,
    distill_mode="response", bis_uncertainty=True, bis_anticipate_steps=0,
    bis_transition_weight=3.0, huber_delta=0.05,
    phase2_start_epoch=1, phase3_start_epoch=16,
)

wave = torch.randn(B, T, 2, 512)
features = torch.randn(B, T, FEAT_DIM)
sqi = torch.rand(B, T, 2).clamp(0.3, 1.0)
drug_t = torch.randn(B, T, 6)
vitals = torch.randn(B, T, 5)
label_bis = torch.rand(B, T)
phase_labels = torch.randint(0, 4, (B, T))
stim_labels = torch.zeros(B, T); stim_labels[:, -4:] = 1
mask_drug_t = torch.ones(B, T)
mask_vital_t = torch.ones(B, T)
sqi_mean = sqi.mean(-1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)


def run_step(epoch, use_mm, anticipate=0):
    criterion.bis_anticipate_steps = anticipate
    model.train(); opt.zero_grad()
    if use_mm:
        out = model(wave, features, sqi, drug_ce=drug_t, vitals=vitals,
                    mask_drug=mask_drug_t, mask_vital=mask_vital_t, use_vital=True)
    else:
        out = model(wave, features, sqi)
    losses = criterion(
        pred_bis=out["pred_bis"], phase_logits=out["phase_logits"],
        stim_logits=out["stim_logits"], label_bis=label_bis,
        phase_labels=phase_labels, stim_labels=stim_labels, sqi_mean=sqi_mean,
        epoch=epoch,
        pred_logvar=out.get("pred_logvar"),
        bis_teacher=out.get("bis_teacher"), mask_resp=out.get("mask_resp"),
        bis_pkd=out.get("bis_pkd"), bis_vitald=out.get("bis_vitald"),
        mask_drug=mask_drug_t if use_mm else None,
        mask_vital=mask_vital_t if use_mm else None,
        drug_ce=drug_t if use_mm else None,
    )
    assert torch.isfinite(losses["loss"]), "loss not finite"
    losses["loss"].backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return out, losses, float(gn)


banner("P1+P2 Phase-3 forward (response + uncertainty)")
out, losses, gn = run_step(epoch=16, use_mm=True)
assert "pred_logvar" in out, "uncertainty head must emit pred_logvar"
assert "bis_teacher" in out and "mask_resp" in out, "response mode must emit teacher"
assert "loss_distill_pk" not in out, "BYOL distill must be absent in response mode"
print(f"  loss={losses['loss']:.4f}  bis={losses['bis_loss']:.4f}  "
      f"pkd={losses['pkd_loss']:.4f}  vitald={losses['vitald_loss']:.4f}  "
      f"resp={losses['resp_loss']:.4f}  grad_norm={gn:.3f}")
assert losses["resp_loss"].item() >= 0
print("  OK")

banner("P2 anticipatory lag (k=5)")
out2, losses2, gn2 = run_step(epoch=16, use_mm=True, anticipate=5)
print(f"  loss={losses2['loss']:.4f}  bis={losses2['bis_loss']:.4f}  grad_norm={gn2:.3f}")
assert torch.isfinite(losses2["loss"])
print("  OK")

banner("Phase-1 inference path (EEG-only, uncertainty on)")
out3, losses3, gn3 = run_step(epoch=1, use_mm=False, anticipate=0)
assert "pred_logvar" in out3
assert "bis_teacher" not in out3 and "bis_pkd" not in out3
print(f"  loss={losses3['loss']:.4f}  bis={losses3['bis_loss']:.4f}  grad_norm={gn3:.3f}")
print("  OK")


# ── P3: inference bias calibration logic (no full model needed) ────────────────
banner("P3 awake-anchor bias calibration")


class _Stub:
    pass


from EEGMonitor.EEGProcessingService.models.bis_predictor import BISPredictor
stub = _Stub()
# bind the unbound method with the needed attributes
stub._bias_cal_enabled = True
stub._bias_target_awake = 95.0
stub._bias_window_n = 10
stub._bias_clamp = 15.0
stub._bias_min_awake = 75.0
stub._bias = 0.0
stub._bias_locked = False
stub._cal_preds = []
apply = BISPredictor._apply_bias_calibration.__get__(stub, _Stub)
# awake start: model predicts ~88 → bias should be +7
outs = [apply(88.0) for _ in range(12)]
print(f"  awake-start bias = {stub._bias:+.1f} (expect about +7, clamped to +15)")
assert abs(stub._bias - 7.0) < 1e-6
assert abs(outs[-1] - 95.0) < 1e-6
# non-awake start: model predicts ~40 → skip calibration (bias 0)
stub2 = _Stub()
for k, v in dict(_bias_cal_enabled=True, _bias_target_awake=95.0, _bias_window_n=10,
                 _bias_clamp=15.0, _bias_min_awake=75.0, _bias=0.0,
                 _bias_locked=False, _cal_preds=[]).items():
    setattr(stub2, k, v)
apply2 = BISPredictor._apply_bias_calibration.__get__(stub2, _Stub)
for _ in range(12):
    apply2(40.0)
print(f"  anesthetized-start bias = {stub2._bias:+.1f} (expect 0)")
assert stub2._bias == 0.0
print("  OK")

print("\n=== ALL v16 SMOKE CHECKS PASSED ===")
