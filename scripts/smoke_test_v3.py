"""
Quick smoke test for AnesthesiaNetV3 training path changes:
  - Distillation BYOL EMA teacher update
  - use_vital flag (skip VitalEncoder when lambda=0)
  - Variance regularization
  - Updated loss defaults

Runs 3 synthetic training steps (Phase 3 with vitals disabled, matching v11 config).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from src.models.anesthesia_net_v3 import AnesthesiaNetV3
from src.training.loss_v3 import MultiTaskLossV3

torch.manual_seed(42)

# Load v11 config
with open("configs/pipeline_v11.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Build model
model = AnesthesiaNetV3(
    n_eeg_channels=2,
    window_samples=512,
    feature_dim=28,
    d_model=128,
    gru_layers=2,
    gru_dropout=0.15,
    lnn_backend="gru",
    sqi_inertia_threshold=0.5,
    bsr_layer=True,
    grad_checkpoint=False,
    pk_hidden=64,
    vital_hidden=64,
    d_proj=64,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

# Build loss criterion (matching v11 config)
tcfg = cfg["training"]
criterion = MultiTaskLossV3(
    lambda_bis=tcfg["lambda_bis"],
    lambda_phase=tcfg["lambda_phase"],
    lambda_stim=tcfg["lambda_stim"],
    lambda_pkd=tcfg["lambda_pkd"],
    lambda_vitald=tcfg["lambda_vitald"],
    lambda_distill_pk=tcfg["lambda_distill_pk"],
    lambda_distill_vital=tcfg["lambda_distill_vital"],
    lambda_trans=tcfg["lambda_trans"],
    phase2_start_epoch=tcfg["phase2_start_epoch"],
    phase3_start_epoch=tcfg["phase3_start_epoch"],
    stim_warmup_epochs=tcfg.get("stim_warmup_epochs", 5),
    phase3_warmup_epochs=tcfg.get("phase3_warmup_epochs", 5),
    huber_delta=tcfg.get("huber_delta", 0.10),
    focal_alpha=tcfg.get("focal_alpha", 0.5),
    focal_gamma=tcfg.get("focal_gamma", 2.0),
    stim_pos_weight=tcfg.get("stim_pos_weight", 15.0),
)
print(f"Criterion: vitals_disabled=lambda_vitald={tcfg['lambda_vitald']}, lambda_distill_vital={tcfg['lambda_distill_vital']}")

# Synthetic batch (B=4, T=16)
B, T = 4, 16
wave = torch.randn(B, T, 2, 512)
features = torch.randn(B, T, 28)
sqi = torch.rand(B, T, 2).clamp(0.3, 1.0)
drug_ce = torch.randn(B, T, 6)
vitals = torch.randn(B, T, 5)
label_bis = torch.rand(B, T)

# Phase labels (integer 0-3)
phase_labels = torch.randint(0, 4, (B, T))
stim_labels = torch.zeros(B, T)
stim_labels[:, -4:] = 1  # some stimulation

sqi_mean = sqi.mean(-1)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("\n=== Step 1: Phase 3 (epoch 61), use_vital=False (v11 default) ===")
model.train()
optimizer.zero_grad()

out = model(wave, features, sqi, drug_ce=drug_ce, vitals=vitals, use_vital=False)

# Verify vital branch was skipped
assert "bis_vitald" not in out, "FAIL: bis_vitald should NOT be in output when use_vital=False"
assert "loss_distill_vital" not in out, "FAIL: loss_distill_vital should NOT be in output when use_vital=False"
print("  use_vital=False: VitalEncoder correctly skipped")

losses = criterion(
    pred_bis=out["pred_bis"],
    phase_logits=out["phase_logits"],
    stim_logits=out["stim_logits"],
    label_bis=label_bis,
    phase_labels=phase_labels,
    stim_labels=stim_labels,
    sqi_mean=sqi_mean,
    epoch=61,
    bis_pkd=out.get("bis_pkd"),
    bis_vitald=out.get("bis_vitald"),
    loss_distill_pk=out.get("loss_distill_pk"),
    loss_distill_vital=out.get("loss_distill_vital"),
    reg_distill_pk=out.get("reg_distill_pk"),
    reg_distill_vital=out.get("reg_distill_vital"),
    drug_ce=drug_ce,
)

print(f"  loss={losses['loss'].item():.4f}  "
      f"bis={losses['bis_loss'].item():.4f}  "
      f"phase={losses['phase_loss'].item():.4f}  "
      f"stim={losses['stim_loss'].item():.4f}  "
      f"pkd={losses.get('pkd_loss', 0):.4f}  "
      f"dist_pk={losses.get('distill_pk_loss', 0):.4f}  "
      f"reg_pk={losses.get('reg_pk', 0):.4f}")

assert torch.isfinite(losses["loss"]), "FAIL: loss is not finite"
assert "vitald_loss" not in losses or losses.get("vitald_loss", 0) == 0, \
    "FAIL: vitald_loss should be 0 when bis_vitald is None"

losses["loss"].backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"  grad_norm={grad_norm:.4f}")
optimizer.step()

# EMA update (as trainer does)
model.distill.update_teacher_ema()
print("  EMA teacher update OK")

print("\n=== Step 2: Phase 3 (epoch 61), use_vital=True (test VitalEncoder path) ===")
optimizer.zero_grad()

out2 = model(wave, features, sqi, drug_ce=drug_ce, vitals=vitals, use_vital=True)

assert "bis_vitald" in out2, "FAIL: bis_vitald should be in output when use_vital=True"
assert "loss_distill_vital" in out2, "FAIL: loss_distill_vital should be in output when use_vital=True"
print("  use_vital=True: VitalEncoder correctly enabled")

losses2 = criterion(
    pred_bis=out2["pred_bis"],
    phase_logits=out2["phase_logits"],
    stim_logits=out2["stim_logits"],
    label_bis=label_bis,
    phase_labels=phase_labels,
    stim_labels=stim_labels,
    sqi_mean=sqi_mean,
    epoch=61,
    bis_pkd=out2.get("bis_pkd"),
    bis_vitald=out2.get("bis_vitald"),
    loss_distill_pk=out2.get("loss_distill_pk"),
    loss_distill_vital=out2.get("loss_distill_vital"),
    reg_distill_pk=out2.get("reg_distill_pk"),
    reg_distill_vital=out2.get("reg_distill_vital"),
    drug_ce=drug_ce,
)

print(f"  loss={losses2['loss'].item():.4f}  "
      f"dist_vital={losses2.get('distill_vital_loss', 0):.4f}  "
      f"vitald={losses2.get('vitald_loss', 0):.4f}  "
      f"reg_vital={losses2.get('reg_vital', 0):.4f}")

assert torch.isfinite(losses2["loss"]), "FAIL: loss (use_vital=True) is not finite"

losses2["loss"].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
model.distill.update_teacher_ema()

print("\n=== Step 3: Phase 1 (epoch 1), inference path (drug_ce=None) ===")
optimizer.zero_grad()
out3 = model(wave, features, sqi)  # no drug_ce, no vitals

assert "bis_pkd" not in out3, "FAIL: bis_pkd should not be in inference output"
assert "bis_vitald" not in out3, "FAIL: bis_vitald should not be in inference output"
print("  Inference: multi-modal branches correctly skipped")

losses3 = criterion(
    pred_bis=out3["pred_bis"],
    phase_logits=out3["phase_logits"],
    stim_logits=out3["stim_logits"],
    label_bis=label_bis,
    phase_labels=phase_labels,
    stim_labels=stim_labels,
    sqi_mean=sqi_mean,
    epoch=1,
)
print(f"  loss={losses3['loss'].item():.4f}  "
      f"bis={losses3['bis_loss'].item():.4f}  "
      f"phase={losses3['phase_loss'].item():.4f}")

assert torch.isfinite(losses3["loss"]), "FAIL: Phase 1 loss is not finite"

losses3["loss"].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()

print("\n=== ALL CHECKS PASSED ===")
print(f"Model: {n_params:,} params")
print(f"Config: lambda_vitald={tcfg['lambda_vitald']}, lambda_distill_vital={tcfg['lambda_distill_vital']}")
print("Ready for full training with configs/pipeline_v11.yaml")
