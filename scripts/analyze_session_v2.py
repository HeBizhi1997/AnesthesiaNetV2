"""
V2: Corrected analysis accounting for sample rate mismatch.
NSM device outputs at ~100 Hz but was configured as 200 Hz.
All frequency bands are shifted by factor of 2.
"""
import json, sys, struct
from pathlib import Path
from collections import defaultdict
import numpy as np

SESSION_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\Users\admin\Documents\EEGMonitor\Sessions\5ea1ee1f-af16-4238-86a2-0c75e6f3df09"
)

with open(SESSION_DIR / "session.json") as f:
    session = json.load(f)

print("=" * 105)
print(f"  SESSION ANALYSIS — Corrected for 100Hz (session metadata says {session['SampleRate']}Hz)")
print(f"  Patient: {session['PatientId']}  Duration: {session['Duration']}")
print(f"  Start: {session['StartTime']}  End: {session['EndTime']}")
print("=" * 105)

# Load
results = []
with open(SESSION_DIR / "processed.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except:
                pass
n = len(results)

# ── CORRECTED band mapping ───────────────────────────────────────────────
# Configured rate = 200 Hz, actual = 100 Hz → factor = 2.0
# Stored band:     actual freq range
#   delta  (0.5-4)   → 0.25-2 Hz  (sub-delta + DC residual)
#   theta  (4-8)     → 2-4 Hz     (TRUE DELTA)
#   alpha  (8-13)    → 4-6.5 Hz   (theta)
#   beta   (13-30)   → 6.5-15 Hz  (alpha + low beta)
#   gamma  (30-47)   → 15-23.5 Hz (mid beta)
#   dom Hz (4-8)     → 2-4 Hz     (true delta peak)

FACTOR = 2.0  # configured/actual

# Extract as-stored values
delta_stored = np.array([r.get("delta_power", 0) * 100 for r in results])
theta_stored = np.array([r.get("theta_power", 0) * 100 for r in results])
alpha_stored = np.array([r.get("alpha_power", 0) * 100 for r in results])
beta_stored  = np.array([r.get("beta_power", 0) * 100 for r in results])
gamma_stored = np.array([r.get("gamma_power", 0) * 100 for r in results])
bis_arr = np.array([r.get("bis") if r.get("bis") is not None else float("nan") for r in results])
sqi_arr = np.array([r.get("sqi", float("nan")) for r in results])
clip_arr = np.array([r.get("hw_clipping_pct", 0) for r in results])
dom_stored = np.array([r.get("eeg_dominant_hz", 0) for r in results])
amp_arr = np.array([r.get("eeg_amplitude_uv", 0) for r in results])
spindle_arr = np.array([r.get("spindle_density", 0) for r in results])
sleep_arr = np.array([r.get("is_likely_sleep", False) for r in results])
dc_arr = np.array([r.get("hw_dc_offset_uv", 0) for r in results])
tonal_arr = np.array([r.get("eeg_tonal_ratio", 0) for r in results])

# CORRECTED band values (remapped to true frequencies)
# sub_delta = 0.25-2 Hz (what was labeled "delta")
# true_delta = 2-4 Hz (what was labeled "theta")
# theta = 4-6.5 Hz (what was labeled "alpha")
# alpha_beta = 6.5-15 Hz (what was labeled "beta")
# mid_beta = 15-23.5 Hz (what was labeled "gamma")
sub_delta = delta_stored   # actually 0.25-2 Hz
true_delta = theta_stored  # actually 2-4 Hz
true_theta = alpha_stored  # actually 4-6.5 Hz
alpha_low_beta = beta_stored  # actually 6.5-15 Hz
mid_beta = gamma_stored    # actually 15-23.5 Hz

# True dominant frequency
dom_true = dom_stored / FACTOR

t_sec = np.arange(n)
t_min = t_sec / 60
t_hr = t_sec / 3600

# ── Overall Statistics (CORRECTED) ───────────────────────────────────────
print("\n  CORRECTED FREQUENCY BAND STATISTICS (actual 100 Hz)")
print("  " + "-" * 80)
print(f"  {'Band':<25} {'Stored As':>12} {'True Range':>18} {'Mean':>8} {'Median':>8} {'Std':>8}")
print("  " + "-" * 80)

bands_corrected = [
    ("Sub-Delta + DC residual", "delta (0.5-4)", "0.25-2 Hz", sub_delta),
    ("TRUE DELTA", "theta (4-8)", "2-4 Hz", true_delta),
    ("Theta", "alpha (8-13)", "4-6.5 Hz", true_theta),
    ("Alpha + Low Beta", "beta (13-30)", "6.5-15 Hz", alpha_low_beta),
    ("Mid Beta", "gamma (30-47)", "15-23.5 Hz", mid_beta),
]
for name, stored, true_range, arr in bands_corrected:
    print(f"  {name:<25} {stored:>12} {true_range:>18} {np.mean(arr):7.1f}% {np.median(arr):7.1f}% {np.std(arr):7.1f}%")

print(f"\n  BIS:          mean={np.nanmean(bis_arr):.1f}  median={np.nanmedian(bis_arr):.1f}  "
      f"min={np.nanmin(bis_arr):.1f}  max={np.nanmax(bis_arr):.1f}")
print(f"  True dominant freq: mean={np.mean(dom_true):.1f} Hz  median={np.median(dom_true):.1f} Hz")
print(f"  (Stored as {np.mean(dom_stored):.1f} Hz due to 2x rate mismatch)")

# ── Frequency domain corrected distribution ──────────────────────────────
print(f"\n  TRUE DOMINANT FREQUENCY DISTRIBUTION (corrected):")
for lo, hi, label in [(0, 0.5, "Sub-delta (<0.5Hz)"), (0.5, 2, "Sub-delta (0.5-2Hz)"),
                        (2, 4, "Delta (2-4Hz)"), (4, 6.5, "Theta (4-6.5Hz)"),
                        (6.5, 15, "Alpha+Beta (6.5-15Hz)"), (15, 23.5, "Beta (15-23.5Hz)")]:
    count = np.sum((dom_true >= lo) & (dom_true < hi))
    print(f"    {label:<28}: {count:>6} epochs ({count/n*100:5.1f}%)")

# ── Time-window analysis ─────────────────────────────────────────────────
print(f"\n  {'='*80}")
print(f"  15-MINUTE WINDOW ANALYSIS (corrected bands)")
print(f"  {'='*80}")
print(f"  {'Window':>10} {'BIS':>5} {'TrueD%':>7} {'Theta%':>7} {'Al+Bt%':>7} {'MidB%':>6} {'SQI':>5} {'Clip%':>6} {'Amp':>6} {'DomHz':>6}")

window_min = 15
window_epochs = int(window_min * 60 / 2.62)  # epochs per window at ~2.62s/epoch
for w_start in range(0, n, window_epochs):
    w_end = min(w_start + window_epochs, n)
    win = slice(w_start, w_end)
    b = bis_arr[win]
    b_valid = b[~np.isnan(b)]
    bis_str = f"{np.mean(b_valid):.0f}" if len(b_valid) > 0 else "N/A"
    t1 = w_start * 2.62 / 3600
    t2 = w_end * 2.62 / 3600
    print(f"  {t1:.1f}h-{t2:.1f}h: BIS={bis_str:>4}  "
          f"Δ={np.mean(true_delta[win]):5.1f}%  "
          f"θ={np.mean(true_theta[win]):5.1f}%  "
          f"α+β={np.mean(alpha_low_beta[win]):5.1f}%  "
          f"mβ={np.mean(mid_beta[win]):4.1f}%  "
          f"SQI={np.mean(sqi_arr[win]):4.0f}  "
          f"Clip={np.mean(clip_arr[win]):5.1f}%  "
          f"{np.mean(amp_arr[win]):5.1f}uV  "
          f"{np.mean(dom_true[win]):5.1f}Hz")

# ── Sleep detection (with corrected spindle interpretation) ───────────────
print(f"\n  {'='*80}")
print(f"  SLEEP / REST ANALYSIS")
print(f"  {'='*80}")

# Spindle detector uses 12-14 Hz (sigma band). At the mismatched rate,
# this corresponds to TRUE 6-7 Hz (theta!). So "spindle density" is
# actually detecting theta activity, not sleep spindles.
# True sigma (12-14 Hz) would need apparent 24-28 Hz filter.
print(f"\n  ⚠ SPINDLE DETECTOR MISMATCH: 12-14 Hz filter at 200 Hz config")
print(f"    = actual 6-7 Hz at 100 Hz true rate → detecting THETA, not spindles.")
print(f"    This explains why 99.4% of epochs were flagged as 'sleep'.")
print(f"    True sleep spindles (12-14 Hz) require reconfiguration.")
print(f"\n  Spindle density (stored): mean={np.mean(spindle_arr):.1f}/min")
print(f"  This is actually theta-band burst density, not sleep spindles.")

# ── Real sleep indicators ─────────────────────────────────────────────────
# For sleep detection with corrected bands:
# N1: theta increases, alpha decreases, slow eye movements
# N2: true sleep spindles (12-14 Hz) + K-complexes
# N3: true delta (2-4 Hz) > 20% of epoch
# REM: theta + low EMG + rapid eye movements

# With our corrected mapping:
# True delta (2-4 Hz) = what's stored as "theta"
# High "theta" (stored) + low BIS = possible N2/N3 sleep

# Find periods where stored "theta" (true delta) is elevated
true_delta_pct = theta_stored  # this is the actual delta (2-4 Hz)
delta_smooth = np.convolve(true_delta_pct.astype(float), np.ones(120)/120, mode='same')

# Deep sleep: true delta > 25% (corrected)
deep_sleep_mask = delta_smooth > 25
deep_count = np.sum(deep_sleep_mask)
print(f"\n  TRUE DELTA (2-4 Hz) > 25%: {deep_count}/{n} epochs ({deep_count/n*100:.1f}%)")
print(f"  (This is the real indicator of deep sleep / N3)")

# Find contiguous deep sleep periods
periods = []
in_period = False
start = 0
for i in range(n):
    if deep_sleep_mask[i] and not in_period:
        start = i
        in_period = True
    elif not deep_sleep_mask[i] and in_period:
        if i - start >= 20:  # at least ~1 minute
            periods.append((start, i))
        in_period = False
if in_period and n - start >= 20:
    periods.append((start, n))

if periods:
    print(f"\n  POTENTIAL DEEP SLEEP PERIODS (true delta 2-4Hz > 25%):")
    for idx, (s, e) in enumerate(periods):
        dur_min = (e - s) * 2.62 / 60
        win = slice(s, e)
        b = bis_arr[win]
        b_valid = b[~np.isnan(b)]
        t1 = s * 2.62 / 3600
        t2 = e * 2.62 / 3600
        print(f"\n  ── Period {idx+1}: {t1:.2f}h → {t2:.2f}h ({dur_min:.0f} min) ──")
        print(f"    BIS: {np.nanmean(b):.0f}  |  "
              f"True Δ (2-4Hz): {np.mean(true_delta[win]):.0f}%  "
              f"θ (4-6.5Hz): {np.mean(true_theta[win]):.0f}%  "
              f"α+β (6.5-15Hz): {np.mean(alpha_low_beta[win]):.0f}%  "
              f"Amp: {np.mean(amp_arr[win]):.0f}uV  "
              f"Dom: {np.mean(dom_true[win]):.1f}Hz")
else:
    print("\n  No deep sleep periods detected.")

# ── Signal quality ───────────────────────────────────────────────────────
print(f"\n  {'='*80}")
print(f"  SIGNAL QUALITY")
print(f"  {'='*80}")
sq_good = np.sum(sqi_arr >= 80)
sq_poor = np.sum(sqi_arr < 30)
sat_count = np.sum(np.array([r.get("hw_is_saturated", False) for r in results]))
print(f"  SQI >= 80: {sq_good}/{n} ({sq_good/n*100:.0f}%)")
print(f"  SQI < 30:  {sq_poor}/{n} ({sq_poor/n*100:.0f}%)")
print(f"  ADC saturation: {sat_count}/{n} epochs ({sat_count/n*100:.1f}%)")
print(f"  Mean clipping: {np.mean(clip_arr):.1f}%")
print(f"  Mean DC offset: {np.mean(dc_arr):.1f} uV")

# ── SUMMARY ───────────────────────────────────────────────────────────────
print(f"\n  {'='*80}")
print(f"  KEY FINDINGS & FIXES NEEDED")
print(f"  {'='*80}")
print(f"""
  1. [CRITICAL] SAMPLE RATE MISMATCH
     Session recorded with 200 Hz config, but NSM device outputs at ~100 Hz.
     ALL frequency bands are shifted by 2×.
     - stored 'delta' = true sub-delta (0.25-2 Hz)
     - stored 'theta' = TRUE DELTA (2-4 Hz)
     - stored 'alpha' = true theta (4-6.5 Hz)
     FIX: Set NSM sample rate to 100 Hz in UI (the default) or auto-detect.

  2. SPINDLE DETECTOR MISMATCH
     Sigma filter (12-14 Hz) maps to true 6-7 Hz (theta).
     "Sleep" flags were detecting theta, not spindles.
     FIX: Adjust spindle detector for actual sample rate.

  3. TRUE DELTA (2-4 Hz) LEVELS
     Mean true delta = {np.mean(true_delta):.1f}% — this is the real deep-sleep indicator.
     For an awake person, true delta should be <20%.
     Elevated true delta suggests the person may have been drowsy/resting,
     but the sample rate mismatch prevents definitive sleep staging.
""")

print("=" * 105)
print("  Analysis complete. All frequency values above are CORRECTED for 100 Hz actual rate.")
print("=" * 105)
