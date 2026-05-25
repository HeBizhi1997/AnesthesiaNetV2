"""
Comprehensive EEG session analysis.
Usage: python scripts/analyze_session.py [session_dir]
"""

import json
import sys
import struct
from pathlib import Path
from collections import defaultdict
import numpy as np

SESSION_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\Users\admin\Documents\EEGMonitor\Sessions\5ea1ee1f-af16-4238-86a2-0c75e6f3df09"
)

# ── Load session metadata ────────────────────────────────────────────────
with open(SESSION_DIR / "session.json") as f:
    session = json.load(f)

print("=" * 100)
print(f"Session: {session['SessionId']}")
print(f"Patient: {session['PatientId']}  Surgery: {session['SurgeryType']}")
print(f"Start: {session['StartTime']}  End: {session['EndTime']}")
print(f"Duration: {session['Duration']}  Rate: {session['SampleRate']} Hz  Ch: {session['ChannelCount']}")
print("=" * 100)

# ── Load processed.jsonl ─────────────────────────────────────────────────
print("\nLoading processed.jsonl...")
results = []
with open(SESSION_DIR / "processed.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

n = len(results)
print(f"Loaded {n} epochs (~{n / 3600:.1f} hours at 1 epoch/sec)")

if n == 0:
    print("ERROR: No data in processed.jsonl")
    sys.exit(1)

# Extract arrays
bis_vals = []
sqi_vals = []
delta_vals = []
theta_vals = []
alpha_vals = []
beta_vals = []
gamma_vals = []
clip_vals = []
dc_vals = []
amplitude_vals = []
dominant_hz_vals = []
tonal_vals = []
spindle_vals = []
sleep_flags = []
hw_saturated = []
dev_delta_vals = []
dev_delta_disc = []
filtered_chunks = 0  # chunks where device band powers were used

for r in results:
    bis = r.get("bis")
    bis_vals.append(bis if bis is not None else float("nan"))
    sqi_vals.append(r.get("sqi", float("nan")))
    delta_vals.append(r.get("delta_power", 0) * 100)
    theta_vals.append(r.get("theta_power", 0) * 100)
    alpha_vals.append(r.get("alpha_power", 0) * 100)
    beta_vals.append(r.get("beta_power", 0) * 100)
    gamma_vals.append(r.get("gamma_power", 0) * 100)
    clip_vals.append(r.get("hw_clipping_pct", 0))
    dc_vals.append(r.get("hw_dc_offset_uv", 0))
    amplitude_vals.append(r.get("eeg_amplitude_uv", 0))
    dominant_hz_vals.append(r.get("eeg_dominant_hz", 0))
    tonal_vals.append(r.get("eeg_tonal_ratio", 0))
    spindle_vals.append(r.get("spindle_density", 0))
    sleep_flags.append(r.get("is_likely_sleep", False))
    hw_saturated.append(r.get("hw_is_saturated", False))
    dev_delta_vals.append(r.get("device_delta_ratio"))
    dev_delta_disc.append(r.get("device_delta_discrepancy"))

bis_arr = np.array(bis_vals)
sqi_arr = np.array(sqi_vals)
delta_arr = np.array(delta_vals)
theta_arr = np.array(theta_vals)
alpha_arr = np.array(alpha_vals)
beta_arr = np.array(beta_vals)
gamma_arr = np.array(gamma_vals)
clip_arr = np.array(clip_vals)
dc_arr = np.array(dc_vals)
amp_arr = np.array(amplitude_vals)
dom_arr = np.array(dominant_hz_vals)
tonal_arr = np.array(tonal_vals)
spindle_arr = np.array(spindle_vals)
sleep_arr = np.array(sleep_flags)
sat_arr = np.array(hw_saturated)

# Time axis (seconds from start)
t_sec = np.arange(n)
t_min = t_sec / 60.0
t_hr = t_sec / 3600.0

# ── Overall Statistics ───────────────────────────────────────────────────
print("\n" + "=" * 100)
print("OVERALL STATISTICS")
print("=" * 100)

def stats(arr, name, unit="", is_pct=False):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        print(f"  {name:<25}: NO DATA")
        return
    fmt = "5.1f" if is_pct else ".1f"
    print(f"  {name:<25}: mean={np.mean(valid):{fmt}}{unit}  "
          f"median={np.median(valid):{fmt}}{unit}  "
          f"std={np.std(valid):{fmt}}{unit}  "
          f"min={np.min(valid):{fmt}}{unit}  "
          f"max={np.max(valid):{fmt}}{unit}")

stats(bis_arr, "BIS (ML-inferred)")
stats(delta_arr, "Delta Power", "%", True)
stats(theta_arr, "Theta Power", "%", True)
stats(alpha_arr, "Alpha Power", "%", True)
stats(beta_arr, "Beta Power", "%", True)
stats(gamma_arr, "Gamma Power", "%", True)
stats(clip_arr, "Clipping", "%", True)
stats(dc_arr, "DC Offset", "uV")
stats(amp_arr, "EEG Amplitude", "uV")
stats(dom_arr, "Dominant Frequency", "Hz")
stats(tonal_arr, "Tonal Ratio", "", True)
stats(spindle_arr, "Spindle Density", "/min")
stats(sqi_arr, "SQI")

# BIS availability
bis_valid = bis_arr[~np.isnan(bis_arr)]
print(f"\n  BIS available: {len(bis_valid)}/{n} epochs ({len(bis_valid)/n*100:.0f}%)")

# Hardware issues
sat_count = np.sum(sat_arr)
print(f"  ADC saturated epochs: {sat_count}/{n} ({sat_count/n*100:.1f}%)")
clip_high = np.sum(clip_arr > 5)
print(f"  Clipping >5% epochs: {clip_high}/{n} ({clip_high/n*100:.1f}%)")
clip_any = np.sum(clip_arr > 0)
print(f"  Any clipping epochs: {clip_any}/{n} ({clip_any/n*100:.1f}%)")

# Sleep detection
sleep_count = np.sum(sleep_arr)
print(f"  Sleep-flagged epochs: {sleep_count}/{n} ({sleep_count/n*100:.1f}%)")

# Device validation
dev_valid = [x for x in dev_delta_vals if x is not None]
print(f"  Device delta cross-ref epochs: {len(dev_valid)}/{n}")

# ── Period Analysis: Split into 10-minute windows ────────────────────────
print("\n" + "=" * 100)
print("10-MINUTE WINDOW ANALYSIS")
print("=" * 100)
print(f"{'Window':>8} {'Time':>6} {'BIS':>6} {'Delta%':>7} {'Theta%':>7} "
      f"{'Alpha%':>7} {'Beta%':>6} {'SQI':>5} {'Clip%':>6} {'Sleep%':>7} "
      f"{'Amp(uV)':>8} {'DomHz':>6}")

window_min = 10
window_epochs = window_min * 60  # ~60 epochs per minute at 1/sec
for w_start in range(0, n, window_epochs):
    w_end = min(w_start + window_epochs, n)
    win = slice(w_start, w_end)
    b = bis_arr[win]
    b_valid = b[~np.isnan(b)]

    t_str = f"{w_start/3600:.1f}-{w_end/3600:.1f}h"
    bis_str = f"{np.mean(b_valid):.0f}" if len(b_valid) > 0 else "N/A"
    print(f"  {t_str:>15}: BIS={bis_str:>4}  "
          f"d={np.mean(delta_arr[win]):5.1f}% "
          f"t={np.mean(theta_arr[win]):5.1f}% "
          f"a={np.mean(alpha_arr[win]):5.1f}% "
          f"b={np.mean(beta_arr[win]):5.1f}% "
          f"SQI={np.mean(sqi_arr[win]):4.0f} "
          f"Clip={np.mean(clip_arr[win]):5.1f}% "
          f"Sleep={np.sum(sleep_arr[win])/sleep_arr[win].size*100:5.1f}% "
          f"Amp={np.mean(amp_arr[win]):6.1f}uV "
          f"Dom={np.mean(dom_arr[win]):5.1f}Hz")

# ── Sleep / Rest Period Detection ────────────────────────────────────────
print("\n" + "=" * 100)
print("SLEEP / REST PERIOD DETECTION")
print("=" * 100)

# Criteria for sleep/rest: high delta + low beta + spindle activity + low BIS
# Smooth the spindle density with a 5-minute window
smooth_window = 5 * 60
spindle_smooth = np.convolve(spindle_arr, np.ones(smooth_window)/smooth_window, mode='same')

# Find periods with elevated spindle density (potential sleep)
spindle_threshold = 0.5  # spindles/min
sleep_periods = []
in_period = False
period_start = 0
for i in range(n):
    is_sleepy = (spindle_smooth[i] >= spindle_threshold or sleep_arr[i]) and not np.isnan(bis_arr[i])
    if is_sleepy and not in_period:
        period_start = i
        in_period = True
    elif not is_sleepy and in_period:
        if i - period_start >= 60:  # at least 1 minute
            sleep_periods.append((period_start, i))
        in_period = False
if in_period and n - period_start >= 60:
    sleep_periods.append((period_start, n))

if sleep_periods:
    print(f"Found {len(sleep_periods)} potential sleep/rest periods:")
    for idx, (s, e) in enumerate(sleep_periods):
        dur_min = (e - s) / 60.0
        win = slice(s, e)
        b = bis_arr[win]
        b_valid = b[~np.isnan(b)]
        bis_mean = np.mean(b_valid) if len(b_valid) > 0 else float("nan")
        print(f"\n  Period {idx+1}: {s/3600:.2f}h → {e/3600:.2f}h ({dur_min:.0f} min)")
        print(f"    BIS: {bis_mean:.0f}  Delta: {np.mean(delta_arr[win]):.1f}%  "
              f"Theta: {np.mean(theta_arr[win]):.1f}%  Alpha: {np.mean(alpha_arr[win]):.1f}%  "
              f"Beta: {np.mean(beta_arr[win]):.1f}%")
        print(f"    Spindle: {np.mean(spindle_arr[win]):.2f}/min  "
              f"Sleep flags: {np.sum(sleep_arr[win])}/{e-s}  "
              f"Amplitude: {np.mean(amp_arr[win]):.1f}uV  "
              f"Dom Hz: {np.mean(dom_arr[win]):.1f}Hz  "
              f"Clipping: {np.mean(clip_arr[win]):.1f}%  "
              f"SQI: {np.mean(sqi_arr[win]):.0f}")
else:
    print("No significant sleep periods detected.")

# Also analyze non-sleep periods for contrast
if sleep_periods:
    awake_mask = np.ones(n, dtype=bool)
    for s, e in sleep_periods:
        awake_mask[s:e] = False
    awake_idx = np.where(awake_mask)[0]
    if len(awake_idx) > 0:
        b = bis_arr[awake_idx]
        b_valid = b[~np.isnan(b)]
        print(f"\n  AWAKE periods (contrast):")
        print(f"    Duration: {len(awake_idx)/3600:.2f}h")
        print(f"    BIS: {np.mean(b_valid):.0f}" if len(b_valid) > 0 else "    BIS: N/A")
        print(f"    Delta: {np.mean(delta_arr[awake_idx]):.1f}%  "
              f"Theta: {np.mean(theta_arr[awake_idx]):.1f}%  "
              f"Alpha: {np.mean(alpha_arr[awake_idx]):.1f}%  "
              f"Beta: {np.mean(beta_arr[awake_idx]):.1f}%")
        print(f"    Spindle: {np.mean(spindle_arr[awake_idx]):.2f}/min  "
              f"Amplitude: {np.mean(amp_arr[awake_idx]):.1f}uV  "
              f"Clipping: {np.mean(clip_arr[awake_idx]):.1f}%")

# ── Signal Quality Assessment ────────────────────────────────────────────
print("\n" + "=" * 100)
print("SIGNAL QUALITY ASSESSMENT")
print("=" * 100)

sq_poor = np.sum(sqi_arr < 30)
sq_good = np.sum(sqi_arr >= 80)
print(f"  Excellent (SQI >= 80): {sq_good}/{n} epochs ({sq_good/n*100:.1f}%)")
print(f"  Poor (SQI < 30):      {sq_poor}/{n} epochs ({sq_poor/n*100:.1f}%)")

# Tonal interference (>40% power in narrow band = likely electrical noise)
tonal_bad = np.sum(tonal_arr > 0.40)
print(f"  Tonal interference:   {tonal_bad}/{n} epochs ({tonal_bad/n*100:.1f}%)")

# Dominant frequency distribution
print(f"\n  Dominant frequency distribution:")
for lo, hi, label in [(0, 4, "Delta (0-4Hz)"), (4, 8, "Theta (4-8Hz)"),
                        (8, 13, "Alpha (8-13Hz)"), (13, 30, "Beta (13-30Hz)"),
                        (30, 47, "Gamma (30-47Hz)")]:
    count = np.sum((dom_arr >= lo) & (dom_arr < hi))
    print(f"    {label:<20}: {count:>6} epochs ({count/n*100:5.1f}%)")

# ── Band Power vs Time (key trend) ───────────────────────────────────────
print("\n" + "=" * 100)
print("BAND POWER TREND (hourly)")
print("=" * 100)
print(f"{'Hour':>5} {'Delta%':>8} {'Theta%':>8} {'Alpha%':>8} {'Beta%':>8} {'Gamma%':>8} {'BIS':>6} {'SQI':>5} {'Amp(uV)':>8}")
for h in range(0, int(np.ceil(n/3600))):
    win = slice(h*3600, min((h+1)*3600, n))
    b = bis_arr[win]
    b_valid = b[~np.isnan(b)]
    bis_str = f"{np.mean(b_valid):.0f}" if len(b_valid) > 0 else "N/A"
    print(f"{h:>4}h  {np.mean(delta_arr[win]):7.1f}% {np.mean(theta_arr[win]):7.1f}% "
          f"{np.mean(alpha_arr[win]):7.1f}% {np.mean(beta_arr[win]):7.1f}% "
          f"{np.mean(gamma_arr[win]):7.1f}% {bis_str:>5} "
          f"{np.mean(sqi_arr[win]):4.0f}  {np.mean(amp_arr[win]):7.1f}uV")

# ── Delta vs Beta ratio (depth indicator) ────────────────────────────────
print("\n" + "=" * 100)
print("DELTA/BETA RATIO ANALYSIS (anesthesia depth indicator)")
print("=" * 100)
# Smooth with 5-min window for trend
db_ratio = delta_arr / (beta_arr + 0.01)
db_smooth = np.convolve(db_ratio, np.ones(300)/300, mode='same')
print(f"  Overall delta/beta ratio: {np.mean(db_ratio):.1f}")
print(f"  Max delta/beta (5min smooth): {np.max(db_smooth):.1f} at {np.argmax(db_smooth)/60:.1f}min")
print(f"  Min delta/beta (5min smooth): {np.min(db_smooth):.1f} at {np.argmin(db_smooth)/60:.1f}min")

# Find periods with lowest BIS (deepest "anesthesia" / deepest sleep)
print("\n" + "=" * 100)
print("LOWEST BIS PERIODS (deepest sleep/anesthesia)")
print("=" * 100)
bis_finite = np.where(~np.isnan(bis_arr))[0]
if len(bis_finite) > 0:
    # Find the lowest BIS contiguous periods
    bis_smooth = np.convolve(np.nan_to_num(bis_arr, nan=100), np.ones(300)/300, mode='same')
    low_threshold = np.percentile(bis_smooth[bis_finite], 10)  # bottom 10%
    print(f"  Low BIS threshold (10th percentile): {low_threshold:.0f}")
    low_periods = []
    in_low = False
    low_start = 0
    for i in range(n):
        if not np.isnan(bis_arr[i]) and bis_smooth[i] < low_threshold:
            if not in_low:
                low_start = i
                in_low = True
        else:
            if in_low and i - low_start >= 30:
                low_periods.append((low_start, i))
            in_low = False
    if in_low and n - low_start >= 30:
        low_periods.append((low_start, n))

    for idx, (s, e) in enumerate(low_periods[:5]):
        dur_min = (e - s) / 60.0
        win = slice(s, e)
        print(f"  Period {idx+1}: {s/3600:.2f}h → {e/3600:.2f}h ({dur_min:.0f}min)  "
              f"BIS={np.mean(bis_arr[win]):.0f}  "
              f"d={np.mean(delta_arr[win]):.0f}%  "
              f"a={np.mean(alpha_arr[win]):.0f}%  "
              f"b={np.mean(beta_arr[win]):.0f}%  "
              f"sp={np.mean(spindle_arr[win]):.2f}/min")
else:
    print("  No BIS data available.")

print("\n" + "=" * 100)
print("Analysis complete.")
print("=" * 100)
