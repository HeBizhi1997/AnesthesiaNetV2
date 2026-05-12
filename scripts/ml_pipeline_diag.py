"""
Diagnostic: capture live NSM EEG → trace ML BIS inference end-to-end.
Identifies exactly WHERE the low BIS originates.
"""
import sys, time, math, struct
import numpy as np
from pathlib import Path
import serial

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "EEGMonitor" / "EEGProcessingService"))

from collections import deque
from scipy.signal import butter, sosfiltfilt, welch, resample_poly

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM8"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

# ── Collect enough NSM packets for ~4 seconds of EEG ──
ser = serial.Serial(PORT, BAUD, timeout=5.0)
print(f"Connected to {PORT}. Collecting EEG for analysis (15 sec)...")

buf = bytearray()
t0 = time.time()
last_report = t0
while time.time() - t0 < 35:  # need ~30+ sec for 25+ packets at ~1 pkt/s
    n = ser.in_waiting or 1
    chunk = ser.read(n)
    if not chunk:
        time.sleep(0.1)
        continue
    buf.extend(chunk)
    now = time.time()
    if now - last_report > 5:
        print(f"  ... {len(buf)} bytes received ({time.time()-t0:.0f}s)")
        last_report = now
ser.close()
print(f"\nTotal captured: {len(buf)} bytes in {time.time()-t0:.0f}s")

# Parse NSM packets (live format: 353 bytes) — sliding window search for 0x80
PACKET_SIZE = 353
eeg_samples = []
csi_vals = []
pos = 0
while pos + PACKET_SIZE <= len(buf):
    if buf[pos] == 0x80:
        pkt = buf[pos:pos+PACKET_SIZE]
        length = pkt[1] | (pkt[2] << 8)
        if length == 351:
            for i in range(22, 122):
                eeg_samples.append(pkt[i] if pkt[i] < 128 else pkt[i] - 256)
            csi_vals.append(pkt[13])
            pos += PACKET_SIZE
            continue
    pos += 1

print(f"Captured {len(eeg_samples)} EEG samples ({len(eeg_samples)/100:.1f} sec)")
if csi_vals:
    print(f"NSM CSI values: {csi_vals}")

if len(eeg_samples) < 512:
    print(f"ERROR: Need at least 512 samples, got {len(eeg_samples)}")
    sys.exit(1)

eeg = np.array(eeg_samples[:2048], dtype=np.float64)
print(f"\n{'='*60}")
print("1. RAW EEG STATISTICS")
print(f"   samples: {len(eeg)}")
print(f"   mean (DC): {eeg.mean():.2f} uV")
print(f"   std: {eeg.std():.2f} uV")
print(f"   min: {eeg.min():.0f} uV  max: {eeg.max():.0f} uV")
print(f"   peak-to-peak: {eeg.max() - eeg.min():.0f} uV")
print(f"   zero-crossings: {sum(1 for i in range(1,len(eeg)) if (eeg[i]>=0) != (eeg[i-1]>=0))}")

# ── Sample rate estimation ──
if len(eeg_samples) >= 200:
    # Autocorrelation to find fundamental period
    eeg_ac = eeg_samples[:500] if len(eeg_samples) >= 500 else eeg_samples
    eeg_ac = np.array(eeg_ac) - np.mean(eeg_ac)
    # Count packets per second from capture time
    est_rate = len(eeg_samples) / (time.time() - t0)
    print(f"\n   Estimated sample rate: {est_rate:.0f} Hz ({len(eeg_samples)} samples / {time.time()-t0:.1f}s)")

# ── Apply the SAME preprocessing the pipeline uses ──
print(f"\n{'='*60}")
print("2. PREPROCESSOR PATH (matching eeg_preprocessor.py)")

# Determine actual sample rate
fs = int(round(len(eeg_samples) / (time.time() - t0)))
fs = max(50, min(300, fs))
print(f"   Using fs={fs} Hz")

# Bandpass 0.5-47 Hz
nyq = fs / 2.0
try:
    sos = butter(4, [0.5/nyq, min(47.0, nyq*0.99)/nyq], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, eeg)
    print(f"   After bandpass 0.5-47Hz: std={filtered.std():.2f} uV, min={filtered.min():.0f}, max={filtered.max():.0f}")
except Exception as e:
    print(f"   Bandpass error: {e}")
    filtered = eeg - eeg.mean()

# Notch 50 Hz
try:
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(50.0 / nyq, 30.0)
    filtered = filtfilt(b, a, filtered)
    print(f"   After 50Hz notch: std={filtered.std():.2f} uV")
except Exception as e:
    print(f"   Notch error: {e}")

# ── Band powers ──
print(f"\n{'='*60}")
print("3. BAND POWERS (Welch PSD, relative)")

nperseg = min(fs, len(filtered))
f, pxx = welch(filtered, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
total_power = float(pxx.sum()) + 1e-12

bands = {
    "delta (0.5-4)": (0.5, 4), "theta (4-8)": (4, 8),
    "alpha (8-13)": (8, 13), "beta (13-30)": (13, 30), "gamma (30-47)": (30, 47)
}
band_power_pct = {}
for name, (lo, hi) in bands.items():
    mask = (f >= lo) & (f < hi)
    pct = float(pxx[mask].sum()) / total_power * 100
    band_power_pct[name] = pct
    print(f"   {name}Hz: {pct:5.1f}%")

# Also compute as dict (matching router.py band_powers format)
bp_dict = {
    "delta": band_power_pct["delta (0.5-4)"] / 100.0,
    "theta": band_power_pct["theta (4-8)"] / 100.0,
    "alpha": band_power_pct["alpha (8-13)"] / 100.0,
    "beta":  band_power_pct["beta (13-30)"] / 100.0,
    "gamma": band_power_pct["gamma (30-47)"] / 100.0,
}

# ── Try to load model and do inference ──
print(f"\n{'='*60}")
print("4. ML MODEL INFERENCE")

try:
    import torch
    _MODEL_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_MODEL_ROOT))

    # Load model
    ckpt_path = _MODEL_ROOT / "outputs" / "checkpoints" / "v11" / "best_model_v3.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or ck.get("config") or {}
    print(f"   Model: epoch={ck['epoch']}, val_MAE={ck.get('val_mae','?')}")

    from src.models.anesthesia_net_v3 import AnesthesiaNetV3
    model = AnesthesiaNetV3.from_config(cfg)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"   Device: {device}")

    # Prepare input per BISPredictor._streaming_predict
    TARGET_FS = 128
    WIN_SAMP = 512
    N_CH = 2

    # Resample to 128 Hz
    input_fs = fs
    from math import gcd
    g = math.gcd(TARGET_FS, input_fs)
    up, down = TARGET_FS // g, input_fs // g

    # Ensure 2 channels
    n_ch = 2  # we only have 1, duplicate
    eeg_2ch = np.column_stack([eeg, eeg])

    resampled = np.stack(
        [resample_poly(eeg_2ch[:, ch].astype(np.float64), up, down)
         for ch in range(N_CH)],
        axis=0
    ).astype(np.float32)
    print(f"   Resampled: {eeg_2ch.shape} → {resampled.shape} (target 128Hz)")

    # Take first 512 samples
    window = resampled[:, :WIN_SAMP].astype(np.float64)
    if window.shape[1] < WIN_SAMP:
        print(f"   WARNING: only {window.shape[1]} samples after resampling, need {WIN_SAMP}")
        sys.exit(1)

    # Apply per-session amplitude normalization (matching BISPredictor)
    norm = np.array([
        max(np.median(np.abs(window[ch] - np.median(window[ch]))) / 0.6745, 0.1)
        for ch in range(N_CH)
    ], dtype=np.float64)
    print(f"   Normalization scale (MAD/0.6745): ch0={norm[0]:.3f}, ch1={norm[1]:.3f}")
    window_norm = window / norm[:, np.newaxis]
    print(f"   After normalization: std={window_norm.std():.2f}, range=[{window_norm.min():.2f}, {window_norm.max():.2f}]")

    # Apply window filters
    filter_cfg = {
        "highpass_hz": 0.5, "lowpass_hz": 47.0, "notch_hz": [60.0], "notch_q": 30.0
    }
    nyq_128 = TARGET_FS / 2.0
    sos_hp = butter(4, 0.5/nyq_128, btype="high", output="sos")
    sos_lp = butter(4, 47.0/nyq_128, btype="low", output="sos")
    for ch in range(N_CH):
        window_norm[ch] = sosfiltfilt(sos_hp, window_norm[ch])
        window_norm[ch] = sosfiltfilt(sos_lp, window_norm[ch])
    print(f"   After model filters: std={window_norm.std():.2f}")

    # Feature extraction
    from src.pipeline.context import EEGContext
    from src.pipeline.steps.features import FeatureExtractor
    from src.pipeline.steps.sqi import SQIComputer

    feat_cfg = cfg.get("features", {})
    sqi_cfg = cfg.get("sqi", {})
    feat_ext = FeatureExtractor(feat_cfg, fs=128.0)
    sqi_comp = SQIComputer(sqi_cfg)

    ctx = EEGContext(data=window_norm, fs=128.0)
    ctx = sqi_comp.process(ctx)
    ctx = feat_ext.process(ctx)

    sqi_arr = ctx.sqi.astype(np.float32)
    feat_arr = ctx.features.astype(np.float32)
    print(f"   Features: {feat_arr.shape}, SQI: {sqi_arr}")
    print(f"   Feature values: {np.array2string(feat_arr, precision=2, max_line_width=120)}")

    # Check specific features
    fpc = feat_ext.feats_per_channel  # features per channel
    for ch in range(N_CH):
        start = ch * fpc
        print(f"\n   Channel {ch} features:")
        names = ["δ_pwr", "θ_pwr", "α_pwr", "β_pwr", "γ_pwr", "PE", "SEF95", "LZC", "BSR2", "BSR5", "BSR10", "slope", "γ_emg"]
        for i, name in enumerate(names):
            if start + i < len(feat_arr):
                print(f"     {name:8s}: {feat_arr[start+i]:.4f}")
        if ch == N_CH - 1:
            print(f"     asym: {feat_arr[-2]:.4f}")
            print(f"     SQI_mean: {feat_arr[-1]:.4f}")

    # Model forward
    wave_t = torch.tensor(window_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    feat_t = torch.tensor(feat_arr).unsqueeze(0).unsqueeze(0).to(device)
    sqi_t = torch.tensor(sqi_arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(wave_t, feat_t, sqi_t, hx=None)

    bis_norm = float(out["pred_bis"].squeeze().cpu().item())
    bis = float(np.clip(bis_norm * 100.0, 0.0, 100.0))

    phase_logits = out["phase_logits"].squeeze().cpu().numpy()
    phase_probs = np.exp(phase_logits) / np.exp(phase_logits).sum()

    print(f"\n{'='*60}")
    print("5. MODEL OUTPUT")
    print(f"   pred_bis (normalized): {bis_norm:.4f}")
    print(f"   BIS (scaled 0-100): {bis:.1f}")
    print(f"   Phase logits: {phase_logits}")
    print(f"   Phase probs:  {np.array2string(phase_probs, precision=3)}")
    print(f"   Phase labels: 0=induction 1=deep 2=maint 3=emergence")

    # Heuristic BIS for comparison
    bp = bp_dict
    arousal = 0.4*bp["beta"] + 0.3*bp["gamma"] + 0.15*bp["alpha"] - 0.35*bp["delta"]
    heuristic_bis = float(np.clip(50.0 + arousal * 80.0, 0.0, 100.0))
    print(f"\n   Heuristic BIS (fallback): {heuristic_bis:.1f}")

    print(f"\n{'='*60}")
    print("6. DIAGNOSIS")

    # Check beta/delta ratio
    bdr = bp["beta"] / (bp["delta"] + 1e-6)
    print(f"   Beta/Delta ratio: {bdr:.2f} (awake > 1.0, deep anesthesia < 0.2)")

    if bis < 60 and csi_vals and all(c >= 95 for c in csi_vals):
        print(f"\n   *** ML BIS={bis:.0f} does not match awake state (CSI ~99) ***")
        print(f"   Probable causes:")
        if filtered.std() < 5.0:
            print(f"   [1] After filtering, EEG amplitude too low ({filtered.std():.1f} uV)")
            print(f"       → Model sees near-flatline → interprets as deep anesthesia")
        if bp["delta"] > 40:
            print(f"   [2] Delta-dominant spectrum ({bp['delta']:.0f}%)")
            print(f"       → Raw EEG has strong low-frequency content (DC drift?)")
        if band_power_pct["beta (13-30)"] < 10:
            print(f"   [3] Very low beta power ({bp['beta']*100:.0f}%)")
            print(f"       → Awake EEG should have 15-30% beta")

    # Compare to what model sees in training
    print(f"\n   Training data features (typical awake):")
    print(f"     δ=10-20% θ=15-25% α=20-35% β=15-25% γ=5-15%  PE=0.6-0.9  SEF95=0.4-0.7")
    print(f"   Current features:")
    print(f"     δ={bp['delta']*100:.0f}% θ={bp['theta']*100:.0f}% α={bp['alpha']*100:.0f}% β={bp['beta']*100:.0f}% γ={bp['gamma']*100:.0f}%")

except ImportError as e:
    print(f"   Cannot load model: {e}")
except Exception as e:
    import traceback
    print(f"   Error: {e}")
    traceback.print_exc()
