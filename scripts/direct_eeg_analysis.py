"""
Direct EEG: read ADS1299 from COM7, feed to v13 model, print BIS vs device status.
ADS1299 frame: [0x2A][pktNum][CH1:64×f32 LE][CH2:64×f32 LE][0x40 0x40] = 516 bytes
Sample rate: 256 Hz, UV = float × 0.02235
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import serial, numpy as np
from datetime import datetime
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly

# ── Protocol ───────────────────────────────────────────────────────
UV_SCALE = 4.5 / (24 * 8388607) * 1e6
DEV_FS = 256  # ADS1299 sample rate
SAMPLES = 64  # per frame
TARGET_FS = 128
WIN_SAMP = 512  # 4 seconds at 128 Hz

def parse_frame(buf):
    if len(buf) < 516 or buf[0] != 0x2A or buf[514] != 0x40 or buf[515] != 0x40:
        return None
    ch1 = np.array([struct.unpack_from('<f', buf, 2 + i*4)[0] for i in range(64)], dtype=np.float64) * UV_SCALE
    ch2 = np.array([struct.unpack_from('<f', buf, 258 + i*4)[0] for i in range(64)], dtype=np.float64) * UV_SCALE
    return ch1, ch2, buf[1]

# ── Filters ────────────────────────────────────────────────────────
def filt(data, fs):
    d = data.astype(np.float64)
    d = sosfiltfilt(butter(4, 0.1/(fs/2), 'high', output='sos'), d)
    b, a = iirnotch(60/(fs/2), 30); d = filtfilt(b, a, d)
    d = sosfiltfilt(butter(4, 47/(fs/2), 'low', output='sos'), d)
    return d

# ── Model ──────────────────────────────────────────────────────────
import torch, yaml
from src.models.anesthesia_net_v3 import AnesthesiaNetV3
from src.pipeline.steps.features import FeatureExtractor
from src.pipeline.steps.sqi import SQIComputer
from src.pipeline.context import EEGContext

cfg = yaml.safe_load(open('configs/pipeline_v13.yaml','r',encoding='utf-8'))
ck = torch.load('outputs/checkpoints/v13/best_model_v3.pt', map_location='cpu', weights_only=False)
model = AnesthesiaNetV3.from_config(cfg)
model.load_state_dict(ck['model_state_dict'], strict=True)
model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
feat_ext = FeatureExtractor(cfg['features'], fs=TARGET_FS)
sqi_comp = SQIComputer(cfg['sqi'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hx = None
print(f"v13 loaded: dim={feat_ext.total_feature_dim_for(2)} val_mae={ck['val_mae']:.2f} device={device}")

# ── Buffers ────────────────────────────────────────────────────────
MIRROR_CH1_TO_CH0 = True  # ch0 hardware is dead — mirror ch1 as workaround
buf_ch0, buf_ch1 = [], []
calib0, calib1 = [], []
scale = np.ones(2); calibrated = False
raw_buf = bytearray()
frame_n, bis_n = 0, 0

def run_bis():
    global hx, bis_n
    w = np.array([buf_ch0[:WIN_SAMP], buf_ch1[:WIN_SAMP]], dtype=np.float64)
    s = scale if calibrated else np.array([max(np.median(np.abs(w[ch]-np.median(w[ch])))/0.6745, 0.1) for ch in range(2)])
    w = w / s[:, np.newaxis]
    try:
        ctx = EEGContext(data=w, fs=float(TARGET_FS))
        ctx = sqi_comp.process(ctx); ctx = feat_ext.process(ctx)
        wt = torch.tensor(w.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        ft = torch.tensor(ctx.features.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        st = torch.tensor(ctx.sqi.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(): out = model(wt, ft, st, hx=hx)
        hx = out['h']
        bis = float(out['pred_bis'].squeeze().cpu().item()) * 100.0
        sqi_val = float(ctx.sqi.mean())
        bis_n += 1
        return bis, sqi_val
    except Exception as e:
        print(f"\n[ERR] {e}")
        hx = None
        return None, None

print(f"Opening COM7...")
ser = serial.Serial("COM7", 115200, timeout=0.5)
print(f"Connected. Reading...\n")
print(f"{'Time':>8s} {'Frame':>6s} {'BIS':>6s} {'ch0_uV':>8s} {'ch1_uV':>8s} {'ch0_Δ%':>7s} {'ch0_α%':>7s} {'SQI':>5s}")
print("-" * 60)

try:
    while True:
        raw_buf.extend(ser.read(max(1, ser.in_waiting)))
        while len(raw_buf) >= 516:
            hi = raw_buf.find(b'\x2a')
            if hi < 0: raw_buf.clear(); break
            if hi > 0: raw_buf = raw_buf[hi:]
            if len(raw_buf) < 516: break
            result = parse_frame(bytes(raw_buf[:516]))
            if result is None: raw_buf.pop(0); continue
            ch1, ch2, pkt = result
            raw_buf = raw_buf[516:]
            frame_n += 1

            # Filter + resample to 128 Hz
            try:
                ch1_f = filt(ch1, DEV_FS)
                ch2_f = filt(ch2, DEV_FS)
            except: continue
            ch1_128 = resample_poly(ch1_f, 1, 2).astype(np.float64)  # 256→128: 64→32
            ch2_128 = resample_poly(ch2_f, 1, 2).astype(np.float64)

            if MIRROR_CH1_TO_CH0:
                buf_ch0.extend(float(v) for v in ch2_128)  # mirror ch1→ch0
                buf_ch1.extend(float(v) for v in ch2_128)
            else:
                buf_ch0.extend(float(v) for v in ch1_128)
                buf_ch1.extend(float(v) for v in ch2_128)

            # Calibration (first 60 seconds of data)
            if not calibrated:
                if MIRROR_CH1_TO_CH0:
                    calib1.extend(ch2_128)  # only calibrate on ch1 (good channel)
                else:
                    calib0.extend(ch1_128); calib1.extend(ch2_128)
                if len(calib1) >= 60 * TARGET_FS:
                    arr = np.array(calib1[:60*TARGET_FS]); mad = np.median(np.abs(arr-np.median(arr)))
                    scale[:] = max(mad/0.6745, 0.1)  # both channels use same scale
                    calibrated = True; calib0.clear(); calib1.clear()
                    print(f"[CALIB] σ={scale[0]:.1f} uV (mirror mode — both channels use same scale)")

            # Run BIS every 1 second (128 samples at 128 Hz)
            while len(buf_ch0) >= WIN_SAMP:
                bis, sqi = run_bis()
                ts = datetime.now().strftime('%H:%M:%S')
                if bis is not None:
                    ch0_rms = np.std(np.array(buf_ch1[:WIN_SAMP]))  # ch1 RMS (only good channel)
                    ch1_rms = ch0_rms if MIRROR_CH1_TO_CH0 else np.std(np.array(buf_ch0[:WIN_SAMP]))
                    from scipy.signal import welch
                    fx, px = welch(np.array(buf_ch1[:WIN_SAMP]), fs=TARGET_FS, nperseg=256)
                    total = px.sum() + 1e-12
                    delta_pct = px[(fx>=0.5)&(fx<4)].sum()/total*100
                    alpha_pct = px[(fx>=8)&(fx<13)].sum()/total*100
                    print(f"{ts} {frame_n:>6d} {bis:>6.1f} {ch0_rms:>8.1f} {ch1_rms:>8.1f} {delta_pct:>6.1f}% {alpha_pct:>6.1f}% {sqi:>5.1f}")
                # Slide 1 second
                buf_ch0 = buf_ch0[TARGET_FS:]
                buf_ch1 = buf_ch1[TARGET_FS:]

except KeyboardInterrupt:
    print(f"\nDone. {bis_n} BIS predictions from {frame_n} frames")
finally:
    ser.close()
