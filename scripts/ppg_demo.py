"""ppg_demo.py — read the PPG/SpO2 module (CH340, COM7, 57600 8N1), show the in-frame
HR/SpO2, extract the PPG pulse waveform from IR_Data, and compute pulse rate + HRV.

Frame (17 bytes): 0A FA | 0A 00 02 | IR(4,int32 LE) | RED(4,int32 LE) | SpO2 | HR | Status | 0B
Sampling: ~125 Hz.

Usage: python scripts/ppg_demo.py [seconds] [COMport]
"""
import sys, time
import numpy as np
import serial
from scipy.signal import butter, filtfilt, find_peaks

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
PORT = sys.argv[2] if len(sys.argv) > 2 else "COM7"
BAUD = 57600
FRAME = 17
H0, H1, TAIL = 0x0A, 0xFA, 0x0B

def parse_frames(buf, ir, red, spo2, hr, status):
    """Consume whole frames from buf (bytearray), append fields to lists. Returns bytes consumed."""
    i, n = 0, len(buf)
    while i + FRAME <= n:
        if buf[i] == H0 and buf[i+1] == H1 and buf[i+16] == TAIL:
            f = buf[i:i+FRAME]
            ir.append(int.from_bytes(f[5:9],   "little", signed=True))
            red.append(int.from_bytes(f[9:13], "little", signed=True))
            spo2.append(f[13]); hr.append(f[14]); status.append(f[15])
            i += FRAME
        else:
            i += 1
    return i

# ── capture ──────────────────────────────────────────────────────────────────
print(f"打开 {PORT} @ {BAUD} … 把手指放稳,采集 {SECS:.0f}s\n")
try:
    s = serial.Serial(PORT, BAUD, timeout=0.1)
except Exception as e:
    print(f"无法打开 {PORT}: {e}"); raise SystemExit
s.dtr = False; s.rts = False
s.reset_input_buffer()

ir, red, spo2, hr, status = [], [], [], [], []
buf = bytearray(); t0 = time.time(); next_tick = 1.0; first_shown = False
while time.time() - t0 < SECS:
    c = s.read(4096)
    if c:
        buf.extend(c)
        used = parse_frames(buf, ir, red, spo2, hr, status)
        if used: del buf[:used]
    if not first_shown and hr:
        print(f"首帧解析: IR={ir[0]}  RED={red[0]}  SpO2={spo2[0]}%  HR={hr[0]}bpm  Status={status[0]}")
        first_shown = True
    el = time.time() - t0
    if el >= next_tick:
        if hr:
            print(f"[{el:4.0f}s] 设备 HR={hr[-1]:3d} bpm  SpO2={spo2[-1]:3d}%   样本={len(ir)}")
        else:
            print(f"[{el:4.0f}s] 尚无有效帧 … (检查接线/手指)")
        next_tick += 1.0
s.close()

n = len(ir)
elapsed = time.time() - t0
if n < 50:
    print(f"\n采集到的有效帧太少({n})。确认 {PORT} 是该模组、手指放好。"); raise SystemExit

fs_meas = n / elapsed
fs = fs_meas if 80 < fs_meas < 200 else 125.0
ir = np.asarray(ir, float); red = np.asarray(red, float)
hr_dev = np.asarray(hr, float); spo2_dev = np.asarray(spo2, float)
np.savez("scripts/ppg_capture.npz", ir=ir, red=red, hr=hr_dev, spo2=spo2_dev, fs=fs)

# Analyse only the finger-SETTLED segment (device HR>0) so warm-up noise doesn't
# corrupt PR/HRV; +0.5s guard after onset.
onset = int(np.argmax(hr_dev > 0)) if np.any(hr_dev > 0) else 0
onset = min(onset + int(fs*0.5), n - 1)
ir_a, red_a = ir[onset:], red[onset:]

def bandpass(x, lo=0.5, hi=5.0):
    b, a = butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

ppg_ir  = bandpass(ir_a)
ppg_red = bandpass(red_a)
na = len(ir_a)

# ── pulse-rate + HRV from IR PPG peaks ──
prom = max(np.std(ppg_ir) * 0.3, 1.0)
peaks, _ = find_peaks(ppg_ir, distance=int(fs*0.33), prominence=prom)   # <= ~180 bpm
ibi_ms = np.diff(peaks) / fs * 1000.0
ibi_ms = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]   # physiological 30-200 bpm

hv = hr_dev[hr_dev > 0]; sv = spo2_dev[spo2_dev > 0]
print("\n" + "="*58)
print(f"采集完成: {n} 帧 / {elapsed:.1f}s  -> 实测采样率 ~ {fs_meas:.0f} Hz  (按 {fs:.0f}Hz 处理)")
print(f"分析窗: 第 {onset/fs:.1f}s 起(手指稳定后) {na} 样本")
print("="*58)
print("\n[设备直出]")
print(f"  HR   : {np.median(hv):.0f} bpm (中位)   范围 {hv.min():.0f}-{hv.max():.0f}" if len(hv) else "  HR   : 无有效读数")
print(f"  SpO2 : {np.median(sv):.0f} %  (中位)   范围 {sv.min():.0f}-{sv.max():.0f}" if len(sv) else "  SpO2 : 无有效读数")

print("\n[PPG 波形 + HRV(由 IR 算)]")
if len(ibi_ms) >= 3:
    pr   = 60000.0 / np.mean(ibi_ms)
    sdnn = np.std(ibi_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(ibi_ms) ** 2))
    pnn50 = np.mean(np.abs(np.diff(ibi_ms)) > 50) * 100
    dev_hr = np.median(hv) if len(hv) else 0
    print(f"  检出脉搏 {len(peaks)} 个,有效间期 {len(ibi_ms)} 个")
    print(f"  脉率 PR : {pr:.0f} bpm   (设备 {dev_hr:.0f})")
    print(f"  平均 IBI: {np.mean(ibi_ms):.0f} ms")
    print(f"  SDNN    : {sdnn:.1f} ms")
    print(f"  RMSSD   : {rmssd:.1f} ms")
    print(f"  pNN50   : {pnn50:.0f} %")
else:
    print(f"  有效脉搏间期不足({len(ibi_ms)}),手指放稳后重测。")

# ── perfusion index + ratio-of-ratios SpO2 (未标定,仅交叉验证) ──
pi = (np.percentile(ppg_ir, 95) - np.percentile(ppg_ir, 5)) / max(np.mean(ir_a), 1) * 100
ac_ir, dc_ir = np.std(ppg_ir),  np.mean(ir_a)
ac_red, dc_red = np.std(ppg_red), np.mean(red_a)
R = (ac_red/dc_red) / (ac_ir/dc_ir) if ac_ir > 0 and dc_red > 0 else float("nan")
spo2_calc = float(np.clip(110 - 25*R, 50, 100)) if np.isfinite(R) else float("nan")
dev_spo2 = np.median(sv) if len(sv) else 0
print("\n[附加(由 IR/RED 算)]")
print(f"  灌注指数 PI : {pi:.2f} %   (越大信号越强/接触越好)")
print(f"  比值 R      : {R:.3f}")
print(f"  SpO2 估算   : {spo2_calc:.0f} %  (110-25R, 未标定; 以设备 {dev_spo2:.0f}% 为准)")

# ── ASCII 脉搏波(最后 ~6 秒) ──
win = ppg_ir[-int(fs*6):] if na > fs*6 else ppg_ir
cols = 100
ds = np.interp(np.linspace(0, len(win)-1, cols), np.arange(len(win)), win)
ds = (ds - ds.min()) / (np.ptp(ds) + 1e-9)
blocks = "_.-=*#%@"
print("\n[PPG 脉搏波(最近 6s, IR 去基线)]")
print("  " + "".join(blocks[int(v*7)] for v in ds))

# ── 保存图 (若有 matplotlib) ──
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(na) / fs
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(t, ir_a, lw=0.6, label="IR raw (DC+AC)"); ax[0].plot(t, red_a, lw=0.6, label="RED raw")
    ax[0].legend(loc="upper right"); ax[0].set_ylabel("ADC counts"); ax[0].set_title("Raw PPG")
    ax[1].plot(t, ppg_ir, lw=0.8, color="#c0392b", label="IR PPG (0.5-5Hz)")
    ax[1].plot(peaks/fs, ppg_ir[peaks], "x", color="k", label="peaks")
    ax[1].legend(loc="upper right"); ax[1].set_ylabel("AC"); ax[1].set_xlabel("time (s)"); ax[1].set_title("Pulse wave + peaks")
    fig.tight_layout()
    out = "scripts/ppg_demo_plot.png"
    fig.savefig(out, dpi=110)
    print(f"\n波形图已保存: {out}")
except Exception as e:
    print(f"\n(未画图: {e})")
