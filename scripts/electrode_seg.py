"""electrode_seg.py — capture one labeled segment from COM6 (CH0), save + analyze.
Usage: python scripts/electrode_seg.py <label> [seconds]"""
import sys, time, struct
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
import serial

LABEL = sys.argv[1] if len(sys.argv) > 1 else "X"
SECS  = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
FS = 250

def fr(c, w, d=b""):
    t = 8 + len(d); ln = struct.pack("<H", t); cb = c | 0x80 if w else c
    hc = ln[0] ^ ln[1] ^ 0 ^ cb; dc = 0
    for b in d: dc ^= b
    return bytes([0xA5, ln[0], ln[1], 0, cb, hc]) + d + bytes([dc, 0x5A])

s = serial.Serial("COM6", 230400, timeout=0.1); s.dtr = True; s.rts = True
s.write(fr(3,1,bytes([1]))); time.sleep(0.15)
s.write(fr(5,1,bytes([0xf4,1,0x0c,0xff,0,0]))); time.sleep(0.15)
s.write(fr(4,1,bytes([1])))
s.reset_input_buffer(); buf = bytearray(); fl = []; t0 = time.time()
while time.time() - t0 < SECS:
    c = s.read(8192)
    if c: buf.extend(c)
    while True:
        i = buf.find(0xAA)
        if i < 0:
            if len(buf) > 16384: del buf[:]
            break
        if i > 0: del buf[:i]
        if len(buf) < 3: break
        tot = buf[1] | (buf[2] << 8)
        if tot < 8 or tot > 16384:
            del buf[0]; continue
        if len(buf) < tot: break
        if buf[tot-1] != 0x55: del buf[0]; continue
        if (buf[4] & 0x7f) == 6:
            d = bytes(buf[6:tot-2]); fl.extend(struct.unpack(f"<{len(d)//4}f", d))
        del buf[:tot]
s.write(fr(4,1,bytes([0]))); s.write(fr(3,1,bytes([0]))); s.close()

a = np.array(fl); n = len(a)//8; ch0 = a[:n*8].reshape(n,8)[:,0]
np.save(f"/tmp/seg_{LABEL}.npy", ch0)
fs = len(ch0)/SECS

raw = ch0 - ch0.mean()
# broadband filtered 1-45 + 50Hz notch (zero-phase, whole segment)
nyq = fs/2
sos = butter(4, [1/nyq, 45/nyq], btype='band', output='sos')
filt = sosfiltfilt(sos, raw)
bn, an = iirnotch(50/nyq, 30); filt = filtfilt(bn, an, filt)

f, P = welch(raw, fs=fs, nperseg=int(fs*2))
tot = P[(f>=1)&(f<=60)].sum() + 1e-12
def frac(lo, hi): return P[(f>=lo)&(f<hi)].sum()/tot*100
ff, Pf = welch(filt, fs=fs, nperseg=int(fs*2))
totf = Pf[(ff>=1)&(ff<45)].sum()+1e-12
def fracf(lo,hi): return Pf[(ff>=lo)&(ff<hi)].sum()/totf*100
dom = ff[(ff>=1)&(ff<45)][np.argmax(Pf[(ff>=1)&(ff<45)])]

print(f"==== 段 {LABEL}  fs≈{fs:.0f}Hz  n={len(ch0)} ====")
print(f"原始: std={raw.std():.1f}  p2p={raw.max()-raw.min():.0f}")
print(f"原始功率分布(1-60Hz内): 50Hz工频={frac(48,52):.0f}%  低频<4Hz漂移={frac(1,4):.0f}%")
print(f"滤波后(1-45,去50): std={filt.std():.1f}  主频={dom:.1f}Hz")
print(f"  频段占比: δ={fracf(0.5,4):.0f}% θ={fracf(4,8):.0f}% α(8-13)={fracf(8,13):.0f}% "
      f"β={fracf(13,30):.0f}% γ={fracf(30,45):.0f}%")
print(f"  >>> α(8-13Hz)绝对功率指标: {Pf[(ff>=8)&(ff<13)].sum():.3g}")
