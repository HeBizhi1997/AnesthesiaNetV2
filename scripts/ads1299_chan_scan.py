"""ads1299_chan_scan.py — capture from COM6 and report per-channel stats for ALL 8
de-interleaved channels, plus the true sample rate. Finds which interleave index holds
the clean connected electrode (the vendor app shows CH1 clean @ 500sps).

Requires COM6 free (close the LK-M1299 vendor app first).
Usage: python scripts/ads1299_chan_scan.py [seconds] [rate]
"""
import sys, time, struct
import numpy as np
from scipy.signal import welch
import serial

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
RATE = int(sys.argv[2]) if len(sys.argv) > 2 else 500   # vendor uses 500sps

def fr(c, w, d=b""):
    t = 8 + len(d); ln = struct.pack("<H", t); cb = c | 0x80 if w else c
    hc = ln[0] ^ ln[1] ^ 0 ^ cb; dc = 0
    for b in d: dc ^= b
    return bytes([0xA5, ln[0], ln[1], 0, cb, hc]) + d + bytes([dc, 0x5A])

s = serial.Serial("COM6", 230400, timeout=0.1); s.dtr = True; s.rts = True
s.write(fr(3,1,bytes([1]))); time.sleep(0.15)
# rate(LE), gain=0x0c, chMask=0xff(all8), inputSwitch=0, mode=1(diff)
s.write(fr(5,1,bytes([RATE & 0xFF, (RATE >> 8) & 0xFF, 0x0c, 0xff, 0, 1]))); time.sleep(0.15)
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
elapsed = time.time() - t0
s.write(fr(4,1,bytes([0]))); s.write(fr(3,1,bytes([0]))); s.close()

a = np.array(fl); ntot = len(a)
n = ntot // 8
print(f"==== chan scan  secs={elapsed:.1f}  floats={ntot}  throughput={ntot/elapsed:.0f} f/s ====")
print(f"若 8 通道: 每通道 {n/elapsed:.0f} sps   若 1 通道: {ntot/elapsed:.0f} sps")
print()

def stats(x, fs):
    x = x - x.mean()
    f, P = welch(x, fs=fs, nperseg=min(len(x), int(fs*2)))
    band = (f >= 1) & (f <= 60)
    totp = P[band].sum() + 1e-12
    p50 = P[(f >= 48) & (f <= 52)].sum() / totp * 100
    dom = f[band][np.argmax(P[band])]
    return x.std(), p50, dom

# Interpret as 8-ch interleaved
m = a[:n*8].reshape(n, 8)
fs8 = n / elapsed
print(f"--- 按 8 通道交织解读 (每路 fs≈{fs8:.0f}Hz) ---")
print(f"{'idx':>3} {'std':>10} {'50Hz%':>7} {'主频':>7}")
for k in range(8):
    sd, p50, dom = stats(m[:, k], fs8)
    flag = "  <== 干净?" if (p50 < 40 and sd < 5000) else ""
    print(f"{k:>3} {sd:>10.1f} {p50:>6.0f}% {dom:>6.1f}Hz{flag}")

# Also interpret as single channel (vendor selects only CH1)
print(f"\n--- 按 1 通道解读 (fs≈{ntot/elapsed:.0f}Hz) ---")
sd, p50, dom = stats(a, ntot/elapsed)
print(f"    std={sd:.1f}  50Hz%={p50:.0f}%  主频={dom:.1f}Hz")
np.save("/tmp/chan_scan_all.npy", a)
print("\n原始 float 流已存 /tmp/chan_scan_all.npy")
