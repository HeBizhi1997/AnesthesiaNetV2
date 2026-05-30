"""ads1299_chmask_test.py — does the board honor chMask, or always stream 8 floats?
Replicates the WPF request (chMask=0x01, rate=250, diff) vs the proven scan (0xff, 500).

Requires COM6 free (disconnect WPF first).
"""
import time, struct
import numpy as np
from scipy.signal import welch
import serial

def fr(c, w, d=b""):
    t = 8 + len(d); ln = struct.pack("<H", t); cb = c | 0x80 if w else c
    hc = ln[0] ^ ln[1] ^ 0 ^ cb; dc = 0
    for b in d: dc ^= b
    return bytes([0xA5, ln[0], ln[1], 0, cb, hc]) + d + bytes([dc, 0x5A])

def capture(rate, chmask, mode, secs=6.0):
    s = serial.Serial("COM6", 230400, timeout=0.1); s.dtr = True; s.rts = True
    s.write(fr(3,1,bytes([1]))); time.sleep(0.15)
    s.write(fr(5,1,bytes([rate & 0xFF, (rate>>8)&0xFF, 0x0c, chmask, 0, mode]))); time.sleep(0.2)
    s.reset_input_buffer()
    s.write(fr(4,1,bytes([1])))
    buf=bytearray(); fl=[]; t0=time.time()
    while time.time()-t0 < secs:
        c=s.read(8192)
        if c: buf.extend(c)
        while True:
            i=buf.find(0xAA)
            if i<0:
                if len(buf)>16384: del buf[:]
                break
            if i>0: del buf[:i]
            if len(buf)<3: break
            tot=buf[1]|(buf[2]<<8)
            if tot<8 or tot>16384: del buf[0]; continue
            if len(buf)<tot: break
            if buf[tot-1]!=0x55: del buf[0]; continue
            if (buf[4]&0x7f)==6:
                d=bytes(buf[6:tot-2]); fl.extend(struct.unpack(f"<{len(d)//4}f", d))
            del buf[:tot]
    el=time.time()-t0
    s.write(fr(4,1,bytes([0]))); s.write(fr(3,1,bytes([0]))); s.close()
    return np.array(fl), el

def dom(x, fs):
    x = x - x.mean()
    f,P = welch(x, fs=fs, nperseg=min(len(x), int(fs*2)))
    b=(f>=1)&(f<=120); return f[b][np.argmax(P[b])]

for label, rate, chmask, mode in [
    ("WPF请求 (chMask=0x01 rate=250 diff)", 250, 0x01, 1),
    ("扫描请求 (chMask=0xFF rate=500 diff)", 500, 0xFF, 1),
]:
    a, el = capture(rate, chmask, mode)
    n = len(a)
    print(f"\n==== {label} ====")
    print(f"  floats={n}  throughput={n/el:.0f} f/s  ({el:.1f}s)")
    if n < 16:
        print("  几乎无数据"); continue
    # how many of every 8 are nonzero? -> tells frame layout
    m8 = a[:(n//8)*8].reshape(-1,8)
    nz = [int(np.count_nonzero(np.abs(m8[:,k])>1e-9)) for k in range(8)]
    print(f"  按8解读: 每列非零计数 {nz}  (列0有值/其余为0 => 板子固定发8路)")
    # dominant if treated as 8ch keep idx0 (fs=throughput/8)
    fs8 = n/8/el
    print(f"  [8路取idx0] fs≈{fs8:.0f}Hz  主频={dom(m8[:,0], fs8):.1f}Hz")
    # dominant if treated as single channel = ALL floats (what WPF does when _streamChannels=1)
    fs1 = n/el
    print(f"  [全取单路] fs≈{fs1:.0f}Hz(标称250)  主频(按250解读)={dom(a, 250):.1f}Hz <== WPF当前路径")
