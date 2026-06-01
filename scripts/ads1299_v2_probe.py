"""ads1299_v2_probe.py — start the board using the VENDOR's exact protocol (decompiled from
LK-M1299 ADS1299.exe): R/W bit INVERTED vs the doc (write=bit7 clear, read=bit7 set),
DTR/RTS = false, sequence STOP -> PARAMS -> START. Confirms cold-start streaming.

Requires COM6 free (close the vendor LK-M1299 app first)."""
import time, struct, serial

def fr(cmd, write, data=b""):
    total = len(data) + 8
    ln = struct.pack("<H", total)
    cb = (cmd & 0x7F) if write else (cmd | 0x80)   # <-- INVERTED, matches firmware
    hc = ln[0] ^ ln[1] ^ 0 ^ cb
    dc = 0
    for b in data: dc ^= b
    return bytes([0xA5, ln[0], ln[1], 0, cb, hc]) + bytes(data) + bytes([dc, 0x5A])

STOP   = fr(4, True, [0])
START  = fr(4, True, [1])
# params: rate=500(0xF4,0x01), gain=12(±375mV), ch_en=0x01(CH1), inputSwitch=0, mode=1(diff)
PARAMS = fr(5, True, [0xF4, 0x01, 0x0C, 0x01, 0x00, 0x01])

try:
    s = serial.Serial("COM6", 230400, timeout=0.1)
except Exception as e:
    print("COM6 BUSY — close the vendor app first:", e); raise SystemExit
s.rts = False; s.dtr = False
time.sleep(0.1)
s.write(STOP);   time.sleep(0.12)
s.write(PARAMS); time.sleep(0.25)
s.write(START)
s.reset_input_buffer()

buf = bytearray(); data_frames = 0; other = 0; floats = []; t0 = time.time(); total = 0
while time.time() - t0 < 5:
    c = s.read(8192)
    if c: buf.extend(c); total += len(c)
    while True:
        i = buf.find(0xAA)
        if i < 0:
            if len(buf) > 16384: del buf[:]
            break
        if i > 0: del buf[:i]
        if len(buf) < 3: break
        tot = buf[1] | (buf[2] << 8)
        if tot < 8 or tot > 16384: del buf[0]; continue
        if len(buf) < tot: break
        cmd = buf[4] & 0x7F
        if cmd == 6:
            data_frames += 1
            d = bytes(buf[6:tot-2]); floats.extend(struct.unpack(f"<{len(d)//4}f", d))
        else:
            other += 1
        del buf[:tot]
s.write(STOP); s.close()

print(f"bytes={total}  data_frames(0x06)={data_frames}  other(ACK)={other}  floats={len(floats)}")
if data_frames > 10:
    import numpy as np
    a = np.array(floats); n = len(a)//8
    ch0 = a[:n*8].reshape(n,8)[:,0] if n else a
    print(f"throughput={len(floats)/5:.0f} f/s  CH0 std={ch0.std():.1f}uV")
    print("VERDICT: ✅ COLD-START STREAMING WORKS with corrected protocol")
else:
    print("VERDICT: ❌ still no stream")
