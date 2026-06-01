"""ads1299_recover.py — try several reset/handshake strategies to get the board streaming
again, report bytes received for each. Helps recover a hung board after many open/close cycles."""
import time, struct, sys
import serial

def fr(c, w, d=b""):
    t = 8 + len(d); ln = struct.pack("<H", t); cb = c | 0x80 if w else c
    hc = ln[0] ^ ln[1] ^ 0 ^ cb; dc = 0
    for b in d: dc ^= b
    return bytes([0xA5, ln[0], ln[1], 0, cb, hc]) + d + bytes([dc, 0x5A])

def handshake(s):
    s.write(fr(3, 1, bytes([1]))); time.sleep(0.15)          # conn connected
    s.write(fr(5, 1, bytes([0xFA, 0, 0x0c, 0xff, 0, 1]))); time.sleep(0.15)  # params rate=250 chMask=0xff diff
    s.write(fr(4, 1, bytes([1])))                            # start

def count_bytes(s, secs):
    s.reset_input_buffer(); n = 0; t0 = time.time()
    while time.time() - t0 < secs:
        c = s.read(4096)
        n += len(c)
    return n

def try_strategy(name, dtr, rts, reset_pulse=False, reboot=False):
    try:
        s = serial.Serial("COM6", 230400, timeout=0.1)
    except Exception as e:
        print(f"[{name}] open failed: {e}"); return 0
    try:
        s.dtr = dtr; s.rts = rts
        if reset_pulse:
            # pulse DTR/RTS to reset the MCU (active-low reset on many CP210x boards)
            s.dtr = False; s.rts = False; time.sleep(0.05)
            s.dtr = True;  s.rts = True;  time.sleep(0.05)
            s.dtr = dtr;   s.rts = rts;   time.sleep(0.6)
        if reboot:
            s.write(fr(2, 1, bytes([1]))); time.sleep(2.0)   # CMD_REBOOT then wait
        handshake(s)
        n = count_bytes(s, 4.0)
        # stop streaming cleanly so the next strategy starts fresh
        try: s.write(fr(4, 1, bytes([0])))
        except Exception: pass
        print(f"[{name}] dtr={dtr} rts={rts} reset={reset_pulse} reboot={reboot}  ->  {n} bytes  ({n/4.0:.0f} B/s)")
        return n
    finally:
        s.close(); time.sleep(0.4)

print("Probing COM6 recovery strategies...\n")
results = {}
results['A plain dtr/rts=True']   = try_strategy("A", True,  True)
results['B dtr/rts=False']        = try_strategy("B", False, False)
results['C DTR/RTS reset pulse']  = try_strategy("C", True,  True,  reset_pulse=True)
results['D reboot cmd']           = try_strategy("D", True,  True,  reboot=True)
results['E reset+False lines']    = try_strategy("E", False, False, reset_pulse=True)

print("\n==== summary ====")
best = max(results, key=results.get)
for k, v in results.items():
    print(f"  {k:28s}: {v} bytes" + ("   <== WORKS" if v > 0 and v == results[best] else ""))
print(f"\nbest: {best} ({results[best]} bytes)")
