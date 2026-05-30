"""
ads1299_live_infer.py — 从 COM6 实采真实脑电，喂给 Python 推理(BISPredictor + EEGPreprocessor)。

流程:
  1. 连接 COM6，电极模式采集 N 秒，de-interleave 取 CH0(电极通道)。
  2. 按 1 秒 chunk 顺序喂给 BISPredictor(维护滚动缓冲+GRU状态) 与 EEGPreprocessor(频段/SQI)。
  3. 打印 BIS / SQI / 频段 / 幅值时间线。

用法:
  python scripts/ads1299_live_infer.py --seconds 90 --port COM6
  python scripts/ads1299_live_infer.py --ckpt outputs/checkpoints/v14/best_model_v3.pt
"""
from __future__ import annotations
import argparse, struct, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EEGMonitor" / "EEGProcessingService"))

try:
    import serial
except ImportError:
    print("需要 pyserial"); sys.exit(1)

NCH = 8

def _frame(cmd, write, data=b""):
    total = 8 + len(data); ln = struct.pack("<H", total)
    cmdb = cmd | 0x80 if write else cmd
    hc = ln[0] ^ ln[1] ^ 0 ^ cmdb; dc = 0
    for b in data: dc ^= b
    return bytes([0xA5, ln[0], ln[1], 0x00, cmdb, hc]) + data + bytes([dc, 0x5A])

def _parse(buf):
    out = []
    while True:
        i = buf.find(0xAA)
        if i < 0:
            if len(buf) > 65536: del buf[:]
            return out
        if i > 0: del buf[:i]
        if len(buf) < 3: return out
        total = buf[1] | (buf[2] << 8)
        if total < 8 or total > 65536: del buf[0]; continue
        if len(buf) < total: return out
        if buf[total - 1] != 0x55: del buf[0]; continue
        cmd = buf[4] & 0x7F; n = total - 8
        data = bytes(buf[6:6 + n]); del buf[:total]
        out.append((cmd, data))

def capture(port, secs):
    ser = serial.Serial(port, 230400, timeout=0.1); ser.dtr = True; ser.rts = True
    ser.write(_frame(0x03, True, bytes([0x01]))); time.sleep(0.15)
    ser.write(_frame(0x05, True, bytes([0xF4, 0x01, 0x0C, 0xFF, 0x00, 0x00]))); time.sleep(0.15)
    ser.write(_frame(0x04, True, bytes([0x01])))
    ser.reset_input_buffer()
    raw = bytearray(); t0 = time.time(); last = 0
    while time.time() - t0 < secs:
        c = ser.read(8192)
        if c: raw.extend(c)
        el = int(time.time() - t0)
        if el != last:
            last = el; print(f"\r采集中 {el}/{int(secs)}s ...", end="", flush=True)
    ser.write(_frame(0x04, True, bytes([0x00]))); ser.write(_frame(0x03, True, bytes([0x00]))); ser.close()
    print()
    floats = []
    for cmd, data in _parse(raw):
        if cmd == 0x06:
            nf = len(data) // 4
            floats.extend(struct.unpack(f"<{nf}f", data[:nf * 4]))
    a = np.array(floats, dtype=np.float64)
    nper = len(a) // NCH
    a = a[:nper * NCH].reshape(nper, NCH)
    fs = nper / secs
    return a[:, 0], fs   # CH0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--ckpt", default="outputs/checkpoints/v14/best_model_v3.pt")
    args = ap.parse_args()

    print(f"== 实采 {args.seconds:.0f}s @ {args.port}（电极模式，取 CH0）==")
    ch0, fs = capture(args.port, args.seconds)
    fs_i = int(round(fs))
    print(f"采到 {len(ch0)} 点  实测 fs≈{fs:.0f} Hz  原始幅值 std={ch0.std():.0f}\n")

    from models.bis_predictor import BISPredictor
    from preprocessing.eeg_preprocessor import EEGPreprocessor
    print(f"加载模型: {args.ckpt}")
    pred = BISPredictor(model_path=args.ckpt, sample_rate=fs_i)
    pred.reset_state()
    pre = EEGPreprocessor(sample_rate=fs_i); pre.reset()

    n = fs_i
    nchunks = len(ch0) // n
    print(f"按 1s chunk({n}样本) 顺序推理，共 {nchunks} 个\n")
    print(f"{'t(s)':>5} {'BIS':>6} {'SQI':>5} {'amp':>6} {'δ':>5} {'θ':>5} {'α':>5} {'β':>5} {'γ':>5}")
    bis_hist = []
    for k in range(nchunks):
        chunk = ch0[k * n:(k + 1) * n].reshape(-1, 1)
        r = pre.preprocess(chunk)
        bands = {b: r[f"{b}_power"] for b in ["delta", "theta", "alpha", "beta", "gamma"]}
        bis = pred.predict(chunk, bands)
        bis_hist.append(bis)
        if (k + 1) % 5 == 0 or k == nchunks - 1:
            print(f"{k+1:>5} {bis:>6.1f} {r['sqi']:>5.0f} {r['eeg_amplitude_uv']:>6.1f} "
                  f"{bands['delta']:>5.2f} {bands['theta']:>5.2f} {bands['alpha']:>5.2f} "
                  f"{bands['beta']:>5.2f} {bands['gamma']:>5.2f}")

    bh = np.array(bis_hist)
    warm = bh[fs_i and len(bh) > 5 and 5:]  # skip warmup
    tail = bh[-min(20, len(bh)):]
    print(f"\nBIS 统计: 全程 mean={bh.mean():.1f} std={bh.std():.1f}  "
          f"后20s mean={tail.mean():.1f} 范围[{tail.min():.0f},{tail.max():.0f}]")
    print("判读: 清醒静息 BIS 应在 ~85-98；若落在合理区间且随状态变化 → 端到端通。")

if __name__ == "__main__":
    main()
