"""
nsm_verify.py — 校验 WPF 对 NSM 厂家设备(NSA-2000 / NSM_PCDisplay)的协议解析。

从 COM 口实采若干秒 NSM 报文,按**权威协议** doc/NSM_UartProtocol.txt 的偏移解析全部字段,
把频段 dB → 相对%(与厂家界面、与 EEGPreprocessor 交叉校验同一口径),并汇总打印一张对账表,
供你与厂家软件 NSM_PCDisplay 当前读数逐项核对(CSI/BS/SQI/EMG/NOX/SEF/EOG/频段%/采样率)。

用法(需先关闭厂家软件,串口独占):
    python scripts/nsm_verify.py COM8 115200 --seconds 20

权威偏移(doc/NSM_UartProtocol.txt,0 基):
    CSI=13 BS=14 SQI=15 BlackImp=16 WhiteImp=17 EMG=18 AlarmHigh=20 AlarmLow=21
    EEG=22..121(100×signed µV) NOX=125 δ=126 θ=127 α=128 β=129 γ=130 EOG=131 SEF=176
设备报文比文档短 2 字节(351 payload + 2 CRC = 353),差异在 D10 尾块(偏移>177),不影响以上字段。
"""
import sys
import time
import argparse
from collections import defaultdict

import numpy as np

try:
    import serial
except ImportError:
    print("需要 pyserial:  pip install pyserial"); sys.exit(1)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FRAME_HEADER = 0x80
PAYLOAD_LENGTH = 351
PACKET_SIZE = PAYLOAD_LENGTH + 2   # 353(payload + CRC16)
EEG_OFFSET = 22
EEG_N = 100

# 权威字段偏移(doc/NSM_UartProtocol.txt)
OFF = dict(csi=13, bs=14, sqi=15, black_imp=16, white_imp=17, emg=18,
           alarm_high=20, alarm_low=21, nox=125,
           delta=126, theta=127, alpha=128, beta=129, gamma=130, eog=131, sef=176)


def _sb(b):                       # unsigned byte → signed (C# sbyte 语义)
    return b - 256 if b >= 128 else b


def parse_packet(buf):
    if len(buf) < PACKET_SIZE or buf[0] != FRAME_HEADER:
        return None
    if (buf[1] | (buf[2] << 8)) != PAYLOAD_LENGTH:
        return None
    blk = buf[10]
    eeg = np.array([_sb(buf[EEG_OFFSET + i]) for i in range(EEG_N)], dtype=np.float64)
    return dict(
        device_time=buf[6] | (buf[7] << 8) | (buf[8] << 16) | (buf[9] << 24),
        electrode_alarm=bool(blk & 0x02), impedance_high=bool(blk & 0x08),
        electrode_invalid=bool(blk & 0x80),
        event_number=buf[11], event_type=buf[12],
        csi=buf[OFF["csi"]], bs=buf[OFF["bs"]], sqi=buf[OFF["sqi"]],
        emg=buf[OFF["emg"]], nox=buf[OFF["nox"]], sef=buf[OFF["sef"]], eog=buf[OFF["eog"]],
        black_imp=buf[OFF["black_imp"]], white_imp=buf[OFF["white_imp"]],
        alarm_high=buf[OFF["alarm_high"]], alarm_low=buf[OFF["alarm_low"]],
        delta_db=_sb(buf[OFF["delta"]]), theta_db=_sb(buf[OFF["theta"]]),
        alpha_db=_sb(buf[OFF["alpha"]]), beta_db=_sb(buf[OFF["beta"]]),
        gamma_db=_sb(buf[OFF["gamma"]]),
        eeg=eeg,
    )


def db_to_pct(db5):
    """dB(signed byte)→ 相对%。与 EEGPreprocessor._validate_against_device 同一公式:
    linear = 10^(dB/10) 后在 5 频段内归一。"""
    lin = {k: 10.0 ** (v / 10.0) for k, v in db5.items()}
    tot = sum(lin.values()) or 1.0
    return {k: 100.0 * lin[k] / tot for k in db5}


def _valid(v):                    # 0xEE/0xFF 视为无效
    return None if v in (0xEE, 0xFF) else v


def main():
    ap = argparse.ArgumentParser(description="校验 NSM 协议解析(对账厂家软件)")
    ap.add_argument("port", nargs="?", default="COM8")
    ap.add_argument("baud", nargs="?", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    print(f"打开 {args.port} @ {args.baud}(请确认厂家软件已关闭,串口独占)...")
    ser = serial.Serial(args.port, args.baud, timeout=0.5)

    buf = bytearray()
    pkts, t_arr = [], []
    raw_bytes = length_fail = 0
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            raw_bytes += len(chunk); buf.extend(chunk)
            while len(buf) >= PACKET_SIZE:
                try:
                    hi = buf.index(FRAME_HEADER)
                except ValueError:
                    buf.clear(); break
                if hi > 0:
                    buf = buf[hi:]
                if len(buf) < PACKET_SIZE:
                    break
                pkt = parse_packet(bytes(buf[:PACKET_SIZE]))
                if pkt is not None:
                    pkts.append(pkt); t_arr.append(time.time())
                    buf = buf[PACKET_SIZE:]
                else:
                    if buf[0] == FRAME_HEADER and (buf[1] | (buf[2] << 8)) != PAYLOAD_LENGTH:
                        length_fail += 1
                    buf.pop(0)
            el = int(time.time() - t0)
            print(f"\r采集中 {el}/{int(args.seconds)}s  包={len(pkts)} ...", end="", flush=True)
    finally:
        ser.close()
    print()

    if not pkts:
        print("未解析到有效 NSM 报文。检查:端口/波特率、厂家软件是否占用串口、设备是否在输出。")
        return

    # 采样率:包率 × 100 samples/包
    rate_hz = None
    if len(t_arr) > 3:
        dt = (t_arr[-1] - t_arr[0]) / (len(t_arr) - 1)
        if dt > 0:
            rate_hz = 100.0 / dt
    pkt_rate = (len(pkts) - 1) / (t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else 0

    def med(key, valid=False):
        vals = [(_valid(p[key]) if valid else p[key]) for p in pkts]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    band_db = {b: med(f"{b}_db") for b in ["delta", "theta", "alpha", "beta", "gamma"]}
    band_pct = db_to_pct(band_db)
    eeg_all = np.concatenate([p["eeg"] for p in pkts])

    print("=" * 72)
    print(f"  NSM 协议解析校验  ——  {args.port}  共 {len(pkts)} 包 / {raw_bytes/1024:.1f} KB")
    print("=" * 72)
    print(f"  包率 ≈ {pkt_rate:.2f} 包/s   →   采样率 ≈ {rate_hz:.1f} Hz   (NSM 应 ≈100 Hz)")
    print(f"  长度字段不符次数: {length_fail}")
    print("-" * 72)
    print(f"  {'字段':<14}{'偏移':>6}   {'实采(中位)':>12}      厂家软件显示(请填)")
    print("-" * 72)
    rows = [
        ("CSI 麻醉深度", OFF["csi"], med("csi", True)),
        ("BS 爆发抑制", OFF["bs"], med("bs", True)),
        ("SQI 信号质量", OFF["sqi"], med("sqi", True)),
        ("EMG 肌电", OFF["emg"], med("emg", True)),
        ("NOX 伤害", OFF["nox"], med("nox", True)),
        ("EOG 眼动", OFF["eog"], med("eog")),
        ("SEF95 (Hz)", OFF["sef"], med("sef")),
        ("BlackImp 阻抗", OFF["black_imp"], med("black_imp")),
        ("WhiteImp 阻抗", OFF["white_imp"], med("white_imp")),
    ]
    for name, off, val in rows:
        vs = "INVALID/--" if val is None else f"{val:.0f}"
        print(f"  {name:<14}{off:>6}   {vs:>12}      __________")
    print("-" * 72)
    print(f"  频段(dB 原始 → 相对%,与厂家界面 δθαβγ 口径一致):")
    for b, sym in [("delta", "δ"), ("theta", "θ"), ("alpha", "α"), ("beta", "β"), ("gamma", "γ")]:
        print(f"    {sym} off={OFF[b]:>3}  dB={band_db[b]:+5.0f}  → {band_pct[b]:5.1f}%      厂家:____%")
    print(f"    (相对% 合计 = {sum(band_pct.values()):.0f}%)")
    print("-" * 72)
    print(f"  EEG 波形:范围 {eeg_all.min():.0f}..{eeg_all.max():.0f} µV  std={eeg_all.std():.1f}  "
          f"|x|=127 占比 {100*np.mean(np.abs(eeg_all) >= 127):.1f}%(±127 削顶)")
    last = pkts[-1]
    print(f"  电极:alarm={last['electrode_alarm']} hiZ={last['impedance_high']} "
          f"invalid={last['electrode_invalid']}   事件:#{last['event_number']} type={last['event_type']}")
    print("=" * 72)
    print("  对照说明:把上面「实采(中位)」与厂家软件 NSM_PCDisplay 当前数值逐行填到右列核对。")
    print("  截图示例参考:CSI99 NOX99 BS0 EMG100 EOG89 SEF44 δ18 θ13 α22 β31 γ33。")
    print("  若全部吻合 → 偏移/缩放正确;频段须用「相对%」列与厂家 δθαβγ 比较(dB 不可直接比)。")


if __name__ == "__main__":
    main()
