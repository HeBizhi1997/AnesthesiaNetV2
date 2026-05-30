"""
ads1299_com8_test.py — 新 ADS1299 帧协议硬件连接测试（doc/EEG-ads1299-通信协议 20260529）。

与 WPF 的 SerialPortService.cs 完全相同的帧格式，用于在不启动 WPF 的情况下
快速验证串口连通性、握手、采样数据流。

帧结构（8 字节开销 + N 数据）：
  [0]Header(请求0xA5/响应0xAA) [1:2]Len(u16 LE=8+N) [3]Addr(0x00广播)
  [4]Cmd(bit7写/读, bit6:0命令码) [5]HdrChk=XOR(len0,len1,addr,cmd)
  [6:6+N]Data [6+N]DataChk=XOR(data) [7+N]Tail(请求0x5A/响应0x55)

用法:
  python scripts/ads1299_com8_test.py                  # COM8 @230400, 单通道差分, 采5秒
  python scripts/ads1299_com8_test.py --port COM8 --baud 230400 --seconds 5
  python scripts/ads1299_com8_test.py --rate 500 --gain 12 --single   # 单端模式
"""
from __future__ import annotations
import argparse
import struct
import sys
import time

try:
    import serial
except ImportError:
    print("需要 pyserial: pip install pyserial")
    sys.exit(1)

REQ_HEADER, RSP_HEADER = 0xA5, 0xAA
REQ_TAIL,   RSP_TAIL   = 0x5A, 0x55
ADDR_BROADCAST = 0x00
WRITE_BIT = 0x80
OVERHEAD = 8

CMD_FW, CMD_HW = 0x00, 0x01
CMD_CONN, CMD_START, CMD_PARAMS, CMD_DATA = 0x03, 0x04, 0x05, 0x06


def build_frame(cmd: int, write: bool, data: bytes = b"") -> bytes:
    n = len(data)
    total = OVERHEAD + n
    ln = struct.pack("<H", total)
    addr = ADDR_BROADCAST
    cmdb = (cmd | WRITE_BIT) if write else cmd
    hdr_chk = ln[0] ^ ln[1] ^ addr ^ cmdb
    data_chk = 0
    for b in data:
        data_chk ^= b
    return bytes([REQ_HEADER, ln[0], ln[1], addr, cmdb, hdr_chk]) + data + bytes([data_chk, REQ_TAIL])


def parse_frames(buf: bytearray):
    """Yield (cmd_low7, data, checksum_ok); consume parsed bytes from buf."""
    while True:
        # sync to 0xAA
        i = buf.find(RSP_HEADER)
        if i < 0:
            if len(buf) > 4096:
                del buf[:]
            return
        if i > 0:
            del buf[:i]
        if len(buf) < 3:
            return
        total = buf[1] | (buf[2] << 8)
        if total < OVERHEAD or total > 4096:
            del buf[0]
            continue
        if len(buf) < total:
            return
        if buf[total - 1] != RSP_TAIL:
            del buf[0]
            continue
        cmd = buf[4]
        hdr_chk = buf[1] ^ buf[2] ^ buf[3] ^ buf[4]
        n = total - OVERHEAD
        data = bytes(buf[6:6 + n])
        dchk = 0
        for b in data:
            dchk ^= b
        ok = (hdr_chk == buf[5]) and (dchk == buf[6 + n])
        del buf[:total]
        yield (cmd & 0x7F, data, ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--baud", type=int, default=230400)
    ap.add_argument("--rate", type=int, default=500, choices=[250, 500, 1000, 2000])
    ap.add_argument("--gain", type=int, default=12)
    ap.add_argument("--single", action="store_true", help="单端模式（默认差分）")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    differential = not args.single
    print(f"打开 {args.port} @ {args.baud} 8N1 ...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1,
                            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE, rtscts=False, dsrdtr=False)
    except Exception as e:
        print(f"[X] 打开串口失败: {e}")
        print("   可能原因：端口被占用（WPF 已连接？）/ 设备未插 / 端口号不对")
        sys.exit(2)

    try:
        ser.dtr = True   # most ADS1299 USB-UART boards require DTR/RTS asserted
        ser.rts = True
    except Exception:
        pass

    buf = bytearray()
    n_data = n_samp = n_badchk = 0
    total_raw = 0
    raw_sample = bytearray()
    versions = {}
    samp_min, samp_max, samp_sum = 1e9, -1e9, 0.0
    first_data_t = None

    def drain(deadline):
        nonlocal n_data, n_samp, n_badchk, samp_min, samp_max, samp_sum, first_data_t, total_raw
        while time.time() < deadline:
            chunk = ser.read(4096)
            if chunk:
                total_raw += len(chunk)
                if len(raw_sample) < 64:
                    raw_sample.extend(chunk[:64 - len(raw_sample)])
                buf.extend(chunk)
                for cmd, data, ok in parse_frames(buf):
                    if cmd in (CMD_FW, CMD_HW):
                        versions[cmd] = data.decode("ascii", "replace").strip("\x00 ")
                    elif cmd == CMD_DATA:
                        n_data += 1
                        if not ok:
                            n_badchk += 1
                        nf = len(data) // 4
                        if nf:
                            if first_data_t is None:
                                first_data_t = time.time()
                            vals = struct.unpack(f"<{nf}f", data[:nf * 4])
                            n_samp += nf
                            for v in vals:
                                samp_min = min(samp_min, v)
                                samp_max = max(samp_max, v)
                                samp_sum += v
            else:
                time.sleep(0.005)

    # ── 1. 连接 + 版本查询 ────────────────────────────────────────────────
    print("-> 发送 连接状态=已连接 / 请求固件&硬件版本")
    ser.write(build_frame(CMD_CONN, True, bytes([0x01])))   # connected
    ser.write(build_frame(CMD_FW, False))                    # read fw version
    ser.write(build_frame(CMD_HW, False))                    # read hw version
    drain(time.time() + 1.0)

    # ── 2. 采样参数 + 开始采集 ────────────────────────────────────────────
    params = bytes([
        args.rate & 0xFF, (args.rate >> 8) & 0xFF,   # rate u16 LE
        args.gain,                                    # gain
        0x01,                                         # channel mask: CH1 only
        0x00,                                         # input switch: electrode
        0x01 if differential else 0x00,               # single/diff
    ])
    print(f"-> 采样参数 rate={args.rate} gain={args.gain} ch=CH1 "
          f"{'差分' if differential else '单端'}；开始采集")
    ser.write(build_frame(CMD_PARAMS, True, params))
    ser.write(build_frame(CMD_START, True, bytes([0x01])))   # start

    drain(time.time() + args.seconds)

    # ── 3. 停止 + 断开 ────────────────────────────────────────────────────
    print("-> 停止采集 / 断开")
    ser.write(build_frame(CMD_START, True, bytes([0x00])))
    ser.write(build_frame(CMD_CONN, True, bytes([0x00])))
    time.sleep(0.1)
    ser.close()

    # ── 报告 ──────────────────────────────────────────────────────────────
    print("\n================ 测试结果 ================")
    if versions:
        for cmd, v in versions.items():
            print(f"  {'固件' if cmd == CMD_FW else '硬件'}版本: {v!r}")
    else:
        print("  版本响应: 无（设备未回版本帧）")

    print(f"  原始接收字节总数: {total_raw}")
    if total_raw:
        print(f"  前若干原始字节(hex): {raw_sample.hex(' ')}")

    if n_data == 0:
        print("  [X] 未收到任何采样数据帧(0x86)")
        if total_raw == 0:
            print("     -> 设备完全静默(0 字节)。检查：设备是否上电 / TX(设备)->RX(PC)接线 / 端口号是否选对")
        else:
            print("     -> 收到了字节但解析不出帧。多半是【波特率不对】或协议不符。")
            print("        建议依次试: --baud 115200 / 460800 / 921600；并核对帧头是否 0xAA")
        sys.exit(3)

    dur = (time.time() - first_data_t) if first_data_t else args.seconds
    eff_rate = n_samp / dur if dur > 0 else 0
    print(f"  [OK] 数据帧: {n_data}  采样点: {n_samp}  坏校验帧: {n_badchk}")
    print(f"  实测采样率: ~{eff_rate:.0f} Hz (设备设定 {args.rate} Hz)")
    print(f"  幅值范围: [{samp_min:.2f}, {samp_max:.2f}] µV  均值: {samp_sum / max(n_samp,1):.2f} µV")
    if n_badchk:
        print(f"  [!] {n_badchk} 帧校验失败 —— 可能是协议校验约定与文档不一致(已容错接收)")
    if abs(eff_rate - args.rate) > args.rate * 0.2:
        print(f"  [!] 实测采样率与设定偏差>20% —— 检查波特率/丢包")
    print("==========================================")


if __name__ == "__main__":
    main()
