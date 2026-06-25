#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
α 反应性诊断工具 —— 排查"闭眼 α 占比不回升"的根因
=====================================================

把问题切成两层:**采集层(硬件/电极/参考/工频)** 与 **算法层(滤波/频段/归一化)**,
用同一段"睁眼→闭眼"数据做对照,定位 α 不反应到底卡在哪一层。

用法
----
1) 直接从设备采集(需 pyserial: ``pip install pyserial``)并立即分析:
     python eeg_alpha_diagnose.py both --port COM4 --gain 12 --out cap.npz

   采集时按屏幕提示:先**睁眼放松** 30s,再**闭眼放松** 30s,脚本自动打标签。

2) 只采集:
     python eeg_alpha_diagnose.py capture --port COM4 --out cap.npz

3) 只分析(不需要设备):
     python eeg_alpha_diagnose.py analyze --in cap.npz
     # 或直接分析上位机录制的会话目录(含 raw_signal.bin):
     python eeg_alpha_diagnose.py analyze --session "D:\\...\\张三_..._20260624_101010" \
            --open 5:35 --closed 45:75      # 睁眼/闭眼时间段(秒),可选

协议/格式与上位机一致:
  - 串口帧:见 SerialPortService.cs(响应头 0xAA / 尾 0x55,数据命令 0x06,8 路交织 float32,CH0=电极)。
  - 录制:见 RecordingService.cs(raw_signal.bin: [int64 ticks][byte tag]; tag1 EEG=[float32 µV])。
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt, detrend

# Windows 控制台默认 GBK,无法输出 µ / ✅ 等字符 → 强制 UTF-8(失败则降级替换)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 频段定义 —— 与 eeg_preprocessor.py 的 BANDS 完全一致(γ 上限取 45 避肌电/工频)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# 与 SerialPortService.cs 的电极阈值一致
FLAT_UV = 0.5        # std 低于此 ⇒ 无信号(短路/断开)
LEAD_OFF_UV = 1500.0  # std 高于此 ⇒ 电极悬空/导联脱落

MAINS_HZ = 50.0      # 国内市电

# ── 标记 ──────────────────────────────────────────────────────────────────────
OK, WARN, BAD = "✅", "⚠️", "❌"


# ============================================================================
#  采集层:ADS1299 串口直采(复刻 SerialPortService.cs)
# ============================================================================
REQ_HEADER, RSP_HEADER, REQ_TAIL, RSP_TAIL = 0xA5, 0xAA, 0x5A, 0x55
ADDR = 0x00
CMD_CONN, CMD_STARTSTOP, CMD_PARAMS, CMD_DATA = 0x03, 0x04, 0x05, 0x06
CONN_KEEPALIVE, CONN_DISCONNECT = 0x02, 0x00
STREAM_CH = 8        # 板子恒定输出 8 路交织,CH0 为电极信号(单位已是 µV)


def _build_frame(cmd: int, data: bytes) -> bytes:
    """写命令: bit7 清零(与固件实测一致,和文档相反)。"""
    n = len(data)
    total = 8 + n
    f = bytearray(total)
    f[0] = REQ_HEADER
    f[1] = total & 0xFF
    f[2] = (total >> 8) & 0xFF
    f[3] = ADDR
    f[4] = cmd & 0x7F                       # write
    f[5] = f[1] ^ f[2] ^ f[3] ^ f[4]
    dc = 0
    for i, b in enumerate(data):
        f[6 + i] = b
        dc ^= b
    f[6 + n] = dc
    f[7 + n] = REQ_TAIL
    return bytes(f)


def _parse_into(buf: bytearray, out, label: int):
    """从滚动缓冲提取 0xAA..0x55 帧;对数据帧(0x06)去交织取 CH0,追加 (label, µV)。"""
    while True:
        i = buf.find(bytes([RSP_HEADER]))
        if i < 0:
            buf.clear()
            return
        if i > 0:
            del buf[:i]
        if len(buf) < 3:
            return
        total = buf[1] | (buf[2] << 8)
        if total < 8 or total > 4096:
            del buf[0]
            continue
        if len(buf) < total:
            return
        if buf[total - 1] != RSP_TAIL:
            del buf[0]
            continue
        cmd = buf[4] & 0x7F
        n = total - 8
        data = bytes(buf[6:6 + n])
        del buf[:total]
        if cmd == CMD_DATA:
            nf = n // 4
            if nf >= STREAM_CH:
                vals = struct.unpack("<%df" % nf, data[:nf * 4])
                for s in range(nf // STREAM_CH):
                    uv = vals[s * STREAM_CH]      # CH0
                    if uv != uv or abs(uv) == float("inf"):  # NaN/Inf
                        uv = 0.0
                    out.append((label, float(uv)))


def capture(port, baud, gain, rate, differential, open_s, closed_s, ready_s, out_path,
            phases=None, input_switch=0x00):
    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("需要 pyserial:  pip install pyserial")

    ser = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=0.05)
    # 厂商板上电 DTR/RTS 拉高会复位 MCU → 必须保持 false
    try:
        ser.dtr = False
        ser.rts = False
    except Exception:
        pass

    # 启动序列:STOP → PARAMS →(等待)→ START(与上位机一致)
    ser.write(_build_frame(CMD_STARTSTOP, bytes([0x00])))
    time.sleep(0.12)
    # PARAMS: [rate:u16][gain][chMask][inputSwitch][diff]; inputSwitch 0=电极 1=内部短路 2=测试信号
    params = bytes([rate & 0xFF, (rate >> 8) & 0xFF, gain & 0xFF,
                    0x01, input_switch & 0xFF, 0x01 if differential else 0x00])
    ser.write(_build_frame(CMD_PARAMS, params))
    time.sleep(0.25)
    ser.write(_build_frame(CMD_STARTSTOP, bytes([0x01])))

    # 1Hz 保活线程
    import threading
    stop_flag = threading.Event()

    def _keepalive():
        while not stop_flag.is_set():
            try:
                ser.write(_build_frame(CMD_CONN, bytes([CONN_KEEPALIVE])))
            except Exception:
                pass
            stop_flag.wait(1.0)

    ka = threading.Thread(target=_keepalive, daemon=True)
    ka.start()

    samples = []   # (label, µV);  label: -1=准备 0=睁眼 1=闭眼
    buf = bytearray()
    if phases is None:
        phases = [
            (-1, ready_s,  "准备:请坐稳放松,稍后【睁眼】注视前方"),
            (0,  open_s,   "采集中【睁眼】放松,注视前方不动"),
            (-1, ready_s,  "准备:请慢慢【闭眼】放松,不要用力闭"),
            (1,  closed_s, "采集中【闭眼】放松,保持安静"),
        ]
    fs_meas = {}
    in_name = {0: "电极", 1: "内部短路", 2: "测试信号"}.get(input_switch, str(input_switch))
    print(f"\n开始采集 @ {port} {baud}  rate={rate}Hz gain={gain} "
          f"{'差分' if differential else '单端'}  输入={in_name}\n")
    try:
        for label, dur, name in phases:
            t0 = time.time()
            c0 = len(samples)
            while True:
                now = time.time()
                left = dur - (now - t0)
                if left <= 0:
                    break
                d = ser.read(4096)
                if d:
                    buf.extend(d)
                    _parse_into(buf, samples, label)
                print(f"\r{name}  剩余 {left:4.1f}s  已采 {len(samples):7d} 点", end="")
            if label in (0, 1):
                got = len(samples) - c0
                fs_meas[label] = got / dur if dur > 0 else 0.0
                print(f"\r{name}  完成,{got} 点,实测 fs≈{fs_meas[label]:.1f}Hz" + " " * 12)
            else:
                print()
    finally:
        stop_flag.set()
        try:
            ser.write(_build_frame(CMD_STARTSTOP, bytes([0x00])))
            ser.write(_build_frame(CMD_CONN, bytes([CONN_DISCONNECT])))
        except Exception:
            pass
        ser.close()

    labels = np.array([s[0] for s in samples], dtype=np.int8)
    uv = np.array([s[1] for s in samples], dtype=np.float64)
    fs_measured = float(np.mean([v for v in fs_meas.values() if v > 0]) or rate)
    np.savez(out_path, uv=uv, labels=labels, fs_nominal=float(rate),
             fs_measured=fs_measured, gain=int(gain))
    print(f"\n已保存 → {out_path}  (共 {len(uv)} 点, 实测 fs≈{fs_measured:.1f}Hz)")
    return out_path


# ============================================================================
#  读取上位机录制的 raw_signal.bin
# ============================================================================
def load_session(session_dir):
    meta_path = os.path.join(session_dir, "raw_signal.meta.json")
    fs = 250.0
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        fs = float(meta.get("eeg_sample_rate_hz", 250))
    raw = open(os.path.join(session_dir, "raw_signal.bin"), "rb").read()
    uv = []
    i, L = 0, len(raw)
    while i + 9 <= L:
        tag = raw[i + 8]
        i += 9
        if tag == 1:                       # EEG: float32 µV
            if i + 4 > L:
                break
            uv.append(struct.unpack_from("<f", raw, i)[0])
            i += 4
        elif tag == 2:                     # VITAL: 3×float32
            i += 12
        elif tag == 3:                     # PPG: 2×int32
            i += 8
        else:
            break
    return np.array(uv, dtype=np.float64), fs


# ============================================================================
#  分析层
# ============================================================================
def psd(x, fs, res_hz=0.5):
    x = detrend(np.asarray(x, float))
    nper = int(round(fs / res_hz))
    nper = min(nper, len(x))
    if nper < 32:
        return None, None
    f, p = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2)
    return f, p


def clean(x, fs):
    """复刻服务端:0.5–47Hz 带通(零相位)+ 50Hz 陷波。"""
    nyq = fs / 2.0
    sos = butter(4, [0.5 / nyq, min(47.0, nyq * 0.99) / nyq], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    if MAINS_HZ < nyq:
        b, a = iirnotch(MAINS_HZ / nyq, 30.0)
        y = filtfilt(b, a, y)
    return y


def band_abs(f, p):
    out = {}
    for b, (lo, hi) in BANDS.items():
        idx = (f >= lo) & (f < hi)
        out[b] = float(np.trapz(p[idx], f[idx])) if idx.any() else 0.0
    return out


def band_rel(bp):
    tot = sum(bp.values()) or 1.0
    return {k: v / tot for k, v in bp.items()}


def power_in(f, p, lo, hi):
    idx = (f >= lo) & (f <= hi)
    return float(np.trapz(p[idx], f[idx])) if idx.any() else 0.0


def sef95(f, p, fmax=45.0, pct=0.95):
    idx = (f >= 0.5) & (f <= fmax)
    ff, pp = f[idx], p[idx]
    c = np.cumsum(pp)
    if c.size == 0 or c[-1] <= 0:
        return float("nan")
    return float(ff[np.searchsorted(c, pct * c[-1])])


def db(v):
    return 10 * np.log10(v) if v > 0 else float("-inf")


def analyze_segment(name, x, fs):
    """返回该段的全部诊断指标(基于 raw 与 cleaned 两版)。"""
    x = np.asarray(x, float)
    r = {"name": name, "n": len(x), "dur": len(x) / fs}
    if len(x) < fs:          # < 1s 不足以分析
        r["too_short"] = True
        return r
    r["dc"] = float(np.mean(x))
    r["std"] = float(np.std(x))
    r["ptp"] = float(np.ptp(x))
    mx = float(np.max(np.abs(x)))
    r["clip_pct"] = float(np.mean(np.abs(x) >= 0.98 * mx) * 100) if mx > 0 else 0.0

    f, p = psd(x, fs)
    r["mains_raw"] = power_in(f, p, MAINS_HZ - 1, MAINS_HZ + 1)
    r["band_0_45_raw"] = power_in(f, p, 0.5, 45)
    r["dom_hz"] = float(f[np.argmax(p)]) if f is not None else float("nan")

    xc = clean(x, fs)
    fc, pc = psd(xc, fs)
    r["abs"] = band_abs(fc, pc)
    r["rel"] = band_rel(r["abs"])
    r["sef95"] = sef95(fc, pc)
    r["mains_clean"] = power_in(fc, pc, MAINS_HZ - 1, MAINS_HZ + 1)
    r["total_clean"] = power_in(fc, pc, 0.5, 45)
    r["f"], r["p"] = fc, pc
    return r


def fmt_seg(r):
    if r.get("too_short"):
        return f"  [{r['name']}] 数据不足({r['dur']:.1f}s),跳过"
    lines = [f"  [{r['name']}]  时长 {r['dur']:.1f}s  采样 {r['n']}"]
    lines.append(f"    DC偏置 {r['dc']:+.1f}µV   幅度std {r['std']:.1f}µV   "
                 f"峰峰 {r['ptp']:.0f}µV   近饱和 {r['clip_pct']:.1f}%")
    mains_frac = r["mains_raw"] / (r["band_0_45_raw"] + 1e-12)
    lines.append(f"    原始频谱: 主频 {r['dom_hz']:.1f}Hz   "
                 f"50Hz工频/带内功率 = {mains_frac*100:.0f}%   "
                 f"陷波后残留 {r['mains_clean']/(r['total_clean']+1e-12)*100:.1f}%")
    lines.append(f"    SEF95 {r['sef95']:.1f}Hz")
    lines.append("    频段(滤波后):  " + "  ".join(
        f"{b[:1]}:{r['rel'][b]*100:4.1f}% / {db(r['abs'][b]):5.1f}dB" for b in BANDS))
    return "\n".join(lines)


def verdict(seg_open, seg_closed, fs_nominal, fs_measured):
    """综合两段给出根因判定与建议。"""
    out = ["", "=" * 72, "  根因判定", "=" * 72]

    def add(mark, msg):
        out.append(f"{mark}  {msg}")

    # 0. 采样率(频段平移的隐形杀手)
    if fs_measured and fs_nominal:
        dev = abs(fs_measured - fs_nominal) / fs_nominal
        if dev > 0.03:
            add(BAD, f"采样率不符: 标称 {fs_nominal:.0f}Hz 实测 {fs_measured:.1f}Hz "
                     f"(偏差 {dev*100:.0f}%) → 所有频段整体平移,α 抓错频率。"
                     "请核对设备配置 / 丢帧情况。")
        else:
            add(OK, f"采样率一致(标称 {fs_nominal:.0f}Hz ≈ 实测 {fs_measured:.1f}Hz)。")

    segs = [s for s in (seg_open, seg_closed) if s and not s.get("too_short")]

    # 1. 电极/信号有效性
    for s in segs:
        if s["std"] < FLAT_UV:
            add(BAD, f"[{s['name']}] 幅度 {s['std']:.2f}µV 过低(<{FLAT_UV}) → "
                     "短路/断开/无信号。先解决接触再谈 α。")
        elif s["std"] > LEAD_OFF_UV:
            add(BAD, f"[{s['name']}] 幅度 {s['std']:.0f}µV 过高(>{LEAD_OFF_UV}) → "
                     "电极悬空/导联脱落,采到的是放大器噪声而非脑电。")
        elif not (5 <= s["std"] <= 150):
            add(WARN, f"[{s['name']}] 幅度 {s['std']:.1f}µV 偏离生理范围(5–150µV)。")

    # 2. 工频污染
    for s in segs:
        frac = s["mains_raw"] / (s["band_0_45_raw"] + 1e-12)
        if frac > 0.5:
            add(BAD, f"[{s['name']}] 50Hz 工频占带内功率 {frac*100:.0f}% → "
                     "接地/参考/屏蔽问题,真实脑电(含 α)被工频淹没。")
        elif frac > 0.2:
            add(WARN, f"[{s['name']}] 50Hz 工频偏高({frac*100:.0f}%),建议检查接地。")

    # 3. α 反应性(核心)
    if seg_open and seg_closed and not seg_open.get("too_short") \
            and not seg_closed.get("too_short"):
        a_open = seg_open["abs"]["alpha"]
        a_closed = seg_closed["abs"]["alpha"]
        ratio = a_closed / (a_open + 1e-12)
        rel_open = seg_open["rel"]["alpha"] * 100
        rel_closed = seg_closed["rel"]["alpha"] * 100
        out.append("")
        add("📊", f"α 绝对功率: 睁眼 {db(a_open):.1f}dB → 闭眼 {db(a_closed):.1f}dB  "
                  f"(闭/睁 = {ratio:.2f}×)")
        add("📊", f"α 占比:     睁眼 {rel_open:.1f}% → 闭眼 {rel_closed:.1f}%  "
                  f"(Δ {rel_closed-rel_open:+.1f}pp)")

        if ratio >= 1.3:
            add(OK, "α 绝对功率随闭眼明显上升 → Berger 效应正常,采集层没问题。")
            if rel_closed - rel_open >= 3:
                add(OK, "占比也同步上升 → 系统工作正常。之前界面‘不回升’"
                        "可能是:闭眼时长/时机、显示平滑、或当时电极尚未稳定。")
            else:
                add(BAD, "但‘占比’几乎不动 → 问题在**算法归一化层**: "
                         "α 真实在涨,却被同时偏大的 δ(漂移/眼动)或宽带肌电"
                         "稀释,导致相对占比看不出变化。")
                _delta_emg_hint(out, seg_closed)
        elif ratio <= 1.1:
            add(BAD, "α 绝对功率闭眼几乎不升 → 问题在**采集/生理层**,"
                     "不是软件归一化能救的。常见原因见下。")
            _no_alpha_hint(out, seg_open, seg_closed)
        else:
            add(WARN, f"α 反应微弱(仅 {ratio:.2f}×)。前额本就 α 偏弱,"
                      "但仍偏低,建议结合下面的肌电/漂移线索排查。")
            _no_alpha_hint(out, seg_open, seg_closed)
    else:
        add(WARN, "缺少睁眼/闭眼两段对照,无法判定 α 反应性。"
                  "请用 capture/both 采集,或对 session 指定 --open/--closed。")

    out.append("=" * 72)
    return "\n".join(out)


def _delta_emg_hint(out, seg):
    d = seg["rel"]["delta"] * 100
    g = seg["rel"]["gamma"] * 100
    if d > 45:
        out.append(f"      → 闭眼段 δ 占 {d:.0f}%(偏高): 前额漂移/眼动主导,"
                   "建议加强去漂移/去眨眼(后端已有 _deblink_for_power,可调强)。")
    if g > 10:
        out.append(f"      → 闭眼段 γ 占 {g:.0f}%(偏高): 额肌肌电渗入,"
                   "建议提示受试者松开额头/下颌,或收紧 γ 上限。")


def _no_alpha_hint(out, seg_open, seg_closed):
    sc = seg_closed if seg_closed and not seg_closed.get("too_short") else seg_open
    if sc is None:
        return
    beta = sc["rel"]["beta"] * 100
    gamma = sc["rel"]["gamma"] * 100
    if beta > 35:
        out.append(f"      → 闭眼段 β 仍占 {beta:.0f}%: 受试者未真正放松/紧张/"
                   "认知活跃,α 起不来。请安静闭眼 20s 再测。")
    if gamma > 10:
        out.append(f"      → γ 占 {gamma:.0f}%: 额肌肌电干扰,松开面部肌肉。")
    out.append("      → 核对: 电极是否贴在前额且接触良好、参考/地是否接好、"
               "导联是否接反;前额 α 本就弱,必要时加测枕区(Oz/O1/O2)确认受试者确有 α。")


# ============================================================================
#  入口
# ============================================================================
def _valid(uv, fs):
    if not np.isfinite(fs) or fs <= 0 or len(uv) < 32 or len(uv) < fs:
        print(f"\n{BAD} 未采到有效数据(共 {len(uv)} 点)——设备没出数据。"
              "请检查:USB/电池/采集器是否复位、上位机是否仍占用 COM4,然后重测。")
        return False
    return True


def noise_report(uv, fs):
    """Stage 0A · 内部短路噪声底(InputSwitch=1,无需电极)。验证前端+ADC本底噪声。"""
    if not _valid(uv, fs):
        return
    x = np.asarray(uv, float)
    xd = detrend(x)
    rms = float(np.std(xd))
    pp = float(np.percentile(xd, 99.7) - np.percentile(xd, 0.3))
    f, p = psd(xd, fs)
    h50 = power_in(f, p, 49, 51)
    tot = power_in(f, p, 0.5, min(45.0, fs / 2 - 1))
    print("\n" + "=" * 60)
    print(f"  Stage 0A · 内部短路噪声底   fs={fs:.1f}Hz  时长{len(uv)/fs:.1f}s")
    print("=" * 60)
    print(f"  噪声 RMS {rms:.2f}µV   峰峰(99%) {pp:.1f}µV   DC {np.mean(x):+.0f}µV")
    print(f"  短路下 50Hz/带内 = {h50/(tot+1e-12)*100:.0f}%  (短路仍有 50Hz ⇒ 内部供电/接地耦合)")
    if rms < 2:
        print(f"  {OK} 噪声底优秀(<2µV RMS),前端/ADC 干净。")
    elif rms < 5:
        print(f"  {WARN} 噪声底偏高({rms:.1f}µV RMS),可用但留意供电/接地。")
    else:
        print(f"  {BAD} 噪声底过高({rms:.1f}µV RMS)→ 前端/供电/接地噪声,后续测量都会受影响。")
    print("=" * 60)


def testsig_report(uv, fs):
    """Stage 0B · 内部方波测试信号(InputSwitch=2)。验证 MUX/PGA/ADC/传输链路 + fs。"""
    if not _valid(uv, fs):
        return
    x = detrend(np.asarray(uv, float))
    pp = float(np.percentile(x, 99.7) - np.percentile(x, 0.3))
    rms = float(np.std(x))
    f, p = psd(x, fs, res_hz=0.1)
    band = (f > 0.2) & (f < fs / 2 - 1)
    f0 = float(f[band][np.argmax(p[band])]) if band.any() else 0.0
    # 方波特征:奇次谐波能量占比高
    def bp(c, bw=0.3):
        idx = (f >= c - bw) & (f <= c + bw)
        return float(np.trapz(p[idx], f[idx])) if idx.any() else 0.0
    odd = sum(bp(k * f0) for k in (3, 5, 7) if k * f0 < fs / 2)
    fund = bp(f0)
    print("\n" + "=" * 60)
    print(f"  Stage 0B · 内部方波测试信号   fs={fs:.1f}Hz  时长{len(uv)/fs:.1f}s")
    print("=" * 60)
    print(f"  基频 {f0:.2f}Hz   幅度峰峰 {pp:.1f}µV   RMS {rms:.1f}µV   "
          f"奇次谐波/基波 {odd/(fund+1e-12):.2f}")
    if pp < 1:
        print(f"  {BAD} 几乎无信号 → 链路没通(MUX/PGA/ADC/固件未切到测试信号?)")
    else:
        print(f"  {OK} 链路连通(MUX→PGA→ADC→传输 正常)。")
        print(f"  → 核对:基频应等于板载测试信号设定值(可交叉验 fs);幅度应符合增益标定。")
    print("=" * 60)


def sinecal_report(uv, fs, expected_hz=None, expected_uv=None):
    """Stage 1 · 外部正弦标定。频率准确度 / 幅度标定 / THD;同时再次锁定 fs。"""
    if not _valid(uv, fs):
        return
    x = detrend(np.asarray(uv, float))
    f, p = psd(x, fs, res_hz=0.1)
    band = (f > 0.5) & (f < fs / 2 - 1)
    fpk = float(f[band][np.argmax(p[band])]) if band.any() else 0.0
    amp = float((np.percentile(x, 99.7) - np.percentile(x, 0.3)) / 2)  # ≈ 正弦幅值

    def bp(c, bw=0.4):
        idx = (f >= c - bw) & (f <= c + bw)
        return float(np.trapz(p[idx], f[idx])) if idx.any() else 0.0
    fund = bp(fpk)
    harm = sum(bp(k * fpk) for k in (2, 3, 4, 5) if k * fpk < fs / 2)
    thd = 100.0 * np.sqrt(harm / (fund + 1e-12))
    print("\n" + "=" * 60)
    print(f"  Stage 1 · 外部正弦标定   fs={fs:.1f}Hz  时长{len(uv)/fs:.1f}s")
    print("=" * 60)
    print(f"  测得频率 {fpk:.2f}Hz   幅值 ≈{amp:.1f}µV   THD {thd:.1f}%")
    if expected_hz:
        err = (fpk - expected_hz) / expected_hz * 100
        mark = OK if abs(err) < 2 else BAD
        print(f"  {mark} 频率误差 {err:+.1f}% (期望 {expected_hz}Hz) ← 同时验证 fs 是否正确")
    if expected_uv:
        aerr = (amp - expected_uv) / expected_uv * 100
        mark = OK if abs(aerr) < 10 else WARN
        print(f"  {mark} 幅度误差 {aerr:+.1f}% (期望 {expected_uv}µV) ← 验证 µV 标定/增益")
    print(f"  {OK if thd < 1 else WARN} THD {'<1%(优)' if thd < 1 else '偏高,检查削顶/失真'}")
    print("=" * 60)


def mains_report(uv, fs):
    """工频/接地快速体检:量化 50/100/150Hz 谐波,用于排查接地·泄漏·隔离的 A/B 对比。"""
    if not _valid(uv, fs):
        return
    x = detrend(np.asarray(uv, float))
    f, p = psd(x, fs, res_hz=0.25)
    nyq = fs / 2.0

    def line_pwr(c, bw=0.6):
        idx = (f >= c - bw) & (f <= c + bw)
        return float(np.trapz(p[idx], f[idx])) if idx.any() else 0.0

    tot = power_in(f, p, 0.5, min(45.0, nyq - 1))
    h50 = line_pwr(50.0)
    h100 = line_pwr(100.0) if nyq > 101 else float("nan")
    h150 = line_pwr(150.0) if nyq > 151 else float("nan")
    frac = h50 / (tot + 1e-12)

    def dbs(v):
        return f"{db(v):.1f}dB" if (v == v and v > 0) else "N/A"

    print("\n" + "=" * 60)
    print(f"  工频/接地体检   fs={fs:.1f}Hz   时长 {len(uv)/fs:.1f}s")
    print("=" * 60)
    print(f"  幅度std {np.std(x):.1f}µV   DC {np.mean(uv):+.0f}µV   原始主频 {f[np.argmax(p)]:.1f}Hz")
    print(f"  50Hz {dbs(h50)}   100Hz {dbs(h100)}   150Hz {dbs(h150)}")
    print(f"  50Hz / 带内(0.5–45Hz) = {frac*100:.0f}%   ← A/B 对比看这个数")
    if frac > 0.15:
        print(f"  {BAD} 工频严重({frac*100:.0f}%):接地/泄漏路径问题。首选 USB 隔离器,并确认 DRL/参考已驱动。")
    elif frac > 0.05:
        print(f"  {WARN} 工频偏高({frac*100:.0f}%):继续优化接地/远离电源线。")
    else:
        print(f"  {OK} 工频低({frac*100:.0f}%):接地良好。")
    print("=" * 60)


def do_analyze(uv, fs, labels=None, open_range=None, closed_range=None,
               fs_nominal=None, fs_measured=None):
    print("\n" + "=" * 72)
    print(f"  α 反应性分析   fs={fs:.1f}Hz   总点数={len(uv)}")
    print("=" * 72)

    seg_open = seg_closed = None
    if labels is not None:
        xo, xc = uv[labels == 0], uv[labels == 1]
        if len(xo):
            seg_open = analyze_segment("睁眼", xo, fs)
        if len(xc):
            seg_closed = analyze_segment("闭眼", xc, fs)
    else:
        def slice_range(rng):
            a, b = rng
            return uv[int(a * fs):int(b * fs)]
        if open_range:
            seg_open = analyze_segment("睁眼", slice_range(open_range), fs)
        if closed_range:
            seg_closed = analyze_segment("闭眼", slice_range(closed_range), fs)
        if not open_range and not closed_range:
            whole = analyze_segment("整段", uv, fs)
            print(fmt_seg(whole))
            print("\n(未指定 --open/--closed,只做整段体检;"
                  "α 反应性需要两段对照。)")
            return

    for s in (seg_open, seg_closed):
        if s:
            print(fmt_seg(s))
    print(verdict(seg_open, seg_closed,
                  fs_nominal or fs, fs_measured or fs))


def main():
    ap = argparse.ArgumentParser(description="α 反应性诊断")
    sub = ap.add_subparsers(dest="mode", required=True)

    def add_cap(p):
        p.add_argument("--port", required=True)
        p.add_argument("--baud", type=int, default=230400)
        p.add_argument("--gain", type=int, default=12)
        p.add_argument("--rate", type=int, default=250)
        p.add_argument("--single-ended", action="store_true", help="单端模式(默认差分)")
        p.add_argument("--open", dest="open_s", type=float, default=30.0, help="睁眼秒数")
        p.add_argument("--closed", dest="closed_s", type=float, default=30.0, help="闭眼秒数")
        p.add_argument("--ready", type=float, default=5.0, help="每段前准备秒数")
        p.add_argument("--out", default="cap.npz")

    pc = sub.add_parser("capture", help="只采集")
    add_cap(pc)
    pb = sub.add_parser("both", help="采集 + 分析")
    add_cap(pb)

    pa = sub.add_parser("analyze", help="只分析")
    pa.add_argument("--in", dest="inp", help="capture 生成的 .npz")
    pa.add_argument("--session", help="上位机录制目录(含 raw_signal.bin)")
    pa.add_argument("--open", dest="open_r", help="睁眼时间段秒, 形如 5:35")
    pa.add_argument("--closed", dest="closed_r", help="闭眼时间段秒, 形如 45:75")

    def add_static(name, helptxt, default_out, default_rate=125):
        p = sub.add_parser(name, help=helptxt)
        p.add_argument("--port", required=True)
        p.add_argument("--baud", type=int, default=230400)
        p.add_argument("--gain", type=int, default=12)
        p.add_argument("--rate", type=int, default=default_rate)
        p.add_argument("--single-ended", action="store_true")
        p.add_argument("--secs", type=float, default=20.0, help="采集秒数")
        p.add_argument("--ready", type=float, default=8.0, help="准备秒数")
        p.add_argument("--out", default=default_out)
        return p

    add_static("mains", "工频/接地快速体检(静坐~20s,做 A/B 对比)", "cap_mains.npz")
    add_static("noise", "Stage 0A 内部短路噪声底(InputSwitch=1,无需电极)", "cap_noise.npz")
    add_static("testsig", "Stage 0B 内部方波测试信号(InputSwitch=2,无需电极)", "cap_testsig.npz")
    ps = add_static("sinecal", "Stage 1 外部正弦标定(频率/幅度/THD)", "cap_sine.npz")
    ps.add_argument("--hz", type=float, default=None, help="注入正弦的期望频率 Hz")
    ps.add_argument("--amp-uv", dest="amp_uv", type=float, default=None, help="注入正弦的期望幅值 µV")

    a = ap.parse_args()

    # Stage 0/工频:单段静态采集 + 对应报告
    STATIC = {
        "mains":   (0x00, "准备:静坐放松、保持不动(即将测工频)", "采集中:静坐不动、放松面部"),
        "noise":   (0x01, "准备:设备静置(内部短路噪声底,无需电极)", "采集中:内部短路噪声底"),
        "testsig": (0x02, "准备:设备静置(内部方波测试信号,无需电极)", "采集中:内部方波测试信号"),
        "sinecal": (0x00, "准备:接好外部正弦源(经分压到 µV 级)", "采集中:外部正弦标定"),
    }
    if a.mode in STATIC:
        insw, ready_prompt, run_prompt = STATIC[a.mode]
        phases = [(-1, a.ready, ready_prompt), (0, a.secs, run_prompt)]
        out = capture(a.port, a.baud, a.gain, a.rate, not a.single_ended,
                      a.secs, 0, a.ready, a.out, phases=phases, input_switch=insw)
        d = np.load(out)
        uv = d["uv"][d["labels"] == 0]
        fs = float(d["fs_measured"])
        if a.mode == "mains":
            mains_report(uv, fs)
        elif a.mode == "noise":
            noise_report(uv, fs)
        elif a.mode == "testsig":
            testsig_report(uv, fs)
        elif a.mode == "sinecal":
            sinecal_report(uv, fs, a.hz, a.amp_uv)
        return

    if a.mode in ("capture", "both"):
        out = capture(a.port, a.baud, a.gain, a.rate, not a.single_ended,
                      a.open_s, a.closed_s, a.ready, a.out)
        if a.mode == "capture":
            return
        a = argparse.Namespace(mode="analyze", inp=out, session=None,
                               open_r=None, closed_r=None)

    # analyze
    def parse_range(s):
        if not s:
            return None
        lo, hi = s.split(":")
        return (float(lo), float(hi))

    if a.inp:
        d = np.load(a.inp)
        do_analyze(d["uv"], float(d["fs_measured"]), labels=d["labels"],
                   fs_nominal=float(d["fs_nominal"]),
                   fs_measured=float(d["fs_measured"]))
    elif a.session:
        uv, fs = load_session(a.session)
        do_analyze(uv, fs, labels=None,
                   open_range=parse_range(a.open_r),
                   closed_range=parse_range(a.closed_r),
                   fs_nominal=fs, fs_measured=fs)
    else:
        sys.exit("analyze 需要 --in 或 --session")


if __name__ == "__main__":
    main()
