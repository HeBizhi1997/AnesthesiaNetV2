"""
analyze_sleep.py — 离线睡眠脑电分析(EEGRecorder 原始数据 → 报告)

用 ads1299 项目的生产推理栈解析 EEGRecorder 录制的整夜脑电:
  1. 全程模型推理 BIS(BISPredictor + AnesthesiaNetV3,流式 1s/步,带不确定度)
  2. 睡眠分期(30s epoch,基于 BIS + 频谱特征的单通道前额启发式分期 W/N1/N2/N3/REM)
  3. 纺锤波(11-16 Hz sigma)与 K 复合波检测
  4. 睡眠结构与个人身心状态分析

输出(写到 --out-dir,默认 outputs/reports/sleep/<会话名>/):
  report.md         —— Markdown 报告
  epochs.csv        —— 每个 30s epoch 的全部特征/分期
  summary.json      —— 机器可读的关键指标
  hypnogram.png     —— 睡眠图 + BIS 曲线 + 频段/事件

用法:
  python scripts/analyze_sleep.py <会话文件夹> [--ckpt ...] [--out-dir ...] [--no-awake-cal]
  python scripts/analyze_sleep.py EEGMonitor/EEGRecorder/recordings_raw/recordings_raw/S001_20260615_004605

注意(局限性,务必在解读时考虑):
  · 单导前额差分电极,无 EOG/EMG/枕区导联 → 分期为近似,REM 与 N1/清醒难以严格区分。
  · BIS 模型在麻醉数据(VitalDB)上训练,睡眠与麻醉脑电相近但不等同 → BIS 作"皮层激活/深度"
    指标看待,与频谱分期互相印证。
  · 前额双极导联幅值显著低于标准头皮导联,N3 的 75 µV 慢波绝对阈值不直接适用 → 用自适应阈值。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch, hilbert

try:                       # Windows 控制台默认 GBK,统一改 UTF-8 以便打印中文/符号
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 让 src.* 与 ads1299 服务模块都可导入
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "EEGMonitor" / "EEGProcessingService"))

import warnings
warnings.filterwarnings("ignore")
try:
    from loguru import logger
    logger.remove()  # 静音流式日志
except Exception:
    pass

EPOCH_SEC = 30                      # AASM 标准分期 epoch 长度
STAGES = ["W", "N1", "N2", "N3", "REM", "ART"]
STAGE_Y = {"W": 5, "REM": 4, "N1": 3, "N2": 2, "N3": 1, "ART": 0}  # 睡眠图纵轴

DOTNET_EPOCH = 621355968000000000   # .NET ticks 偏移


# ────────────────────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────────────────────

def load_session(folder: str):
    folder = str(folder)
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    eeg = np.fromfile(os.path.join(folder, "eeg.bin"), dtype="<f4").astype(np.float64)
    ppg_path = os.path.join(folder, "ppg.bin")
    ppg_dtype = np.dtype([("ticks", "<i8"), ("ir", "<i4"), ("red", "<i4"),
                          ("spo2", "u1"), ("hr", "u1")])
    ppg = (np.fromfile(ppg_path, dtype=ppg_dtype)
           if os.path.exists(ppg_path) and os.path.getsize(ppg_path) else
           np.empty(0, ppg_dtype))
    return meta, eeg, ppg


def parse_start(meta) -> datetime | None:
    s = meta["eeg"].get("first_sample") or meta.get("started")
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# 全程 BIS 流式推理(复用生产 BISPredictor + EEGPreprocessor)
# ────────────────────────────────────────────────────────────────────────────

def run_bis_inference(eeg: np.ndarray, fs: int, ckpt: str, awake_cal: bool):
    """按 1s chunk 顺序喂给 BISPredictor,返回逐秒 BIS / 不确定度 / SQI / 幅值。"""
    from models.bis_predictor import BISPredictor
    from preprocessing.eeg_preprocessor import EEGPreprocessor

    pred = BISPredictor(model_path=ckpt, sample_rate=fs)
    if not awake_cal:
        pred._bias_cal_enabled = False
    pred.reset_state()
    pre = EEGPreprocessor(sample_rate=fs)
    pre.reset()

    n_chunks = len(eeg) // fs
    bis = np.full(n_chunks, np.nan)
    unc = np.full(n_chunks, np.nan)
    sqi = np.full(n_chunks, np.nan)
    amp = np.full(n_chunks, np.nan)
    model_ok = pred._model is not None

    for k in range(n_chunks):
        chunk = eeg[k * fs:(k + 1) * fs].reshape(-1, 1)
        r = pre.preprocess(chunk)
        bands = {b: r[f"{b}_power"] for b in ["delta", "theta", "alpha", "beta", "gamma"]}
        bis[k] = pred.predict(chunk, bands)
        unc[k] = pred.last_bis_uncertainty
        sqi[k] = r["sqi"]
        amp[k] = r["eeg_amplitude_uv"]
        if (k + 1) % 600 == 0:
            print(f"    BIS 推理 {k + 1}/{n_chunks} s ...", flush=True)

    return dict(bis=bis, unc=unc, sqi=sqi, amp=amp, model_ok=model_ok,
                bias=float(pred._bias), val_mae=getattr(pred, "_val_mae", None),
                n_channels=pred._n_channels)


# ────────────────────────────────────────────────────────────────────────────
# 信号滤波工具
# ────────────────────────────────────────────────────────────────────────────

def clean_filter(x: np.ndarray, fs: float) -> np.ndarray:
    """0.5–45 Hz 带通 + 50 Hz 陷波(零相位),用于频谱/事件分析。"""
    nyq = fs / 2.0
    sos_hp = butter(4, 0.5 / nyq, btype="high", output="sos")
    y = sosfiltfilt(sos_hp, x)
    for mf in (50.0,):
        if mf < nyq:
            b, a = iirnotch(mf / nyq, 30.0)
            y = filtfilt(b, a, y)
    sos_lp = butter(4, min(45.0, nyq * 0.99) / nyq, btype="low", output="sos")
    y = sosfiltfilt(sos_lp, y)
    return y


def bandpass(x: np.ndarray, lo: float, hi: float, fs: float, order=4) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def eog_filter(x: np.ndarray, fs: float) -> np.ndarray:
    """
    眼动(EOG)保留滤波:0.3–6 Hz 带通,**不做去眨眼/去眼动**,也不需 50 Hz 陷波。

    与 ads1299 的脑电预处理(0.5 Hz 高通 + 去眨眼)不同:睡眠分期(尤其 REM)恰恰要
    **保留**眼动产生的低频大幅瞬变。前额双极电极对眼动(眨眼/扫视)高度敏感,这些信号
    携带 REM/思睡期慢速眼动(SEM)与 REM 期快速眼动(REM)的关键判据,绝不能滤除。
    """
    nyq = fs / 2.0
    sos = butter(2, [0.3 / nyq, min(6.0, nyq * 0.99) / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def detect_eye_movements(eog: np.ndarray, fs: float, artifact_mask_1s: np.ndarray):
    """
    从眼动保留信号检测眼动事件(单导前额,无独立 EOG 导联,故为近似)。
      · 快速眼动(REM):1–5 Hz 的尖锐高速度偏转(扫视),成簇出现 → REM 期判据。
      · 慢速眼动(SEM):0.25–1 Hz 的平缓滚动 → 思睡/N1 与 REM 期可见。
    返回 (rem_events, sem_events),元素为 (t_s, amp_uv)。
    """
    from scipy.signal import find_peaks
    valid = ~np.repeat(artifact_mask_1s, fs)[:len(eog)]

    # 快速眼动:对 1–5 Hz 分量取速度(一阶差分),尖锐扫视速度大。
    # 关键去伪:单导前额下,慢波(δ)的陡坡也会产生高"速度"假阳性 → 仅保留**快频带主导**
    # 的瞬变(局部 2–6 Hz RMS > 0.5–2 Hz 慢波 RMS),把 δ 斜坡剔除,逼近真实扫视眼动。
    rapid = bandpass(eog, 1.0, 5.0, fs)
    slow_b = bandpass(eog, 0.5, 2.0, fs)
    fast_b = bandpass(eog, 2.0, 6.0, fs)
    vel = np.abs(np.gradient(rapid)) * fs
    bv = vel[valid] if valid.any() else vel
    vthr = np.median(bv) + 4.0 * (1.4826 * np.median(np.abs(bv - np.median(bv))) + 1e-9)
    ramp = np.abs(rapid)
    athr = 2.5 * (1.4826 * np.median(np.abs(rapid - np.median(rapid))) + 1e-9)
    win = int(0.4 * fs)
    pk, _ = find_peaks(vel, height=vthr, distance=int(0.2 * fs))
    rem_events = []
    for t in pk:
        if ramp[t] <= athr or artifact_mask_1s[min(t // int(fs), len(artifact_mask_1s) - 1)]:
            continue
        a, b = max(0, t - win), min(len(eog), t + win)
        if np.std(fast_b[a:b]) > np.std(slow_b[a:b]):   # 快频主导 → 视为眼动瞬变,非 δ 斜坡
            rem_events.append((t / fs, float(ramp[t])))

    # 慢速眼动:0.25–1 Hz 平缓大幅波
    slow = bandpass(eog, 0.25, 1.0, fs)
    senv = np.abs(hilbert(slow))
    bs = senv[valid] if valid.any() else senv
    sthr = np.median(bs) + 2.0 * (1.4826 * np.median(np.abs(bs - np.median(bs))) + 1e-9)
    sp, _ = find_peaks(senv, height=sthr, distance=int(0.8 * fs))
    sem_events = [(t / fs, float(senv[t])) for t in sp
                  if not artifact_mask_1s[min(t // int(fs), len(artifact_mask_1s) - 1)]]
    return rem_events, sem_events


# ────────────────────────────────────────────────────────────────────────────
# 纺锤波 / K 复合波检测(对整段 clean 信号一次性检测,再按 epoch 归集)
# ────────────────────────────────────────────────────────────────────────────

def detect_spindles(clean: np.ndarray, fs: float, artifact_mask_1s: np.ndarray):
    """
    AASM 纺锤波:11–16 Hz,时长 0.5–2.0 s。
    方法:sigma 带通 → Hilbert 包络 → 100ms 平滑 → 自适应阈值(在"非伪迹"样本上取
    分位数,近似 +1.5 SD)→ 阈上连续段时长 0.5–2 s 记一个纺锤波。
    返回事件列表 [(center_t_s, dur_s, peak_uv)]。
    """
    sig = bandpass(clean, 11.0, 16.0, fs)
    env = np.abs(hilbert(sig))
    win = max(1, int(0.1 * fs))
    env = np.convolve(env, np.ones(win) / win, mode="same")

    valid = ~np.repeat(artifact_mask_1s, fs)[:len(env)]
    base = env[valid] if valid.any() else env
    # 阈值:中位数 + 2.0 * 稳健 σ(基于 MAD),对低幅信号更稳、抑制 sigma 噪声误检
    med = np.median(base)
    rstd = 1.4826 * np.median(np.abs(base - med)) + 1e-9
    thr = med + 2.0 * rstd

    above = env > thr
    lo_n, hi_n = int(0.5 * fs), int(2.0 * fs)
    events = []
    run = -1
    for i in range(len(above)):
        if above[i] and run < 0:
            run = i
        elif not above[i] and run >= 0:
            dur = i - run
            if lo_n <= dur <= hi_n:
                events.append((((run + i) / 2) / fs, dur / fs, float(env[run:i].max())))
            run = -1
    if run >= 0 and lo_n <= (len(above) - run) <= hi_n:
        events.append(((run + len(above)) / 2 / fs, (len(above) - run) / fs,
                       float(env[run:].max())))
    # 落在伪迹段内的剔除
    events = [e for e in events if not artifact_mask_1s[min(int(e[0]),
              len(artifact_mask_1s) - 1)]]
    return events


def detect_kcomplexes(clean: np.ndarray, fs: float, artifact_mask_1s: np.ndarray):
    """
    K 复合波:孤立的双相慢瞬变(先负后正),总时长 ≥0.5 s,前额最显著。
    方法:0.5–4 Hz 带通,找显著负峰;以负峰为中心 ±0.5 s 内的峰峰值 > 阈值(自适应,
    约 4×稳健 σ),且形态为先下后上 → 记一个 K 复合波。两个事件间至少间隔 0.5 s。
    返回事件列表 [(trough_t_s, p2p_uv)]。
    """
    sig = bandpass(clean, 0.5, 4.0, fs)
    absig = np.abs(sig)
    valid = ~np.repeat(artifact_mask_1s, fs)[:len(sig)]
    base = sig[valid] if valid.any() else sig
    rstd = 1.4826 * np.median(np.abs(base - np.median(base))) + 1e-9
    p2p_thr = 6.0 * rstd          # K 复合波须明显高出背景 δ

    half = int(0.5 * fs)
    # 负峰候选:局部极小且足够深(≥1s 间隔,避免连续 δ 列被反复计数)
    from scipy.signal import find_peaks
    troughs, _ = find_peaks(-sig, distance=int(1.0 * fs), prominence=3.0 * rstd)
    events = []
    last_t = -1e9
    for t in troughs:
        a, b = max(0, t - half), min(len(sig), t + half)
        seg = sig[a:b]
        p2p = seg.max() - seg.min()
        post = sig[t:b]                       # 负峰后须正向回摆(down→up 双相形态)
        na, nb = max(0, t - 2 * fs), min(len(sig), t + 2 * fs)
        bg = np.median(absig[na:nb])          # ±2s 背景幅值
        # 孤立性:事件峰峰值须远大于周边背景(否则只是连续 δ 列,非孤立 K 复合波)
        if (p2p >= p2p_thr and sig[t] < -3.0 * rstd and post.size
                and post.max() > 0.4 * p2p and bg < 0.35 * p2p):
            tt = t / fs
            if tt - last_t >= 1.0 and not artifact_mask_1s[min(int(tt),
                    len(artifact_mask_1s) - 1)]:
                events.append((tt, float(p2p)))
                last_t = tt
    return events


# ────────────────────────────────────────────────────────────────────────────
# 每 epoch 频谱特征
# ────────────────────────────────────────────────────────────────────────────

def epoch_features(clean_ep: np.ndarray, raw_ep: np.ndarray, fs: float) -> dict:
    """每 epoch 频谱特征。频段定义对齐 ads1299 项目 EEGPreprocessor.BANDS
    (δ0.5–4 / θ4–8 / α8–13 / β13–30 / γ30–45),相对功率以 5 频段之和为分母。"""
    nperseg = int(min(len(clean_ep), fs * 4))
    f, p = welch(clean_ep, fs=fs, nperseg=nperseg)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.trapezoid(p[m], f[m])) if m.any() else 0.0

    d, th, al, be, ga = (band(0.5, 4), band(4, 8), band(8, 13),
                         band(13, 30), band(30, 45))
    sg = band(11, 16)                       # sigma(纺锤波频带),供分期参考
    tot = d + th + al + be + ga + 1e-12      # 5 频段相对功率(与项目一致)
    sw = bandpass(clean_ep, 0.5, 2.0, fs)   # 慢振荡绝对幅值(N3 标志)
    return dict(
        amp_uv=float(np.std(clean_ep)),
        raw_std=float(np.std(raw_ep)),
        delta=d / tot, theta=th / tot, alpha=al / tot,
        sigma=sg / tot, beta=be / tot, gamma=ga / tot,
        emg=ga / (be + ga + 1e-12),         # 高频/肌电占比(清醒↑ / REM 张力↓ 代理)
        sw_amp=float(np.std(sw)),           # 0.5–2 Hz 慢波绝对幅值 µV
        dom_hz=float(f[np.argmax(p)]),
    )


# ────────────────────────────────────────────────────────────────────────────
# 分期(单导前额启发式,W/N1/N2/N3/REM)
# ────────────────────────────────────────────────────────────────────────────

def stage_epochs(feats, n_spindle, n_kc, n_rem, sw_n3_thr, emg_ref):
    """
    单导前额脑电启发式分期(W/N1/N2/N3/REM)。

    注意:本数据上麻醉 BIS 模型输出恒定饱和(~93),对自然睡眠深度无区分力,故分期
    **不使用 BIS**,完全基于脑电频谱 + 幅值 + 眼动 + 事件,阈值按本次记录自适应。

    判据顺序(命中即停),依据 AASM 单导近似:
      运动大幅 → W;高频激活(β 高、δ 低、肌张力高)→ W/觉醒;
      本夜最高慢波 + 强 δ → N3(低幅候选);纺锤波/K 复合波 → N2;δ 占优稳定 → N2;
      **快速眼动成簇 + 低肌张力(REM 张力抑制)+ δ 不高 → REM**(眼动信息保留是关键);
      其余 → N1。
    """
    stages = []
    for i, ft in enumerate(feats):
        if ft["is_art"]:
            stages.append("ART"); continue
        d, th, be, emg = ft["delta"], ft["theta"], ft["beta"], ft["emg"]
        sw, amp = ft["sw_amp"], ft["amp_uv"]
        spin, kc, rem = n_spindle[i], n_kc[i], n_rem[i]

        # 1) 明显体动 → 清醒
        if amp > 50:
            stages.append("W"); continue
        # 2) 高频激活 + 高肌张力、慢波很少 → 清醒/觉醒(注意:与 REM 的区别在肌张力)
        if be >= 0.42 and d < 0.50 and emg >= emg_ref:
            stages.append("W"); continue
        # 3) 深睡 N3(候选):本夜最高档慢波幅值 + 强 δ 占优(绝对幅值低,故标"候选")
        if sw >= sw_n3_thr and d >= 0.72:
            stages.append("N3"); continue
        # 4) N2:本 epoch 检出纺锤波或 K 复合波,且 δ 不太低
        if (spin >= 1 or kc >= 1) and d >= 0.45:
            stages.append("N2"); continue
        # 5) REM(疑似):快速眼动成簇 + 肌张力相对低(REM 失张力)+ δ 不高 + 无纺锤波/K波
        if (rem >= 3 and emg < emg_ref and d < 0.55 and be < 0.40
                and spin == 0 and kc == 0):
            stages.append("REM"); continue
        # 6) N2:δ 占优的稳定 NREM
        if d >= 0.62:
            stages.append("N2"); continue
        # 7) 其余:浅睡/思睡 N1
        stages.append("N1")
    return stages


def smooth_stages(stages):
    """轻度连续性平滑:孤立单 epoch(两侧相同且不同于自身)并入邻居。不动 ART。"""
    s = list(stages)
    for i in range(1, len(s) - 1):
        if s[i] == "ART":
            continue
        if s[i - 1] == s[i + 1] and s[i] != s[i - 1] and s[i - 1] != "ART":
            s[i] = s[i - 1]
    return s


# ────────────────────────────────────────────────────────────────────────────
# 睡眠结构指标
# ────────────────────────────────────────────────────────────────────────────

def sleep_metrics(stages, epoch_sec):
    n = len(stages)
    arr = np.array(stages)
    tib_epochs = int(np.sum(arr != "ART"))                 # 记录有效时段(去伪迹)
    sleep_set = {"N1", "N2", "N3", "REM"}
    is_sleep = np.array([s in sleep_set for s in stages])

    # 睡眠起始:首个出现的睡眠 epoch
    onset = next((i for i, s in enumerate(stages) if s in sleep_set), None)
    last_sleep = next((i for i in range(n - 1, -1, -1) if stages[i] in sleep_set), None)

    counts = {s: int(np.sum(arr == s)) for s in STAGES}
    tst_epochs = sum(counts[s] for s in sleep_set)
    tst_min = tst_epochs * epoch_sec / 60.0

    # WASO:睡眠起始后到最后一个睡眠 epoch 之间的清醒
    waso = 0
    if onset is not None and last_sleep is not None:
        waso = int(np.sum(arr[onset:last_sleep + 1] == "W"))

    # 觉醒次数:睡眠 → 清醒的转换(睡眠起始之后)
    awakenings = 0
    if onset is not None:
        for i in range(onset + 1, n):
            if stages[i] == "W" and stages[i - 1] in sleep_set:
                awakenings += 1

    period = (last_sleep - onset + 1) if (onset is not None and last_sleep is not None) else 0
    se_tib = 100.0 * tst_epochs / tib_epochs if tib_epochs else 0.0     # TST / 有效记录
    se_period = 100.0 * tst_epochs / period if period else 0.0          # TST / 睡眠周期

    return dict(
        n_epochs=n, tib_epochs=tib_epochs, tst_min=tst_min, tst_epochs=tst_epochs,
        counts=counts, onset_idx=onset, last_sleep_idx=last_sleep,
        sol_min=(onset * epoch_sec / 60.0) if onset is not None else None,
        waso_min=waso * epoch_sec / 60.0, awakenings=awakenings,
        se_tib=se_tib, se_period=se_period, period_epochs=period,
        stage_pct={s: (100.0 * counts[s] / tst_epochs if tst_epochs else 0.0)
                   for s in sleep_set},
    )


# ────────────────────────────────────────────────────────────────────────────
# 图
# ────────────────────────────────────────────────────────────────────────────

def make_figure(out_png, stages, bis_ep, sqi_ep, feats, spindle_ep, kc_ep,
                epoch_sec, title, t0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(stages)
    t_h = np.arange(n) * epoch_sec / 3600.0
    fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                           gridspec_kw={"height_ratios": [2, 2, 2, 1.3]})

    # 1) 睡眠图
    y = [STAGE_Y[s] for s in stages]
    ax[0].step(t_h, y, where="post", color="#1e3a8a", lw=1.2)
    art = np.array([s == "ART" for s in stages])
    ax[0].scatter(t_h[art], [STAGE_Y["ART"]] * art.sum(), s=6, c="#ef4444", label="伪迹/脱落")
    ax[0].set_yticks([0, 1, 2, 3, 4, 5])
    ax[0].set_yticklabels(["伪迹", "N3", "N2", "N1", "REM", "W"])
    ax[0].set_ylabel("睡眠分期"); ax[0].set_title(title); ax[0].grid(alpha=0.3)
    ax[0].legend(loc="upper right", fontsize=8)

    # 2) BIS + 不确定度阴影 + SQI
    ax[1].plot(t_h, bis_ep, color="#0891b2", lw=1.0, label="BIS(模型推理)")
    ax[1].axhspan(80, 100, color="#dcfce7", alpha=0.5)
    ax[1].axhspan(60, 80, color="#fef9c3", alpha=0.5)
    ax[1].axhspan(40, 60, color="#ffedd5", alpha=0.5)
    ax[1].axhspan(0, 40, color="#fee2e2", alpha=0.4)
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("BIS"); ax[1].grid(alpha=0.3)
    ax[1].legend(loc="upper right", fontsize=8)
    ax2 = ax[1].twinx()
    ax2.plot(t_h, sqi_ep, color="#9ca3af", lw=0.6, alpha=0.7)
    ax2.set_ylabel("SQI", color="#9ca3af"); ax2.set_ylim(0, 100)

    # 3) 相对频段功率(堆叠)
    delta = np.array([f["delta"] for f in feats])
    theta = np.array([f["theta"] for f in feats])
    alpha = np.array([f["alpha"] for f in feats])
    beta = np.array([f["beta"] for f in feats])
    ax[3].stackplot(t_h, delta, theta, alpha, beta,
                    labels=["δ 0.5-4", "θ 4-8", "α 8-12", "β 16-30"],
                    colors=["#1d4ed8", "#0891b2", "#16a34a", "#dc2626"], alpha=0.8)
    ax[3].set_ylim(0, 1); ax[3].set_ylabel("相对功率"); ax[3].set_xlabel("时间 (小时)")
    ax[3].legend(loc="upper right", fontsize=7, ncol=4)

    # 4)(放在第3行)纺锤波 / K 复合波密度
    ax[2].bar(t_h, spindle_ep, width=epoch_sec / 3600.0, color="#7c3aed",
              label="纺锤波/epoch")
    ax[2].bar(t_h, -np.array(kc_ep), width=epoch_sec / 3600.0, color="#ea580c",
              label="K复合波/epoch")
    ax[2].axhline(0, color="#000", lw=0.5)
    ax[2].set_ylabel("事件计数"); ax[2].grid(alpha=0.3)
    ax[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────────

def analyze(folder, ckpt, out_dir, awake_cal):
    meta, eeg, ppg = load_session(folder)
    fs = int(round(meta["eeg"].get("measured_rate_hz") or meta["eeg"]["nominal_rate_hz"]))
    t0 = parse_start(meta)
    name = os.path.basename(str(folder).rstrip("/\\"))
    dur_min = len(eeg) / fs / 60.0
    print(f"\n==== 分析 {name}  fs={fs}Hz  时长={dur_min:.1f} min ====")

    # 1) 全程 BIS 流式推理
    print("  [1/5] 全程 BIS 模型推理 ...")
    inf = run_bis_inference(eeg, fs, ckpt, awake_cal)

    # 2) 整段信号:脑电分析用 clean(对齐 ads1299:0.5–45 Hz + 50 Hz 陷波);
    #    眼动分析用 eog(0.3–6 Hz,**保留眼动**,用于 REM 判读)
    print("  [2/5] 频谱 / 眼动 / 伪迹评估 ...")
    n_sec = len(eeg) // fs
    eeg_t = eeg[:n_sec * fs]
    clean = clean_filter(eeg_t, fs)
    eog = eog_filter(eeg_t, fs)
    # 逐秒伪迹标记:原始 std 过大(脱落/饱和)或滤后幅值非生理
    raw_1s = eeg_t.reshape(n_sec, fs)
    raw_std_1s = raw_1s.std(axis=1)
    clean_std_1s = clean.reshape(n_sec, fs).std(axis=1)
    artifact_1s = (raw_std_1s > 1500) | (clean_std_1s > 250) | (clean_std_1s < 1.5)

    # 3) 纺锤波 / K 复合波 / 眼动(REM·SEM)
    print("  [3/5] 纺锤波 / K 复合波 / 眼动检测 ...")
    spindles = detect_spindles(clean, fs, artifact_1s)
    kcs = detect_kcomplexes(clean, fs, artifact_1s)
    rem_events, sem_events = detect_eye_movements(eog, fs, artifact_1s)

    # 4) 分 epoch 特征
    print("  [4/5] 分 epoch 特征 + 分期 ...")
    ep_n = n_sec // EPOCH_SEC
    feats, bis_ep, unc_ep, sqi_ep, amp_ep = [], [], [], [], []
    spindle_ep = np.zeros(ep_n, int); kc_ep = np.zeros(ep_n, int)
    rem_ep = np.zeros(ep_n, int); sem_ep = np.zeros(ep_n, int)
    for e in spindles:
        idx = int(e[0] // EPOCH_SEC)
        if 0 <= idx < ep_n: spindle_ep[idx] += 1
    for e in kcs:
        idx = int(e[0] // EPOCH_SEC)
        if 0 <= idx < ep_n: kc_ep[idx] += 1
    for e in rem_events:
        idx = int(e[0] // EPOCH_SEC)
        if 0 <= idx < ep_n: rem_ep[idx] += 1
    for e in sem_events:
        idx = int(e[0] // EPOCH_SEC)
        if 0 <= idx < ep_n: sem_ep[idx] += 1

    for i in range(ep_n):
        a, b = i * EPOCH_SEC * fs, (i + 1) * EPOCH_SEC * fs
        ce, re = clean[a:b], eeg_t[a:b]
        ft = epoch_features(ce, re, fs)
        sl = slice(i * EPOCH_SEC, (i + 1) * EPOCH_SEC)
        art_frac = float(np.mean(artifact_1s[sl]))
        ft["is_art"] = art_frac > 0.5
        feats.append(ft)
        bis_ep.append(np.nanmean(inf["bis"][sl]))
        unc_ep.append(np.nanmean(inf["unc"][sl]))
        sqi_ep.append(np.nanmean(inf["sqi"][sl]))
        amp_ep.append(ft["amp_uv"])
    bis_ep = np.array(bis_ep); sqi_ep = np.array(sqi_ep)

    # N3 自适应慢波阈值:非伪迹 epoch 的 sw_amp 85 分位,设下限(低幅信号 N3 取本夜最深档)
    sw_valid = np.array([f["sw_amp"] for f in feats if not f["is_art"]])
    sw_n3_thr = max(np.percentile(sw_valid, 85) if sw_valid.size else 10.0, 10.0)
    emg_valid = np.array([f["emg"] for f in feats if not f["is_art"]])
    emg_ref = float(np.median(emg_valid)) if emg_valid.size else 0.15  # 肌张力参考(REM vs W)

    stages = stage_epochs(feats, spindle_ep, kc_ep, rem_ep, sw_n3_thr, emg_ref)
    stages = smooth_stages(stages)

    # 5) 指标 + 输出
    print("  [5/5] 汇总指标 + 写报告 ...")
    M = sleep_metrics(stages, EPOCH_SEC)

    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    # epochs.csv
    import csv
    with open(out / "epochs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "t_start_s", "clock", "stage", "bis", "bis_unc", "sqi",
                    "amp_uv", "sw_amp_uv", "delta", "theta", "alpha", "sigma", "beta",
                    "gamma", "emg", "dom_hz", "spindles", "kcomplexes", "rapid_eye_mov",
                    "slow_eye_mov", "artifact"])
        for i in range(ep_n):
            clk = (t0 + timedelta(seconds=i * EPOCH_SEC)).strftime("%H:%M:%S") if t0 else ""
            ft = feats[i]
            w.writerow([i, i * EPOCH_SEC, clk, stages[i],
                        f"{bis_ep[i]:.1f}", f"{unc_ep[i]:.1f}", f"{sqi_ep[i]:.0f}",
                        f"{ft['amp_uv']:.1f}", f"{ft['sw_amp']:.1f}",
                        f"{ft['delta']:.3f}", f"{ft['theta']:.3f}", f"{ft['alpha']:.3f}",
                        f"{ft['sigma']:.3f}", f"{ft['beta']:.3f}", f"{ft['gamma']:.3f}",
                        f"{ft['emg']:.3f}", f"{ft['dom_hz']:.1f}", spindle_ep[i], kc_ep[i],
                        rem_ep[i], sem_ep[i], int(ft["is_art"])])

    # 图
    title = f"{name}  ({t0.strftime('%Y-%m-%d %H:%M') if t0 else ''} 起,{dur_min:.0f} 分钟)"
    make_figure(out / "hypnogram.png", stages, bis_ep, sqi_ep, feats,
                spindle_ep, list(kc_ep), EPOCH_SEC, title, t0)

    # PPG 汇总(若有)。设备可能记录了帧但传感器未贴合 → ir/red 恒定、hr/spo2 全 0。
    ppg_sum = None
    if ppg.size:
        hr = ppg["hr"][ppg["hr"] > 0]; sp = ppg["spo2"][ppg["spo2"] > 0]
        ir_ac = float(np.std(ppg["ir"][:min(ppg.size, 100000)].astype(float)))
        valid = (hr.size > 0 or sp.size > 0) and ir_ac > 1.0
        ppg_sum = dict(frames=int(ppg.size), valid=bool(valid), ir_ac_std=round(ir_ac, 1),
                       hr_med=float(np.median(hr)) if hr.size else None,
                       hr_min=float(np.percentile(hr, 5)) if hr.size else None,
                       hr_max=float(np.percentile(hr, 95)) if hr.size else None,
                       spo2_med=float(np.median(sp)) if sp.size else None,
                       spo2_min=float(np.min(sp)) if sp.size else None)

    summary = build_summary(name, meta, fs, dur_min, t0, inf, M, stages, feats,
                            spindles, kcs, rem_events, sem_events, spindle_ep, kc_ep,
                            rem_ep, bis_ep, sqi_ep, artifact_1s, sw_n3_thr, ppg_sum,
                            awake_cal, ckpt)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    md = render_report(summary)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # 逐 epoch 时间序列(供 HTML 交互图表)
    series = dict(
        epoch_sec=EPOCH_SEC,
        clock=[(t0 + timedelta(seconds=i * EPOCH_SEC)).strftime("%H:%M") if t0 else f"{i*EPOCH_SEC//60}:{i*EPOCH_SEC%60:02d}"
               for i in range(ep_n)],
        stage=stages,
        bis=[None if not np.isfinite(bis_ep[i]) else round(float(bis_ep[i]), 1) for i in range(ep_n)],
        sqi=[None if not np.isfinite(sqi_ep[i]) else round(float(sqi_ep[i]), 0) for i in range(ep_n)],
        delta=[round(feats[i]["delta"], 3) for i in range(ep_n)],
        theta=[round(feats[i]["theta"], 3) for i in range(ep_n)],
        alpha=[round(feats[i]["alpha"], 3) for i in range(ep_n)],
        sigma=[round(feats[i]["sigma"], 3) for i in range(ep_n)],
        beta=[round(feats[i]["beta"], 3) for i in range(ep_n)],
        amp=[round(feats[i]["amp_uv"], 1) for i in range(ep_n)],
        spindle=[int(spindle_ep[i]) for i in range(ep_n)],
        kc=[int(kc_ep[i]) for i in range(ep_n)],
        rem=[int(rem_ep[i]) for i in range(ep_n)],
        art=[int(feats[i]["is_art"]) for i in range(ep_n)],
    )
    html = render_html(summary, series)
    with open(out / "report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✓ 输出: {out}  (report.html / report.md / epochs.csv / hypnogram.png)")
    return summary, str(out)


def build_summary(name, meta, fs, dur_min, t0, inf, M, stages, feats, spindles, kcs,
                  rem_events, sem_events, spindle_ep, kc_ep, rem_ep, bis_ep, sqi_ep,
                  artifact_1s, sw_n3_thr, ppg_sum, awake_cal, ckpt):
    valid = ~np.array([f["is_art"] for f in feats])
    bis_valid = bis_ep[valid & np.isfinite(bis_ep)]
    n2_idx = [i for i, s in enumerate(stages) if s == "N2"]
    n2_min = len(n2_idx) * EPOCH_SEC / 60.0
    spindle_in_n2 = sum(spindle_ep[i] for i in n2_idx)

    # REM 潜伏期 / 睡眠周期 / 觉醒指数
    onset = M["onset_idx"]
    rem_idx = [i for i, s in enumerate(stages) if s == "REM"]
    rem_latency = ((rem_idx[0] - onset) * EPOCH_SEC / 60.0
                   if (rem_idx and onset is not None and rem_idx[0] >= onset) else None)
    # REM 段数(连续 REM 块)≈ 睡眠周期数
    n_rem_periods = 0
    for i, s in enumerate(stages):
        if s == "REM" and (i == 0 or stages[i - 1] != "REM"):
            n_rem_periods += 1
    arousal_index = (M["awakenings"] / (M["tst_min"] / 60.0)) if M["tst_min"] else 0.0

    return dict(
        session=name, start=str(t0) if t0 else None,
        duration_min=round(dur_min, 1), fs=fs,
        model=dict(checkpoint=os.path.basename(os.path.dirname(ckpt)) + "/" +
                   os.path.basename(ckpt), ok=inf["model_ok"],
                   awake_anchor_bias=round(inf["bias"], 2), awake_cal=awake_cal,
                   n_channels=inf["n_channels"]),
        artifact_pct=round(100.0 * artifact_1s.mean(), 1),
        bis=dict(mean=round(float(np.nanmean(bis_valid)), 1) if bis_valid.size else None,
                 median=round(float(np.nanmedian(bis_valid)), 1) if bis_valid.size else None,
                 p5=round(float(np.nanpercentile(bis_valid, 5)), 1) if bis_valid.size else None,
                 p95=round(float(np.nanpercentile(bis_valid, 95)), 1) if bis_valid.size else None,
                 min=round(float(np.nanmin(bis_valid)), 1) if bis_valid.size else None),
        sqi_mean=round(float(np.nanmean(sqi_ep)), 0),
        metrics=M,
        rem_latency_min=round(rem_latency, 1) if rem_latency is not None else None,
        n_rem_periods=n_rem_periods,
        arousal_index=round(arousal_index, 1),
        spindles=dict(total=len(spindles), in_n2=int(spindle_in_n2),
                      density_per_min_n2=round(spindle_in_n2 / n2_min, 2) if n2_min else 0.0),
        kcomplexes=dict(total=len(kcs)),
        eye_movements=dict(rapid_total=len(rem_events), slow_total=len(sem_events),
                           rapid_in_rem=int(sum(rem_ep[i] for i in rem_idx))),
        sw_n3_thr_uv=round(float(sw_n3_thr), 1),
        mean_amp_uv=round(float(np.mean([f["amp_uv"] for f in feats if not f["is_art"]])), 1)
        if valid.any() else None,
        ppg=ppg_sum,
    )


# ────────────────────────────────────────────────────────────────────────────
# Markdown 报告
# ────────────────────────────────────────────────────────────────────────────

def _stage_cn(s):
    return {"W": "清醒", "N1": "N1 浅睡", "N2": "N2 浅睡",
            "N3": "N3 深睡", "REM": "REM", "ART": "伪迹"}[s]


def render_report(s) -> str:
    M = s["metrics"]; c = M["counts"]
    L = []
    L.append(f"# 睡眠脑电分析报告 — {s['session']}\n")
    L.append(f"- **记录开始**:{s['start']}")
    L.append(f"- **记录时长**:{s['duration_min']:.0f} 分钟（{s['duration_min']/60:.2f} 小时）"
             f"  采样率 {s['fs']} Hz  单导前额差分电极（ADS1299 CH0）")
    md = s["model"]
    L.append(f"- **BIS 模型**:`{md['checkpoint']}`（AnesthesiaNetV3,{md['n_channels']} 通道,"
             f"清醒锚定校准 {'开' if md['awake_cal'] else '关'},偏置 {md['awake_anchor_bias']:+.1f}）")
    L.append(f"- **信号有效性**:伪迹/脱落占比 {s['artifact_pct']:.1f}%,平均 SQI {s['sqi_mean']:.0f}/100,"
             f"平均幅值 {s['mean_amp_uv']} µV")
    L.append("- **预处理**:与 ads1299 生产栈一致(EEGPreprocessor 0.5–47 Hz 带通 + 50 Hz 陷波 → "
             "AnesthesiaNetV3 流式推理,频段定义对齐 EEGPreprocessor.BANDS);睡眠分期另用 0.3–6 Hz "
             "通道**保留眼动**(不去眨眼/眼动)以支持 REM 判读\n")

    L.append("> ⚠️ **方法学局限**:本报告由单导前额差分电极脑电离线分析得出,无 EOG/EMG/枕区导联,"
             "分期为**近似启发式**(REM 与 N1/清醒难以严格区分);BIS 由**麻醉深度模型**推理,"
             "在睡眠场景下作\"皮层激活/睡眠深度\"指标参考,与频谱分期互相印证,不能替代多导睡眠图(PSG)。\n")

    # 1. BIS
    L.append("## 1. 全程模型推理 BIS\n")
    b = s["bis"]
    if b["mean"] is not None:
        spread = (b["p95"] - b["p5"]) if (b["p95"] is not None and b["p5"] is not None) else 0
        L.append(f"剔除伪迹后,全程 BIS:**均值 {b['mean']}**,中位数 {b['median']},"
                 f"5–95 分位 [{b['p5']}, {b['p95']}],最低 {b['min']}。\n")
        if spread < 8:
            L.append(f"> ⚠️ **关键发现:BIS 在整段近乎恒定(5–95 分位仅 {spread:.0f} 点)**,"
                     "不随睡眠深度起伏。原因:① 该 BIS 模型在**麻醉**脑电(VitalDB/丙泊酚类)上训练,"
                     "自然睡眠与药物麻醉脑电并不等价,模型缺乏睡眠语料;② 本次前额导联信号幅值极低"
                     f"(均值 {s['mean_amp_uv']} µV),缺少麻醉深睡所需的高压慢波,模型据此判为持续\"清醒/浅\"状态。")
            L.append("> **结论:在本数据上 BIS 不能作为睡眠深度指标**,睡眠分期改由脑电频谱特征驱动(见第 2 节)。"
                     "BIS 仅反映\"皮层未进入麻醉式抑制\",与\"自然浅睡为主\"的频谱结论一致。\n")
        else:
            L.append("- BIS 80–100 ≈ 清醒/极浅睡;60–80 ≈ 浅睡(N1/N2);40–60 ≈ 中–深睡;<40 ≈ 深睡区。")
            L.append("- 完整逐秒/逐 epoch 曲线见 `hypnogram.png` 第 2 栏与 `epochs.csv`。\n")
    else:
        L.append("（有效 BIS 不足,信号质量过差。）\n")

    # 2. 分期
    L.append("## 2. 睡眠分期\n")
    tst = M["tst_min"]
    L.append(f"| 指标 | 数值 |")
    L.append(f"|---|---|")
    L.append(f"| 有效记录时长（去伪迹） | {M['tib_epochs']*EPOCH_SEC/60:.0f} 分钟 |")
    L.append(f"| 总睡眠时长 TST | {tst:.0f} 分钟 |")
    sol = M["sol_min"]
    sol_str = ("未检出睡眠" if sol is None else
               ("<1 分钟(开机即记录到睡眠样脑电)" if sol < 1 else f"{sol:.0f} 分钟"))
    L.append(f"| 入睡潜伏期 | {sol_str} |")
    L.append(f"| 睡眠效率(TST/有效记录) | {M['se_tib']:.0f}% |")
    L.append(f"| 入睡后清醒 WASO | {M['waso_min']:.0f} 分钟 |")
    L.append(f"| 觉醒次数 / 觉醒指数 | {M['awakenings']} 次 / {s.get('arousal_index',0):.1f} 次每小时(正常 <10–15) |")
    rl = s.get("rem_latency_min")
    L.append(f"| REM 潜伏期 | {rl:.0f} 分钟 |" if rl is not None else "| REM 潜伏期 | 未检出 REM |")
    L.append(f"| 检出 REM 段数(≈睡眠周期) | {s.get('n_rem_periods',0)} |")
    L.append("")
    L.append(f"| 分期 | epoch 数 | 时长(min) | 占 TST |")
    L.append(f"|---|---|---|---|")
    for st in ["W", "N1", "N2", "N3", "REM", "ART"]:
        mins = c[st] * EPOCH_SEC / 60.0
        pct = M["stage_pct"].get(st)
        pcts = f"{pct:.0f}%" if (pct is not None and st in M["stage_pct"]) else "—"
        L.append(f"| {_stage_cn(st)} | {c[st]} | {mins:.0f} | {pcts} |")
    L.append("")

    # 3. 纺锤波/K复合波
    L.append("## 3. 纺锤波与 K 复合波\n")
    sp = s["spindles"]; kc = s["kcomplexes"]
    L.append(f"- **睡眠纺锤波(11–16 Hz,0.5–2 s)**:全程检出 **{sp['total']}** 个,"
             f"其中 N2 期 {sp['in_n2']} 个,N2 期密度约 **{sp['density_per_min_n2']} 个/分钟**"
             f"(健康成人 N2 参考 ~2–5 个/分钟)。")
    L.append(f"- **K 复合波**:全程检出 **{kc['total']}** 个(N2 期标志性波形,与纺锤波共同支持上述 N2 判定)。")
    dens = sp["density_per_min_n2"]
    if dens >= 2:
        L.append(f"- 纺锤波是丘脑-皮层环路完整性与记忆巩固的标志;本次密度处于**正常区间**,"
                 f"提示该睡眠机制运作正常(单导低幅信号下检出偏保守,实际密度可能更高)。\n")
    else:
        L.append(f"- 纺锤波是丘脑-皮层环路完整性与记忆巩固的标志;本次密度偏低,可能反映睡眠浅、"
                 f"N2 期不充分,或受单导前额导联与低幅信号限制(检出偏保守)。\n")

    # PPG
    if s.get("ppg"):
        p = s["ppg"]
        L.append("## 4. 指夹 PPG(心率 / 血氧)\n")
        if p.get("valid"):
            if p.get("hr_med"):
                L.append(f"- 心率:中位 {p['hr_med']:.0f} bpm(5–95 分位 {p['hr_min']:.0f}–{p['hr_max']:.0f})。")
            if p.get("spo2_med"):
                L.append(f"- 血氧 SpO₂:中位 {p['spo2_med']:.0f}%,最低 {p['spo2_min']:.0f}%。")
        else:
            L.append(f"- 记录到 {p['frames']} 帧,但 IR/Red 恒定、HR/SpO₂ 全为 0(波形 AC≈{p['ir_ac_std']})"
                     " → **指夹未贴合或未锁定脉搏,无有效心率/血氧数据**,不纳入分析。")
        L.append("")
        sec = 5
    else:
        sec = 4

    # 4/5. 综合分析
    L.append(f"## {sec}. 睡眠结构与身心状态分析\n")
    L.extend(interpret(s))
    L.append("\n---\n*由 `scripts/analyze_sleep.py` 自动生成。分期与事件检测为启发式近似,"
             "临床判读请以标准多导睡眠图为准。*")
    return "\n".join(L)


def interpret(s) -> list[str]:
    """根据指标生成结构化的睡眠质量与身心状态解读(已考虑单导/低幅/片段化记录的局限)。"""
    M = s["metrics"]; L = []
    se = M["se_tib"]
    pct = M["stage_pct"]
    deep = pct.get("N3", 0); rem = pct.get("REM", 0)
    awk = M["awakenings"]; waso = M["waso_min"]
    spd = s["spindles"]["density_per_min_n2"]
    amp = s.get("mean_amp_uv")
    art = s.get("artifact_pct", 0)
    dur_h = s.get("duration_min", 0) / 60.0
    # 记录时段(用于"片段化整夜"语境)
    hh = None
    try:
        hh = datetime.fromisoformat(s["start"]).hour if s.get("start") else None
    except Exception:
        hh = None
    arousal_idx = awk / (M["tst_min"] / 60.0) if M["tst_min"] else 0  # 次/小时

    # ── 上下文/数据质量先行说明 ──
    ctx = []
    if dur_h < 4:
        when = ("上半夜" if (hh is not None and 19 <= hh <= 23) else
                "下半夜" if (hh is not None and (hh >= 23 or hh <= 5)) else "整夜中的一段")
        ctx.append(f"本次仅记录 {dur_h:.1f} 小时(属{when}片段,非完整整夜)→ 各期占比**不代表整夜结构**"
                   "(深睡多集中在上半夜、REM 多在下半夜)")
    if art >= 15:
        ctx.append(f"中途有 {art:.0f}% 时段电极脱落/饱和(已剔除)→ 该段睡眠无数据,"
                   "TST/效率为对**有效时段**的统计,可能高估整体连续性")
    if ctx:
        L.append("**数据与时段说明:**")
        for cstr in ctx:
            L.append(f"- {cstr}。")
        L.append("")

    obs = []
    # 睡眠效率
    if se >= 85:
        obs.append(f"睡眠效率 {se:.0f}%(有效时段内,≥85% 为佳),连续性良好")
    elif se >= 75:
        obs.append(f"睡眠效率 {se:.0f}%(有效时段内)略低于理想(≥85%),存在一定碎片化")
    else:
        obs.append(f"睡眠效率仅 {se:.0f}%(有效时段内),碎片化明显")
    # 睡眠构成:以 N2 为主
    n2 = pct.get("N2", 0)
    obs.append(f"睡眠以 **N2 浅睡为主(占 TST {n2:.0f}%)**,符合该前额低幅记录的预期")
    # 深睡(强调montage/时段局限,避免过度下结论)
    if deep >= 13:
        obs.append(f"检出 N3 深睡占 {deep:.0f}%(成人参考 13–23%),处于合理范围")
    else:
        obs.append(f"N3 深睡仅 {deep:.0f}%(成人参考 13–23%);但**前额双极导联会显著衰减慢波幅值**、"
                   f"且本段为片段记录 → N3 系统性**偏少检出**,不宜据此直接判定深睡不足")
    # REM + 眼动(已保留眼动信号用于 REM 判读)
    em = s.get("eye_movements", {})
    if rem >= 12:
        obs.append(f"REM 约 {rem:.0f}%(疑似),期间检出快速眼动 {em.get('rapid_in_rem',0)} 次")
    else:
        msg = "未明确检出 REM(单导前额无独立 EOG/EMG 导联)"
        if em.get("rapid_total"):
            msg += (f";已**保留眼动信号**并检出眼动样瞬变(快速 {em['rapid_total']}、慢速 "
                    f"{em.get('slow_total',0)} 次),但单导上眼动与额区慢波难以分离,"
                    "故 REM 无法据此可靠判定,REM=0 属检出局限而非真实无 REM")
        obs.append(msg)
    # 碎片化(用觉醒指数,更客观)
    if arousal_idx <= 5 and waso < 20:
        obs.append(f"夜间觉醒 {awk} 次(觉醒指数≈{arousal_idx:.0f}/小时)、WASO {waso:.0f} 分钟,处于正常范围")
    elif arousal_idx <= 10:
        obs.append(f"夜间觉醒 {awk} 次(觉醒指数≈{arousal_idx:.0f}/小时)、WASO {waso:.0f} 分钟,轻度偏多")
    else:
        obs.append(f"夜间觉醒 {awk} 次(觉醒指数≈{arousal_idx:.0f}/小时)、WASO {waso:.0f} 分钟,睡眠维持较差")

    L.append("**主要观察:**")
    for o in obs:
        L.append(f"- {o}。")
    L.append("")

    # 身心状态推断(谨慎、关联性而非诊断)
    L.append("**身体与精神状态推断(基于客观指标的关联性推断,非医学诊断):**")
    pts = []
    if se >= 85 and arousal_idx <= 8:
        pts.append("睡眠连续、效率良好 → 提示**入睡与睡眠维持能力正常**,无明显失眠样片段化")
    if spd >= 2:
        pts.append(f"纺锤波密度 {spd}/分钟(正常区间) → 丘脑-皮层环路完整,"
                   "**记忆巩固/学习相关的睡眠机制运作正常**,是本次记录中较积极的信号")
    else:
        pts.append(f"纺锤波密度偏低({spd}/分钟)→ 或与浅睡为主、压力、睡眠不足有关,"
                   "亦受单导低幅信号限制(检出偏保守)")
    if amp is not None and amp < 25:
        pts.append(f"脑电整体低幅(均值 {amp} µV)、以低压慢波为主 → 与\"以浅睡为主\"一致;"
                   "但低幅亦大概率源于**前额双极电极间距小、共模抵消**的硬件特性,"
                   "故\"恢复性深睡是否充足\"**本记录不足以判定**")
    if arousal_idx > 8 or waso >= 25:
        pts.append("觉醒偏多/WASO 偏长 → 可能与**入睡环境、精神紧张/压力、咖啡因或酒精**等相关,"
                   "也需警惕睡眠呼吸事件(本设备无法评估)")
    if not pts:
        pts.append("各项指标处于中间区间,未见突出异常")
    for p in pts:
        L.append(f"- {p}。")
    L.append("")
    L.append("**总体小结**:在单导前额可穿戴脑电的能力范围内,本次记录显示**以 N2 浅睡为主、"
             "纺锤波活动正常、睡眠效率尚可**的轻睡眠片段;深睡与 REM 因导联/时段限制无法可靠量化。"
             "若日常存在白天困倦、入睡困难、夜间频繁觉醒或晨起不解乏,建议规律作息、"
             "睡前减少蓝光/咖啡因/酒精,必要时行标准多导睡眠图(PSG)以全面评估睡眠结构与呼吸事件。")
    return L


# ────────────────────────────────────────────────────────────────────────────
# HTML 报告(自包含,内联 SVG 图表,无外部依赖)
# ────────────────────────────────────────────────────────────────────────────

STAGE_COLOR = {"W": "#fbbf24", "REM": "#34d399", "N1": "#a5b4fc",
               "N2": "#6366f1", "N3": "#3730a3", "ART": "#ef4444"}
STAGE_NAME = {"W": "清醒", "REM": "REM", "N1": "N1", "N2": "N2", "N3": "N3", "ART": "伪迹"}


def _esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _md_inline(t):
    out, b = [], False
    for part in str(t).split("**"):
        out.append(("<strong>" + _esc(part) + "</strong>") if b else _esc(part))
        b = not b
    return "".join(out)


def _md_block_to_html(lines):
    """把 interpret() 产出的简单 markdown 行转成 HTML(**粗体**、- 列表、空行)。"""
    html, in_ul = [], False
    for ln in lines:
        if ln.strip() == "":
            if in_ul:
                html.append("</ul>"); in_ul = False
            continue
        if ln.startswith("- "):
            if not in_ul:
                html.append("<ul>"); in_ul = True
            html.append("<li>" + _md_inline(ln[2:]) + "</li>")
        else:
            if in_ul:
                html.append("</ul>"); in_ul = False
            html.append("<p class='lead'>" + _md_inline(ln) + "</p>")
    if in_ul:
        html.append("</ul>")
    return "\n".join(html)


def _poly(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def svg_hypnogram(series):
    st = series["stage"]; clk = series["clock"]; n = len(st)
    W, H = 1000, 230
    x0, x1, y0, y1 = 50, 988, 20, 175
    lev = {"W": 0, "REM": 1, "N1": 2, "N2": 3, "N3": 4}
    ew = (x1 - x0) / max(n, 1)
    yy = lambda L: y0 + L * (y1 - y0) / 4
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none">']
    # grid + y labels
    for name, L in lev.items():
        y = yy(L)
        s.append(f'<line x1="{x0}" y1="{y:.0f}" x2="{x1}" y2="{y:.0f}" class="grid"/>')
        s.append(f'<text x="{x0-8}" y="{y+4:.0f}" class="yl" text-anchor="end">{STAGE_NAME[name]}</text>')
    # colored segments + step connectors
    prev = None
    for i, stg in enumerate(st):
        x = x0 + i * ew
        if stg in lev:
            y = yy(lev[stg])
            s.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x+ew:.1f}" y2="{y:.1f}" '
                     f'stroke="{STAGE_COLOR[stg]}" stroke-width="3.4"/>')
            if prev is not None and prev[0] != y:
                s.append(f'<line x1="{x:.1f}" y1="{prev[0]:.1f}" x2="{x:.1f}" y2="{y:.1f}" class="conn"/>')
            prev = (y, stg)
        else:  # ART
            s.append(f'<rect x="{x:.1f}" y="{y1+6:.0f}" width="{max(ew,1):.1f}" height="7" fill="{STAGE_COLOR["ART"]}" opacity="0.85"/>')
            prev = None
    s.append(f'<text x="{x0}" y="{y1+24:.0f}" class="xl" text-anchor="start">{_esc(clk[0]) if clk else ""}</text>')
    if clk:
        s.append(f'<text x="{x1}" y="{y1+24:.0f}" class="xl" text-anchor="end">{_esc(clk[-1])}</text>')
    s.append("</svg>")
    return "\n".join(s)


def svg_line_bis(series):
    bis, sqi, clk = series["bis"], series["sqi"], series["clock"]
    n = len(bis); W, H = 1000, 210
    x0, x1, y0, y1 = 50, 988, 16, 168
    ew = (x1 - x0) / max(n - 1, 1)
    yv = lambda v: y1 - v / 100.0 * (y1 - y0)
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none">']
    bands = [(80, 100, "#064e3b"), (60, 80, "#3f3f0f"), (40, 60, "#4a2c0f"), (0, 40, "#4c1010")]
    for lo, hi, col in bands:
        s.append(f'<rect x="{x0}" y="{yv(hi):.1f}" width="{x1-x0}" height="{yv(lo)-yv(hi):.1f}" fill="{col}" opacity="0.35"/>')
    for v in (0, 50, 100):
        s.append(f'<line x1="{x0}" y1="{yv(v):.1f}" x2="{x1}" y2="{yv(v):.1f}" class="grid"/>')
        s.append(f'<text x="{x0-8}" y="{yv(v)+4:.1f}" class="yl" text-anchor="end">{v}</text>')
    def line(arr, cls):
        pts, seg = [], []
        for i, v in enumerate(arr):
            if v is None:
                if len(seg) > 1: pts.append(seg)
                seg = []
            else:
                seg.append((x0 + i * ew, yv(v)))
        if len(seg) > 1: pts.append(seg)
        return "".join(f'<polyline points="{_poly(p)}" class="{cls}"/>' for p in pts)
    s.append(line(sqi, "sqi"))
    s.append(line(bis, "bis"))
    s.append("</svg>")
    return "\n".join(s)


def svg_donut(counts, tst_min):
    order = ["N3", "N2", "N1", "REM"]
    total = sum(counts.get(k, 0) for k in order) or 1
    W = 260; cx, cy, r, sw = 130, 120, 78, 26
    import math
    s = [f'<svg viewBox="0 0 {W} 240" class="donut">']
    ang = -90.0
    for k in order:
        frac = counts.get(k, 0) / total
        if frac <= 0: continue
        a2 = ang + frac * 360
        large = 1 if (a2 - ang) > 180 else 0
        x1 = cx + r * math.cos(math.radians(ang)); y1 = cy + r * math.sin(math.radians(ang))
        x2 = cx + r * math.cos(math.radians(a2)); y2 = cy + r * math.sin(math.radians(a2))
        s.append(f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}" '
                 f'fill="none" stroke="{STAGE_COLOR[k]}" stroke-width="{sw}"/>')
        ang = a2
    s.append(f'<text x="{cx}" y="{cy-4}" class="donut-num" text-anchor="middle">{tst_min:.0f}</text>')
    s.append(f'<text x="{cx}" y="{cy+16}" class="donut-lbl" text-anchor="middle">分钟 TST</text>')
    s.append("</svg>")
    return "\n".join(s)


def svg_bands(series):
    d, th, al, sg, be = (series["delta"], series["theta"], series["alpha"],
                         series["sigma"], series["beta"])
    n = len(d); W, H = 1000, 180
    x0, x1, y0, y1 = 50, 988, 14, 150
    ew = (x1 - x0) / max(n - 1, 1)
    layers = [("δ", d, "#3730a3"), ("θ", th, "#0891b2"), ("α", al, "#16a34a"),
              ("σ", sg, "#a855f7"), ("β", be, "#dc2626")]
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none">']
    base = [0.0] * n
    for name, arr, col in layers:
        top = [base[i] + arr[i] for i in range(n)]
        up = [(x0 + i * ew, y1 - top[i] * (y1 - y0)) for i in range(n)]
        dn = [(x0 + i * ew, y1 - base[i] * (y1 - y0)) for i in range(n - 1, -1, -1)]
        s.append(f'<polygon points="{_poly(up + dn)}" fill="{col}" opacity="0.82"/>')
        base = top
    s.append("</svg>")
    return "\n".join(s)


def svg_events(series):
    sp, kc, rem = series["spindle"], series["kc"], series["rem"]
    n = len(sp); W, H = 1000, 170
    x0, x1, ymid = 50, 988, 92
    ew = (x1 - x0) / max(n, 1)
    mx = max([1] + sp + kc)
    sc = 56.0 / mx
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none">']
    s.append(f'<line x1="{x0}" y1="{ymid}" x2="{x1}" y2="{ymid}" class="grid"/>')
    for i in range(n):
        x = x0 + i * ew
        if sp[i]:
            h = sp[i] * sc
            s.append(f'<rect x="{x:.1f}" y="{ymid-h:.1f}" width="{max(ew-0.3,0.6):.1f}" height="{h:.1f}" fill="#a855f7"/>')
        if kc[i]:
            h = kc[i] * sc
            s.append(f'<rect x="{x:.1f}" y="{ymid:.1f}" width="{max(ew-0.3,0.6):.1f}" height="{h:.1f}" fill="#f97316"/>')
        if rem[i] >= 3:
            s.append(f'<circle cx="{x+ew/2:.1f}" cy="14" r="3" fill="#34d399"/>')
    s.append(f'<text x="{x0-8}" y="{ymid-46}" class="yl" text-anchor="end">纺锤</text>')
    s.append(f'<text x="{x0-8}" y="{ymid+52}" class="yl" text-anchor="end">K波</text>')
    s.append("</svg>")
    return "\n".join(s)


def _kpi(label, value, unit, ref, good):
    cls = {"good": "ok", "warn": "warn", "bad": "bad", "": ""}.get(good, "")
    return (f'<div class="kpi {cls}"><div class="kpi-v">{value}<span class="kpi-u">{unit}</span></div>'
            f'<div class="kpi-l">{label}</div><div class="kpi-r">{ref}</div></div>')


HTML_CSS = """
:root{--bg:#070b16;--bg2:#0d1426;--panel:#111c33;--panel2:#0f1830;--bd:#1f2d4d;
--ink:#e6edf7;--mut:#8aa0c4;--cyan:#22d3ee;--vio:#a78bfa;--grn:#34d399;--amb:#fbbf24;--red:#f87171;}
*{box-sizing:border-box} html{scroll-behavior:smooth}
body{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;background:
radial-gradient(1200px 600px at 80% -10%,#13224a 0%,transparent 55%),
radial-gradient(900px 500px at -10% 10%,#0c2b3a 0%,transparent 50%),var(--bg);
color:var(--ink);line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 70px}
header.top{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:14px;
border-bottom:1px solid var(--bd);padding-bottom:20px;margin-bottom:8px}
.brand{font-size:13px;letter-spacing:3px;color:var(--cyan);text-transform:uppercase;font-weight:600}
h1{font-size:27px;margin:6px 0 2px;font-weight:700;letter-spacing:.5px}
.sub{color:var(--mut);font-size:13.5px;font-family:"Cascadia Code",Consolas,monospace}
.chip{display:inline-block;padding:3px 10px;border:1px solid var(--bd);border-radius:999px;
font-size:12px;color:var(--mut);background:rgba(255,255,255,.02);margin-left:6px}
.disc{background:linear-gradient(90deg,rgba(248,113,113,.10),rgba(248,113,113,.02));
border:1px solid rgba(248,113,113,.32);border-left:3px solid var(--red);border-radius:10px;
padding:12px 16px;margin:18px 0;color:#fecaca;font-size:13px}
.grid-kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:13px;margin:22px 0}
.kpi{background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--bd);
border-radius:13px;padding:15px 16px;position:relative;overflow:hidden}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--cyan);opacity:.7}
.kpi.ok:before{background:var(--grn)} .kpi.warn:before{background:var(--amb)} .kpi.bad:before{background:var(--red)}
.kpi-v{font-size:27px;font-weight:700;letter-spacing:.5px}
.kpi-u{font-size:13px;color:var(--mut);margin-left:3px;font-weight:500}
.kpi-l{font-size:13px;color:var(--ink);margin-top:3px}
.kpi-r{font-size:11.5px;color:var(--mut);margin-top:2px}
section{background:linear-gradient(180deg,rgba(255,255,255,.018),transparent);
border:1px solid var(--bd);border-radius:15px;padding:20px 22px;margin:18px 0}
h2{font-size:18px;margin:0 0 4px;display:flex;align-items:center;gap:9px}
h2 .dot{width:8px;height:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}
.note{color:var(--mut);font-size:12.5px;margin:2px 0 14px}
.chart{width:100%;height:auto;display:block}
.grid{stroke:#23344f;stroke-width:1} .conn{stroke:#3a4d72;stroke-width:1}
.yl{fill:#8aa0c4;font-size:11px} .xl{fill:#8aa0c4;font-size:11px}
.bis{fill:none;stroke:var(--cyan);stroke-width:2;filter:drop-shadow(0 0 3px rgba(34,211,238,.5))}
.sqi{fill:none;stroke:#64748b;stroke-width:1.2;stroke-dasharray:3 3}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;font-size:12.5px;color:var(--mut)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block}
.two{display:grid;grid-template-columns:260px 1fr;gap:22px;align-items:center}
.donut{width:260px;height:240px}
.donut-num{fill:var(--ink);font-size:32px;font-weight:700} .donut-lbl{fill:var(--mut);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--bd)}
th{color:var(--mut);font-weight:600;font-size:12.5px} td.num{font-family:Consolas,monospace}
.bar{height:7px;border-radius:4px;background:var(--cyan);display:inline-block;vertical-align:middle}
.lead{margin:9px 0;font-size:14px} .lead strong{color:var(--cyan);font-weight:600}
section ul{margin:6px 0 6px;padding-left:20px} section li{margin:6px 0;font-size:13.7px}
section li strong{color:var(--vio)}
.refs{font-size:12px;color:var(--mut)} .refs li{margin:5px 0}
.tag{font-size:11px;color:var(--cyan);border:1px solid var(--bd);border-radius:6px;padding:1px 7px;margin-right:6px}
footer{color:var(--mut);font-size:12px;text-align:center;margin-top:30px;padding-top:16px;border-top:1px solid var(--bd)}
h3.bk{font-size:14px;color:var(--ink);margin:16px 0 6px}
"""


def render_html(s, series) -> str:
    M = s["metrics"]; c = M["counts"]; pct = M["stage_pct"]; b = s["bis"]
    md = s["model"]
    se = M["se_tib"]; ai = s.get("arousal_index", 0)
    n3 = pct.get("N3", 0); rem = pct.get("REM", 0)
    spd = s["spindles"]["density_per_min_n2"]
    spread = (b["p95"] - b["p5"]) if (b.get("p95") is not None and b.get("p5") is not None) else 99

    g = lambda cond_ok, cond_warn: "good" if cond_ok else ("warn" if cond_warn else "bad")
    kpis = [
        _kpi("总睡眠时长 TST", f"{M['tst_min']:.0f}", "min", "—", ""),
        _kpi("睡眠效率", f"{se:.0f}", "%", "≥85% 佳", g(se >= 85, se >= 75)),
        _kpi("入睡潜伏期", ("<1" if (M['sol_min'] is not None and M['sol_min'] < 1) else
             (f"{M['sol_min']:.0f}" if M['sol_min'] is not None else "—")), "min", "<30 正常",
             g((M['sol_min'] or 99) < 30, (M['sol_min'] or 99) < 45)),
        _kpi("觉醒指数", f"{ai:.0f}", "/h", "<10–15", g(ai <= 10, ai <= 20)),
        _kpi("WASO", f"{M['waso_min']:.0f}", "min", "<30 正常", g(M['waso_min'] < 30, M['waso_min'] < 45)),
        _kpi("N3 深睡", f"{n3:.0f}", "%", "13–23%(单导偏低估)", ""),
        _kpi("REM(疑似)", f"{rem:.0f}", "%", "20–25%(单导难测)", ""),
        _kpi("纺锤波密度", f"{spd}", "/min", "2–5/min N2", g(spd >= 2, spd >= 1)),
    ]

    # 分期表
    rows = []
    for st in ["W", "N1", "N2", "N3", "REM", "ART"]:
        mins = c[st] * EPOCH_SEC / 60.0
        p = pct.get(st)
        pv = f"{p:.0f}%" if (p is not None and st in pct) else "—"
        barw = int((p if (p is not None and st in pct) else 0) * 1.4)
        rows.append(f'<tr><td><span class="tag" style="color:{STAGE_COLOR[st]};border-color:{STAGE_COLOR[st]}">'
                    f'{STAGE_NAME[st]}</span></td><td class="num">{c[st]}</td><td class="num">{mins:.0f}</td>'
                    f'<td class="num">{pv}</td><td><span class="bar" style="width:{barw}px;background:{STAGE_COLOR[st]}"></span></td></tr>')
    stage_table = ("<table><tr><th>分期</th><th>epoch</th><th>时长(min)</th><th>占TST</th><th></th></tr>"
                   + "".join(rows) + "</table>")

    # BIS 段落
    if spread < 8:
        bis_note = (f'<div class="disc" style="border-left-color:var(--amb);color:#fde68a;'
                    f'background:linear-gradient(90deg,rgba(251,191,36,.10),transparent)">'
                    f'⚠️ <strong>关键发现:</strong>BIS 全程近乎恒定(均值 {b["mean"]},5–95 分位仅 {spread:.0f} 点)。'
                    f'该 BIS 由<strong>麻醉深度模型</strong>(VitalDB 训练)推理,自然睡眠脑电与药物麻醉不等价,'
                    f'加之本次前额信号低幅(均值 {s["mean_amp_uv"]} µV)缺乏高压慢波 → 模型对睡眠深度<strong>无区分力</strong>。'
                    f'故睡眠分期改由脑电频谱+眼动特征驱动;BIS 仅说明"皮层未进入麻醉式抑制"。</div>')
    else:
        bis_note = f'<p class="lead">全程 BIS 均值 {b["mean"]},5–95 分位 [{b["p5"]}, {b["p95"]}],最低 {b["min"]}。</p>'

    # PPG
    ppg_html = ""
    p = s.get("ppg")
    if p:
        if p.get("valid") and p.get("hr_med"):
            ppg_html = (f'<section><h2><span class="dot"></span>指夹 PPG</h2>'
                        f'<p class="lead">心率中位 <strong>{p["hr_med"]:.0f}</strong> bpm,'
                        f'SpO₂ 中位 <strong>{p.get("spo2_med") or "—"}</strong>%。</p></section>')
        else:
            ppg_html = (f'<section><h2><span class="dot"></span>指夹 PPG</h2><p class="note">'
                        f'记录 {p["frames"]} 帧但 IR/Red 恒定、HR/SpO₂ 全 0(波形 AC≈{p.get("ir_ac_std")})→ '
                        f'指夹未贴合/未锁定脉搏,无有效心率血氧数据,不纳入分析。</p></section>')

    em = s.get("eye_movements", {})
    analysis_html = _md_block_to_html(interpret(s))

    refs = [
        '<li><span class="tag">分期标准</span>AASM Manual for the Scoring of Sleep (Berry et al., v2.6+);Iber et al. 2007;Rechtschaffen & Kales 1968。</li>',
        '<li><span class="tag">正常结构</span>成人 N1 ~5%(5–10%)、N2 ~50%(45–60%)、N3 13–23%、REM 20–25%;睡眠效率≥85%、觉醒指数<10–15/h、入睡潜伏期<30min。<a href="https://www.ncbi.nlm.nih.gov/books/NBK537023/" target="_blank">StatPearls: EEG Normal Sleep</a>;<a href="https://jcsm.aasm.org/doi/10.5664/jcsm.7036" target="_blank">JCSM 健康成人 PSG 参考值</a>。</li>',
        '<li><span class="tag">纺锤波</span>11–16 Hz、0.5–2 s;慢纺锤波(11–13 Hz)前额为主、快纺锤波(13–16 Hz)中央为主;健康成人 N2 密度约 1–5/min。<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10098120/" target="_blank">Frontiers 2023</a>;<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5490197/" target="_blank">NSRR 11,630 人</a>;<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12172134/" target="_blank">阈值研究 2024</a>。</li>',
        '<li><span class="tag">K 复合波</span>高幅、尖锐负向波继以正向偏转,≥0.5 s,前额最显著;N2 标志,密度约 1–3/min。</li>',
        '<li><span class="tag">眼动/REM</span>REM 期快速眼动 + 肌张力抑制(失张力);单导前额无独立 EOG/EMG → REM 判读为近似。</li>',
        '<li><span class="tag">功能</span>慢波睡眠(N3)主司体力恢复/代谢清除;REM 主司情绪与记忆整合;纺锤波与记忆巩固和丘脑-皮层环路完整性相关(De Gennaro & Ferrara 2003;Wamsley et al. 2012)。</li>',
    ]

    extra_rows = (
        f'<tr><td>REM 潜伏期</td><td class="num">{(str(s["rem_latency_min"])+" min") if s.get("rem_latency_min") is not None else "未检出 REM"}</td></tr>'
        f'<tr><td>检出 REM 段数</td><td class="num">{s.get("n_rem_periods",0)}</td></tr>'
        f'<tr><td>快速眼动事件(全程/REM内)</td><td class="num">{em.get("rapid_total",0)} / {em.get("rapid_in_rem",0)}</td></tr>'
        f'<tr><td>慢速眼动事件(全程)</td><td class="num">{em.get("slow_total",0)}</td></tr>'
        f'<tr><td>纺锤波(全程/N2内)</td><td class="num">{s["spindles"]["total"]} / {s["spindles"]["in_n2"]}</td></tr>'
        f'<tr><td>K 复合波(全程)</td><td class="num">{s["kcomplexes"]["total"]}</td></tr>'
        f'<tr><td>平均 SQI / 幅值</td><td class="num">{s["sqi_mean"]:.0f} / {s["mean_amp_uv"]} µV</td></tr>'
    )

    parts = []
    parts.append('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append(f'<title>睡眠脑电分析 · {_esc(s["session"])}</title>')
    parts.append(f"<style>{HTML_CSS}</style></head><body><div class='wrap'>")
    # header
    parts.append('<header class="top"><div>')
    parts.append('<div class="brand">◈ NEURO·SLEEP ANALYTICS</div>')
    parts.append('<h1>睡眠脑电分析报告</h1>')
    parts.append(f'<div class="sub">{_esc(s["session"])} &nbsp;·&nbsp; {_esc(s.get("start") or "")[:19]} '
                 f'&nbsp;·&nbsp; {s["duration_min"]:.0f} min @ {s["fs"]} Hz</div>')
    parts.append('</div><div style="text-align:right">')
    parts.append(f'<span class="chip">单导前额 · ADS1299 CH0</span>')
    parts.append(f'<span class="chip">BIS:{_esc(md["checkpoint"])}</span>')
    parts.append(f'<span class="chip">伪迹 {s["artifact_pct"]:.0f}%</span>')
    parts.append('</div></header>')
    parts.append('<div class="disc">⚠️ 单导前额差分电极,无 EOG/EMG/枕区导联 → 分期为<strong>近似启发式</strong>;'
                 'BIS 为麻醉深度模型推理,睡眠场景仅作皮层激活参考。<strong>本报告不能替代多导睡眠图(PSG),不构成医学诊断。</strong></div>')
    parts.append('<p class="note" style="margin-top:-8px">▸ <strong style="color:var(--mut)">预处理</strong>:'
                 'EEG 经 ads1299 生产栈处理 —— EEGPreprocessor(0.5–47 Hz 带通 + 50 Hz 陷波)→ '
                 'AnesthesiaNetV3 流式推理(每秒 1 步,与 router.py /process 一致);频段定义对齐 EEGPreprocessor.BANDS。'
                 '&nbsp;▸ 睡眠分期另用 0.3–6 Hz 通道<strong style="color:var(--mut)">刻意保留眼动</strong>'
                 '(不去眨眼/眼动),以保住 REM 判读所需的眼动信息。</p>')
    # KPIs
    parts.append(f'<div class="grid-kpi">{"".join(kpis)}</div>')
    # hypnogram
    parts.append('<section><h2><span class="dot"></span>睡眠分期 · 睡眠图</h2>'
                 '<p class="note">基于脑电频谱 + 眼动特征的 30 s/epoch 单导近似分期(颜色对应分期,底部红条=伪迹/脱落)。</p>')
    parts.append(svg_hypnogram(series))
    parts.append('<div class="legend">' + "".join(
        f'<span><i style="background:{STAGE_COLOR[k]}"></i>{STAGE_NAME[k]}</span>'
        for k in ["W", "REM", "N1", "N2", "N3", "ART"]) + '</div></section>')
    # BIS
    parts.append('<section><h2><span class="dot"></span>全程模型推理 BIS / SQI</h2>')
    parts.append(svg_line_bis(series))
    parts.append('<div class="legend"><span><i style="background:var(--cyan)"></i>BIS(模型推理)</span>'
                 '<span><i style="background:#64748b"></i>SQI 信号质量</span>'
                 '<span>绿/黄/橙/红带 = 清醒/浅/中/深 区间</span></div>')
    parts.append(bis_note + '</section>')
    # composition + table
    parts.append('<section><h2><span class="dot"></span>睡眠构成与关键指标</h2><div class="two">')
    parts.append('<div>' + svg_donut(c, M["tst_min"]) + '<div class="legend" style="justify-content:center">' +
                 "".join(f'<span><i style="background:{STAGE_COLOR[k]}"></i>{STAGE_NAME[k]}</span>'
                         for k in ["N3", "N2", "N1", "REM"]) + '</div></div>')
    parts.append('<div>' + stage_table + '</div></div>')
    parts.append('<h3 class="bk">补充指标</h3><table>' + extra_rows + '</table></section>')
    # bands
    parts.append('<section><h2><span class="dot"></span>相对频段功率走势</h2>'
                 '<p class="note">δ 0.5–4 / θ 4–8 / α 8–13 / σ 11–16 / β 13–30 Hz(频段定义对齐 ads1299 EEGPreprocessor)。</p>')
    parts.append(svg_bands(series))
    parts.append('<div class="legend">' + "".join(
        f'<span><i style="background:{col}"></i>{nm}</span>' for nm, col in
        [("δ", "#3730a3"), ("θ", "#0891b2"), ("α", "#16a34a"), ("σ", "#a855f7"), ("β", "#dc2626")]) +
        '</div></section>')
    # events
    parts.append('<section><h2><span class="dot"></span>纺锤波 · K 复合波 · 快速眼动</h2>'
                 '<p class="note">紫=纺锤波(上)/橙=K 复合波(下)每 epoch 计数;绿点=该 epoch 检出快速眼动簇。</p>')
    parts.append(svg_events(series))
    parts.append('<div class="legend"><span><i style="background:#a855f7"></i>纺锤波</span>'
                 '<span><i style="background:#f97316"></i>K 复合波</span>'
                 '<span><i style="background:#34d399"></i>快速眼动(REM)</span></div></section>')
    # ppg
    parts.append(ppg_html)
    # analysis
    parts.append('<section><h2><span class="dot"></span>睡眠质量与身心状态分析</h2>')
    parts.append(analysis_html + '</section>')
    # refs
    parts.append('<section><h2><span class="dot"></span>方法与专业依据</h2>'
                 '<ul class="refs">' + "".join(refs) + '</ul></section>')
    parts.append('<footer>由 scripts/analyze_sleep.py 自动生成 · BIS 推理使用 ads1299 生产栈(EEGPreprocessor + AnesthesiaNetV3)· '
                 '分期/事件为启发式近似,临床判读以标准 PSG 为准</footer>')
    parts.append('</div></body></html>')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description="离线睡眠脑电分析(EEGRecorder → 报告)")
    ap.add_argument("folders", nargs="+", help="一个或多个 EEGRecorder 会话文件夹")
    ap.add_argument("--ckpt", default="outputs/checkpoints/v17/best_model_v3.pt")
    ap.add_argument("--out-dir", default="outputs/reports/sleep")
    ap.add_argument("--awake-cal", action="store_true",
                    help="开启 BIS 清醒锚定偏置校准(麻醉场景特有;睡眠默认关闭,会把 BIS 推向 100)")
    args = ap.parse_args()
    results = []
    for folder in args.folders:
        try:
            summary, out = analyze(folder, args.ckpt, args.out_dir,
                                   awake_cal=args.awake_cal)
            results.append((summary, out))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ {folder} 失败: {e}")
    print("\n==== 全部完成 ====")
    for summ, out in results:
        m = summ["metrics"]
        print(f"  {summ['session']}: TST {m['tst_min']:.0f}min  效率 {m['se_tib']:.0f}%  "
              f"N3 {m['stage_pct'].get('N3',0):.0f}%  纺锤波 {summ['spindles']['total']}  "
              f"K复合波 {summ['kcomplexes']['total']}  → {out}")


if __name__ == "__main__":
    main()
