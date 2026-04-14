"""
stim_labeler.py — Cardiovascular Surgical Stimulation Label Generator

原理（参见 MERIDIAN_v9_theory.md §2.3）：
  手术刺激 → 伤害性信号 → 交感激活 → HR↑ + BP↑（5-30s 内）
  → EEG 觉醒（15-30s 延迟）

  用 CV 响应定义刺激标签，比 EEG BIS 上升更早、更可靠、更具因果性。

伪迹防护（三层）：
  1. 生理速率门控：|ΔHR/Δt| > 5 bpm/s → 心率伪迹，mask
  2. NIBP 缺口检测：NIBP 突变 > 40 mmHg 或测量期空白 → 仅使用 HR
  3. 联合条件（AND）：HR 和 BP 必须同时响应，而非 OR

临床文献依据：
  - Cohen 1987, Br J Anaesth: ΔHR>15%, ΔSBP>20mmHg 定义血流动力学应激
  - Iselin-Chaves 2006, Anesthesiology: 非甾体 NSAID + 芬太尼对 CV 反应的抑制研究

输出格式：1 Hz 二值数组，与 BIS/labels 时间轴对齐。
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np


# ── 阈值常数 ──────────────────────────────────────────────────────────────────
HR_DELTA_THRESH      = 0.15   # 心率增幅阈值 (15% of baseline)
SBP_DELTA_THRESH     = 20.0   # 收缩压增幅阈值 (mmHg)
HR_VELOCITY_MAX      = 5.0    # 生理极限：HR 变化速率上限 (bpm/s)
NIBP_JUMP_THRESH     = 40.0   # NIBP 突变检测阈值 (mmHg)，超过则视为袖带充气伪迹
BASELINE_PRE_SEC     = 180    # 基线窗口：当前时刻前 180s（理论 §2.3.2）
BASELINE_SKIP_SEC    = 30     # 基线窗口末端留白（排除刺激本身的影响）
# 实际基线段：arr[t-180 : t-30]，即 150 个数据点（Cohen 1987）
MIN_BASELINE_POINTS  = 30     # 基线至少需要有效点数
SMOOTH_WINDOW_SEC    = 90     # 标签平滑窗口（秒），rolling mean
SMOOTH_THRESH        = 0.5    # 平滑后 > 此阈值 → 标注为 1
MIN_POSITIVE_SEC     = 45     # 连续响应最短持续时间（秒）
VASOACTIVE_DRUGS     = [      # 血管活性药物：使用期间 BP↑ 不代表刺激
    "Orchestra/PHEN_RATE",    # 苯肾上腺素（vasopressor）
    "Orchestra/NEPI_RATE",    # 去甲肾上腺素（vasopressor）
    "Orchestra/DOPA_RATE",    # 多巴胺（inotrope/vasopressor）
    "Orchestra/NTG_RATE",     # 硝酸甘油（vasodilator，BP↓ 药，反向影响但排除干扰）
]


# ── 内部工具函数 ──────────────────────────────────────────────────────────────

def _causal_median(arr: np.ndarray,
                   window: int,
                   skip: int = 0) -> np.ndarray:
    """
    因果滑动中位数（无未来信息泄漏）。
    output[t] = median(arr[t-window : t-skip])
    skip > 0 排除紧邻当前时刻的点（防止响应本身污染基线）。

    向量化实现（pandas rolling），比 Python 循环快 ~100x。
    数学等价性：
      shift(skip+1) 使 s[t] = arr[t-skip-1]；
      rolling(window-skip) 得到 median(arr[t-window : t-skip]) ✓
    """
    import pandas as pd
    effective_w = window - skip          # 实际窗口宽度（如 180-30=150）
    s = pd.Series(arr).shift(skip + 1)  # s[t] = arr[t - skip - 1]
    result = s.rolling(window=effective_w, min_periods=MIN_BASELINE_POINTS).median()
    out = result.to_numpy(dtype=np.float64, na_value=np.nan)
    out[:window + skip] = np.nan        # 与原始循环起点保持一致
    return out


def _artifact_mask_hr(hr: np.ndarray) -> np.ndarray:
    """
    生理速率门控：检测 HR 伪迹（心率不可能在 1 秒内突变 >5 bpm）。
    返回 bool 数组：True = 此点 HR 可信。
    """
    mask = np.ones(len(hr), dtype=bool)
    velocity = np.abs(np.diff(hr, prepend=hr[0]))   # |Δhr/Δt| at 1Hz
    mask[velocity > HR_VELOCITY_MAX] = False
    # 还原 NaN 为不可信
    mask[np.isnan(hr)] = False
    return mask


def _artifact_mask_nibp(sbp: np.ndarray) -> np.ndarray:
    """
    NIBP 袖带充气伪迹检测：突变 > NIBP_JUMP_THRESH 或 NaN 区域。
    返回 bool 数组：True = 此点 BP 可信。

    NIBP 以低频间歇读数为主（每 1-5 分钟一次），其余时刻为 NaN，
    由 to_numpy() 插值填充。突变点即为袖带充气/放气边界。
    """
    mask = np.ones(len(sbp), dtype=bool)
    mask[np.isnan(sbp)] = False
    jump = np.abs(np.diff(sbp, prepend=sbp[0]))
    mask[jump > NIBP_JUMP_THRESH] = False
    return mask


def _rolling_mean_causal(arr: np.ndarray, window: int) -> np.ndarray:
    """因果滑动均值（无未来信息泄漏，向量化实现）。"""
    N = len(arr)
    out = np.full(N, np.nan, dtype=np.float64)
    if N <= window:
        return out
    cs  = np.nancumsum(arr)                       # NaN 按 0 累加
    cnt = np.cumsum(~np.isnan(arr)).astype(float) # 非 NaN 计数
    n = cnt[window:] - cnt[:N - window]            # 各窗口有效点数
    s = cs[window:]  - cs[:N - window]             # 各窗口有效点之和
    out[window:] = np.where(n > 0, s / n, np.nan)
    return out


# ── 主要函数：从 vitaldb 数据生成 CV 刺激标签 ────────────────────────────────

# HR-only 模式的更严格阈值（无连续 BP 时使用，减少假阳性）
HR_DELTA_THRESH_STRICT = 0.20  # 20%（比标准的 15% 严格）

def compute_stim_cv_labels(
    vf,          # vitaldb.VitalFile 实例
    n_sec: int,  # 目标时间轴长度（秒），与 BIS 时间轴对齐
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 VitalFile 中提取心血管应激响应标签。

    策略（根据 BP 可用性自适应）：
      - 有有创 ART_SBP（连续）：HR AND SBP 联合条件（严格，假阳性低）
      - 仅有 NIBP（间歇）：HR-only 模式（更严格阈值 20%，单信号）
      - 无 BP：HR-only 模式（20% 阈值）

    Parameters
    ----------
    vf      : vitaldb.VitalFile 实例（已读取）
    n_sec   : 输出数组长度（应与 BIS 标签数组长度一致）
    verbose : 是否打印调试信息

    Returns
    -------
    stim_label : (n_sec,) float32 — 0.0 / 1.0，CV 刺激标签
    mask_vital : (n_sec,) float32 — 1.0 = CV 数据可用，0.0 = 缺失
    """
    tracks = list(vf.trks.keys())

    # ── 1. 提取 HR（必需信号，100% 可用）───────────────────────────────────
    hr_data = vf.to_numpy(["Solar8000/HR"], 1.0)
    if hr_data is None or np.isnan(hr_data).all():
        if verbose:
            print("  [StimLabeler] No HR data, returning zero labels")
        return (np.zeros(n_sec, dtype=np.float32),
                np.zeros(n_sec, dtype=np.float32))

    hr = hr_data[:n_sec, 0].astype(np.float64)
    hr_mask = _artifact_mask_hr(hr)

    # ── 2. 提取有创 ART_SBP（连续 BP，仅 ~58% 案例）────────────────────────
    art_sbp = None
    has_continuous_bp = False

    if "Solar8000/ART_SBP" in tracks:
        art_data = vf.to_numpy(["Solar8000/ART_SBP"], 1.0)
        if art_data is not None:
            raw = art_data[:n_sec, 0].astype(np.float64)
            valid_frac = (~np.isnan(raw)).mean()
            if valid_frac > 0.3:   # 至少 30% 有效才使用
                art_sbp = raw
                has_continuous_bp = True
                if verbose:
                    print(f"  [StimLabeler] ART mode: {valid_frac*100:.0f}% ART_SBP coverage")

    # ── 3. 血管活性药物掩码 ────────────────────────────────────────────────
    vasoactive = np.zeros(n_sec, dtype=bool)
    for drug_track in VASOACTIVE_DRUGS:
        if drug_track in tracks:
            drug_data = vf.to_numpy([drug_track], 1.0)
            if drug_data is not None:
                rate = drug_data[:n_sec, 0]
                vasoactive |= (~np.isnan(rate) & (rate > 0))

    # ── 4. 因果基线计算 ───────────────────────────────────────────────────
    hr_clean = hr.copy()
    hr_clean[~hr_mask] = np.nan
    hr_base = _causal_median(hr_clean, BASELINE_PRE_SEC, BASELINE_SKIP_SEC)

    # ── 5. 刺激检测（根据 BP 可用性选择模式）────────────────────────────────
    if has_continuous_bp:
        # ── 模式 A：HR AND ART_SBP 联合条件 ─────────────────────────────
        sbp_mask_arr = ~np.isnan(art_sbp)
        sbp_clean = art_sbp.copy()
        sbp_clean[~sbp_mask_arr] = np.nan

        sbp_base = _causal_median(sbp_clean, BASELINE_PRE_SEC, BASELINE_SKIP_SEC)

        delta_hr  = (hr_clean - hr_base)  / (hr_base  + 1e-6)
        delta_sbp = sbp_clean - sbp_base

        hr_ok  = (delta_hr  > HR_DELTA_THRESH)   & hr_mask
        sbp_ok = (delta_sbp > SBP_DELTA_THRESH)  & sbp_mask_arr
        sbp_ok[vasoactive] = False

        raw_stim = (hr_ok & sbp_ok).astype(np.float64)
        raw_stim[np.isnan(delta_hr) | np.isnan(delta_sbp)] = np.nan

        mask_vital = (~np.isnan(hr_base) & ~np.isnan(sbp_base)).astype(np.float32)

    else:
        # ── 模式 B：HR-only（无连续 BP）更严格阈值 ──────────────────────
        delta_hr = (hr_clean - hr_base) / (hr_base + 1e-6)

        hr_ok = (delta_hr > HR_DELTA_THRESH_STRICT) & hr_mask
        hr_ok[vasoactive] = False   # 血管活性药物可能间接影响 HR

        raw_stim = hr_ok.astype(np.float64)
        raw_stim[np.isnan(delta_hr)] = np.nan

        mask_vital = (~np.isnan(hr_base)).astype(np.float32)

        if verbose:
            print(f"  [StimLabeler] HR-only mode (no continuous ART_SBP)")

    # ── 6. 时间平滑（因果 rolling mean）─────────────────────────────────────
    smooth_stim = _rolling_mean_causal(raw_stim, SMOOTH_WINDOW_SEC)
    stim_binary = (smooth_stim > SMOOTH_THRESH).astype(np.float32)
    stim_binary[np.isnan(smooth_stim)] = 0.0

    # ── 7. 最短持续时间过滤 ──────────────────────────────────────────────────
    stim_filtered = _filter_short_events(stim_binary, MIN_POSITIVE_SEC)

    if verbose:
        valid = mask_vital > 0.5
        pos_pct = stim_filtered[valid].mean() * 100 if valid.any() else 0
        mode = "ART+HR" if has_continuous_bp else "HR-only"
        print(f"  [StimLabeler] [{mode}] Positive rate: {pos_pct:.1f}%  "
              f"vasoactive: {vasoactive.sum()}s")

    return stim_filtered[:n_sec].astype(np.float32), mask_vital[:n_sec]


def _filter_short_events(stim: np.ndarray, min_dur: int) -> np.ndarray:
    """
    清除持续时间 < min_dur 的孤立阳性段（连通分量过滤）。
    """
    result = stim.copy()
    N = len(stim)
    i = 0
    while i < N:
        if stim[i] > 0.5:
            j = i
            while j < N and stim[j] > 0.5:
                j += 1
            duration = j - i
            if duration < min_dur:
                result[i:j] = 0.0
            i = j
        else:
            i += 1
    return result


# ── 批量处理：为 HDF5 数据集增加 CV 刺激标签 ─────────────────────────────────

def augment_hdf5_with_stim_cv(
    h5_path: str,
    raw_data_dir: str,
    verbose: bool = True,
) -> None:
    """
    为 HDF5 数据集中的每个 case 添加 CV 刺激标签（stim_cv）和可用性掩码（mask_vital）。

    HDF5 中的标签以"窗口索引"存储（window-level），但 CV 标签是"1 秒级"的。
    此函数将 1Hz 标签按 window 的 `times` 数组映射到 window 索引。

    Parameters
    ----------
    h5_path      : 现有的 HDF5 数据集路径
    raw_data_dir : .vital 文件所在目录
    verbose      : 是否打印进度
    """
    import os
    import h5py
    import vitaldb
    from tqdm import tqdm

    with h5py.File(h5_path, "a") as f:
        case_ids = list(f.keys())
        n_done = 0
        pos_counts = 0
        total_counts = 0

        for cid in tqdm(case_ids, desc="[1/3] StimCV", disable=not verbose):
            grp = f[cid]

            # 跳过已处理的
            if "stim_cv" in grp and "mask_vital" in grp:
                pos_counts  += int(grp["stim_cv"][:].sum())
                total_counts += len(grp["stim_cv"])
                continue

            # 找对应 .vital 文件
            vital_path = os.path.join(raw_data_dir, f"{cid}.vital")
            if not os.path.exists(vital_path):
                continue

            try:
                vf = vitaldb.VitalFile(vital_path)
            except Exception as e:
                tqdm.write(f"  [WARN] Cannot open {vital_path}: {e}")
                continue

            # times 数组：每个 window 对应的 EEG 结束秒
            if "times" not in grp:
                continue

            times = grp["times"][:]          # (N_win,) int，EEG 结束秒
            n_win = len(times)
            n_sec = int(times.max()) + 10   # 足够长的 1Hz 缓冲

            # 生成 1Hz 标签
            stim_1hz, mask_1hz = compute_stim_cv_labels(
                vf, n_sec, verbose=False
            )

            # 向量化 1Hz → window 索引映射
            t_arr   = times.astype(int)
            valid   = (t_arr >= 0) & (t_arr < len(stim_1hz))
            t_safe  = np.clip(t_arr, 0, len(stim_1hz) - 1)
            stim_win = np.where(valid, stim_1hz[t_safe], 0.0).astype(np.float32)
            mask_win = np.where(valid, mask_1hz[t_safe], 0.0).astype(np.float32)

            grp.create_dataset("stim_cv",    data=stim_win, compression="gzip")
            grp.create_dataset("mask_vital", data=mask_win, compression="gzip")

            pos_counts  += int(stim_win.sum())
            total_counts += n_win
            n_done += 1

        if verbose:
            if total_counts > 0:
                pos_pct = pos_counts / total_counts * 100
                print(f"[StimLabeler] Labeled {n_done} cases  "
                      f"positive rate: {pos_pct:.1f}%  "
                      f"({pos_counts:,}/{total_counts:,} windows)")
            else:
                print(f"[StimLabeler] No cases processed (all already labeled)")


# ── 诊断工具：在单个 .vital 文件上验证标签质量 ─────────────────────────────

def validate_single_file(vital_path: str, verbose: bool = True) -> dict:
    """
    在单个 .vital 文件上运行标签生成，返回诊断统计。
    用于验证标签分布是否合理（阳性率 3-6%）。
    """
    import vitaldb

    vf = vitaldb.VitalFile(vital_path)

    # 获取 BIS 时间轴长度
    bis_data = vf.to_numpy(["BIS/BIS"], 1.0)
    if bis_data is None:
        return {"error": "No BIS data"}
    n_sec = len(bis_data)

    stim, mask = compute_stim_cv_labels(vf, n_sec, verbose=verbose)

    valid = mask > 0.5
    if valid.sum() == 0:
        return {"error": "No valid vital data", "n_sec": n_sec}

    pos_rate = stim[valid].mean() * 100
    n_events = int(np.diff(np.concatenate([[0], (stim > 0.5).astype(int), [0]])).clip(0).sum())

    return {
        "n_sec": n_sec,
        "n_valid": int(valid.sum()),
        "pos_rate_pct": round(float(pos_rate), 2),
        "n_stim_events": n_events,
        "mean_event_dur_sec": round(float(stim[valid].sum() / max(n_events, 1)), 1),
        "data_coverage_pct": round(float(valid.mean() * 100), 1),
    }
