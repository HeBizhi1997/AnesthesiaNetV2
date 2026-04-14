"""
pk_model.py — Pharmacokinetic/Pharmacodynamic Feature Extraction

提取麻醉药物的效应室浓度（Effect-site Concentration, CE）并计算等效浓度 CE_eq，
作为跨模态蒸馏的教师信号（见 MERIDIAN_v9_theory.md §2.2）。

核心输出（1Hz，与 BIS 时间轴对齐）：
  CE_prop   : 丙泊酚效应室浓度 (μg/mL)
  CE_rftn   : 瑞芬太尼效应室浓度 (ng/mL)
  MAC       : 最低肺泡浓度（挥发性麻醉剂深度指标）
  CE_eq     : 等效综合浓度（统一度量，病例内归一化）
  CE_lagged : CE_eq 的 90s 前移版本（蒸馏目标：EEG→预测未来 PK 状态）
  ce_vel    : |dCE_eq/dt|，用于过渡期样本加权

药理参数来源：
  - 丙泊酚：Schnider 1998, Anesthesiology（ke0=0.456 min⁻¹）
  - 瑞芬太尼：Minto 1997 interaction model（β=0.30, γ50=2.0 ng/mL）
  - 挥发性等效：BIS-MAC 关系（Avidan 2008，MAC 1.3 ≈ BIS 40）

设计决策（见理论文档 §9.1）：
  Hill 方程个体差异 MAE=13，不可用作绝对预测基准。
  CE 特征作为蒸馏信号，利用其"方向性"而非绝对水平。
  病例内 Z-score 归一化进一步抵消个体 PK/PD 差异。
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import numpy as np


# ── 药理常数 ──────────────────────────────────────────────────────────────────
# 丙泊酚协同系数（瑞芬太尼影响丙泊酚 CE50）
BETA_RFTN        = 0.30    # 最大协同系数（文献值 0.28-0.33）
GAMMA50_RFTN     = 2.0     # 瑞芬太尼半饱和效应浓度 (ng/mL)
# 挥发性麻醉剂等效（MAC 1.3 → BIS ~40，等效 CE_prop=CE50=3.4 μg/mL）
MAC_EQUIV_COEFF  = 3.4 / 1.3   # μg/mL per MAC unit
# CE 速度计算窗口（60s）
VELOCITY_WIN_SEC = 60
# 蒸馏滞后时间（90s，对应 ke0 平衡时间）
DISTILL_LAG_SEC  = 90
# 维持期识别阈值（CE_eq 处于稳定阶段的判定）
MAINT_CE_STABLE_WIN = 300   # 5分钟稳定窗口
MAINT_VEL_THRESH    = 0.005  # μg/mL/s 以下认为稳定


def _safe_fill_nan(arr: np.ndarray) -> np.ndarray:
    """将 NaN 用前向+后向填充处理（用于连续信号插值）。"""
    out = arr.copy()
    # 前向填充
    mask = np.isnan(out)
    idx = np.where(~mask)[0]
    if len(idx) == 0:
        return out
    out[mask] = np.interp(np.where(mask)[0], idx, out[idx])
    return out


def _case_zscore(arr: np.ndarray,
                 mask: np.ndarray,
                 stable_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    病例内 Z-score 归一化（理论文档 §3.3：抵抗个体 PK/PD 差异）。

    归一化基准：维持期（CE 变化缓慢）的均值和标准差。
    若无维持期数据，退化为全局统计。

    Parameters
    ----------
    arr         : 原始信号 (N,)
    mask        : 有效数据掩码 (N,)
    stable_mask : 维持期掩码 (N,)，None 则用全段

    Returns
    -------
    z : 归一化信号，无效处为 0.0（安全填充）
    """
    ref_mask = mask.copy()
    if stable_mask is not None:
        ref_mask &= stable_mask

    valid = arr[ref_mask]
    if len(valid) < 30:
        valid = arr[mask]   # 回退到全局

    if len(valid) < 10:
        return np.zeros_like(arr)

    mu  = float(np.nanmean(valid))
    std = float(np.nanstd(valid)) + 1e-6
    z = np.where(mask, (arr - mu) / std, 0.0)
    return z.astype(np.float32)


def extract_pk_features(
    vf,          # vitaldb.VitalFile 实例
    n_sec: int,  # 输出数组长度（对齐 BIS 时间轴）
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 VitalFile 中提取 PK/PD 特征。

    Parameters
    ----------
    vf      : vitaldb.VitalFile 实例
    n_sec   : 目标时间轴长度（秒）
    verbose : 是否打印调试信息

    Returns
    -------
    features : (n_sec, 6) float32
        列索引：
          0 - CE_prop_raw   丙泊酚 CE 原始值 (μg/mL，NaN 若无数据)
          1 - CE_rftn_raw   瑞芬太尼 CE 原始值 (ng/mL，NaN 若无数据)
          2 - MAC_raw       MAC 原始值（无量纲）
          3 - CE_eq_norm    等效 CE（病例内归一化）
          4 - CE_eq_lagged  CE_eq_norm 的 90s 前移（蒸馏目标）
          5 - ce_velocity   |dCE_eq/dt|（归一化，过渡期权重）

    mask_drug : (n_sec,) float32
        1.0 = 有任意药物数据可用，0.0 = 全部缺失
    """
    tracks = list(vf.trks.keys())
    out = np.full((n_sec, 6), np.nan, dtype=np.float32)

    # 生理范围上限（超出则视为数据错误，置 NaN）
    CE_PROP_MAX = 20.0   # 丙泊酚 CE 生理上限 20 μg/mL（临床极量）
    CE_RFTN_MAX = 30.0   # 瑞芬太尼 CE 生理上限 30 ng/mL
    MAC_MAX     =  3.0   # MAC 生理上限 3.0（超过此值为数据错误）

    # ── 1. 读取原始 PK 信号 ──────────────────────────────────────────────────
    # 丙泊酚效应室（TCI 泵数据）
    CE_prop = np.full(n_sec, np.nan)
    if "Orchestra/PPF20_CE" in tracks:
        d = vf.to_numpy(["Orchestra/PPF20_CE"], 1.0)
        if d is not None:
            raw = d[:n_sec, 0].astype(np.float64)
            raw[raw > CE_PROP_MAX] = np.nan   # 生理截断
            CE_prop[:len(raw)] = raw
            if verbose:
                valid_pct = (~np.isnan(CE_prop)).mean() * 100
                print(f"  [PK] PPF20_CE: {valid_pct:.0f}% valid, "
                      f"max={np.nanmax(CE_prop):.2f} μg/mL")

    # 瑞芬太尼效应室
    CE_rftn = np.full(n_sec, np.nan)
    for rftn_track in ["Orchestra/RFTN20_CE", "Orchestra/RFTN50_CE"]:
        if rftn_track in tracks:
            d = vf.to_numpy([rftn_track], 1.0)
            if d is not None:
                raw = d[:n_sec, 0].astype(np.float64)
                raw[raw > CE_RFTN_MAX] = np.nan
                CE_rftn[:len(raw)] = np.where(np.isnan(CE_rftn[:len(raw)]),
                                               raw, CE_rftn[:len(raw)])
            break

    # MAC（挥发性麻醉剂）
    MAC = np.full(n_sec, np.nan)
    if "Primus/MAC" in tracks:
        d = vf.to_numpy(["Primus/MAC"], 1.0)
        if d is not None:
            raw = d[:n_sec, 0].astype(np.float64)
            raw[raw > MAC_MAX] = np.nan
            MAC[:len(raw)] = raw

    # ── 2. 计算 CE_eq（等效综合浓度）────────────────────────────────────────
    # 有效性掩码
    has_prop = ~np.isnan(CE_prop)
    has_rftn = ~np.isnan(CE_rftn)
    has_mac  = ~np.isnan(MAC)
    mask_drug = (has_prop | has_rftn | has_mac).astype(np.float32)

    if mask_drug.sum() < 100:
        if verbose:
            print("  [PK] Insufficient drug data (<100s), returning zeros")
        return np.zeros((n_sec, 6), dtype=np.float32), mask_drug

    # 初始化 CE_eq（从可用来源组合）
    CE_eq = np.zeros(n_sec, dtype=np.float64)

    # 丙泊酚贡献
    prop_contrib = np.where(has_prop, CE_prop, 0.0)
    CE_eq += prop_contrib

    # 瑞芬太尼：通过降低丙泊酚 CE50 实现协同（简化线性模型）
    # 等效贡献 = β × CE_rftn / (γ50 + CE_rftn) × CE50_prop
    rftn_contrib = np.where(
        has_rftn,
        BETA_RFTN * CE_rftn / (GAMMA50_RFTN + CE_rftn + 1e-9) * 3.4,
        0.0
    )
    CE_eq += rftn_contrib

    # 挥发性贡献（MAC 转换为等效 μg/mL）
    mac_contrib = np.where(has_mac, MAC * MAC_EQUIV_COEFF, 0.0)
    CE_eq += mac_contrib

    CE_eq = CE_eq.astype(np.float32)

    # 设置 CE_eq 在无药物区段为 NaN（不参与归一化统计）
    CE_eq_nan = np.where(mask_drug > 0.5, CE_eq, np.nan)

    # ── 3. 识别维持期（CE_eq 缓慢变化区）────────────────────────────────────
    vel_raw = np.full(n_sec, np.nan)
    vel_raw[VELOCITY_WIN_SEC:] = np.abs(
        CE_eq_nan[VELOCITY_WIN_SEC:] - CE_eq_nan[:-VELOCITY_WIN_SEC]
    ) / VELOCITY_WIN_SEC

    stable_mask = (mask_drug > 0.5) & (
        np.nan_to_num(vel_raw, nan=999) < MAINT_VEL_THRESH
    )

    # ── 4. 病例内 Z-score 归一化 ──────────────────────────────────────────────
    mask_bool = mask_drug > 0.5
    CE_eq_norm = _case_zscore(CE_eq_nan, mask_bool, stable_mask)
    # 截断到合理范围（防止极端值，±5σ 以内）
    CE_eq_norm = np.clip(CE_eq_norm, -5.0, 5.0)

    # ── 5. 蒸馏滞后版本（CE_eq_norm 前移 90s）────────────────────────────────
    # CE_lagged(t) = CE_eq_norm(t + LAG)
    # 含义：EEG(t) 应能预测 LAG 秒后的 PK 状态（前瞻性药效推断）
    CE_lagged = np.full(n_sec, 0.0, dtype=np.float32)
    lag = DISTILL_LAG_SEC
    CE_lagged[:n_sec - lag] = CE_eq_norm[lag:]
    CE_lagged[n_sec - lag:] = 0.0   # 尾部用 0 填充（缺少未来信息）

    # ── 6. CE 速度（|dCE_eq/dt|，过渡期加权因子）────────────────────────────
    vel_for_weight = np.zeros(n_sec, dtype=np.float32)
    valid_vel = ~np.isnan(vel_raw)
    if valid_vel.sum() > 100:
        vel_mu  = float(np.nanmean(vel_raw[valid_vel]))
        vel_std = float(np.nanstd(vel_raw[valid_vel])) + 1e-8
        vel_norm = np.where(valid_vel, (vel_raw - vel_mu) / vel_std, 0.0)
        vel_for_weight = np.clip(vel_norm, 0.0, 5.0).astype(np.float32)

    # ── 7. 组装输出 ───────────────────────────────────────────────────────────
    out[:, 0] = np.where(has_prop, CE_prop, 0.0).astype(np.float32)
    out[:, 1] = np.where(has_rftn, CE_rftn, 0.0).astype(np.float32)
    out[:, 2] = np.where(has_mac,  MAC,     0.0).astype(np.float32)
    out[:, 3] = CE_eq_norm
    out[:, 4] = CE_lagged
    out[:, 5] = vel_for_weight

    if verbose:
        print(f"  [PK] CE_eq: mean={np.nanmean(CE_eq_nan):.2f} μg/mL  "
              f"std={np.nanstd(CE_eq_nan):.2f}  "
              f"transition (vel>thresh): {stable_mask.sum()} stable / "
              f"{mask_bool.sum()} total drug seconds")
        print(f"  [PK] ce_velocity p90={np.percentile(vel_for_weight[mask_bool], 90):.2f} "
              f"(normalized)")

    return out, mask_drug


def map_1hz_to_windows(
    feat_1hz: np.ndarray,    # (n_sec, F) 1Hz 特征
    mask_1hz: np.ndarray,    # (n_sec,) 1Hz 掩码
    times:    np.ndarray,    # (N_win,) int，每个 window 对应的 1Hz 时刻
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 1Hz 特征数组按 window 时刻索引映射到 window-level。

    Parameters
    ----------
    feat_1hz : (n_sec, F) PK 特征（1Hz）
    mask_1hz : (n_sec,) 数据可用性掩码（1Hz）
    times    : (N_win,) 每个 window 的 EEG 结束秒

    Returns
    -------
    feat_win : (N_win, F) window-level PK 特征
    mask_win : (N_win,)   window-level 掩码
    """
    n_sec = feat_1hz.shape[0]
    t_arr   = times.astype(int)
    valid   = (t_arr >= 0) & (t_arr < n_sec)
    t_safe  = np.clip(t_arr, 0, n_sec - 1)
    feat_win = feat_1hz[t_safe].astype(np.float32)   # (N_win, F)
    mask_win = mask_1hz[t_safe].astype(np.float32)   # (N_win,)
    feat_win[~valid] = 0.0
    mask_win[~valid] = 0.0
    return feat_win, mask_win


def augment_hdf5_with_pk(
    h5_path: str,
    raw_data_dir: str,
    verbose: bool = True,
) -> None:
    """
    为 HDF5 数据集中每个 case 添加 PK/PD 特征。

    新增数据集：
      drug_ce   : (N_win, 6)  PK 特征（CE_prop/rftn/mac/eq_norm/lagged/velocity）
      mask_drug : (N_win,)    药物数据可用性
    """
    import os
    import h5py
    import vitaldb
    from tqdm import tqdm

    with h5py.File(h5_path, "a") as f:
        case_ids = list(f.keys())
        n_done = 0
        n_with_drug = 0

        for cid in tqdm(case_ids, desc="[2/3] PK/PD", disable=not verbose):
            grp = f[cid]

            if "drug_ce" in grp and "mask_drug" in grp:
                if (~np.isnan(grp["drug_ce"][:, 3])).any():
                    n_with_drug += 1
                continue

            vital_path = os.path.join(raw_data_dir, f"{cid}.vital")
            if not os.path.exists(vital_path):
                continue

            try:
                vf = vitaldb.VitalFile(vital_path)
            except Exception as e:
                tqdm.write(f"  [WARN] Cannot open {vital_path}: {e}")
                continue

            if "times" not in grp:
                continue

            times = grp["times"][:]
            n_sec = int(times.max()) + DISTILL_LAG_SEC + 10

            feat_1hz, mask_1hz = extract_pk_features(vf, n_sec, verbose=False)
            feat_win, mask_win = map_1hz_to_windows(feat_1hz, mask_1hz, times)

            grp.create_dataset("drug_ce",  data=feat_win, compression="gzip")
            grp.create_dataset("mask_drug", data=mask_win, compression="gzip")

            if mask_win.any():
                n_with_drug += 1
            n_done += 1

        if verbose:
            print(f"[PK] Processed {n_done} new cases  "
                  f"drug data available: {n_with_drug} total cases")


# ── 诊断工具 ──────────────────────────────────────────────────────────────────

def validate_pk_single(vital_path: str) -> dict:
    """验证单个文件的 PK 特征提取质量。"""
    import vitaldb

    vf = vitaldb.VitalFile(vital_path)
    bis_data = vf.to_numpy(["BIS/BIS"], 1.0)
    if bis_data is None:
        return {"error": "No BIS"}
    n_sec = len(bis_data)

    feat, mask = extract_pk_features(vf, n_sec, verbose=False)

    has_drug = mask.mean() > 0.1
    CE_prop_max = float(np.nanmax(feat[:, 0])) if has_drug else 0.0
    ce_vel_p90  = float(np.percentile(feat[:, 5][mask > 0.5], 90)) if has_drug else 0.0

    return {
        "n_sec": n_sec,
        "has_drug": has_drug,
        "drug_coverage_pct": round(float(mask.mean() * 100), 1),
        "CE_prop_max": round(CE_prop_max, 2),
        "CE_eq_norm_std": round(float(np.std(feat[mask > 0.5, 3])) if has_drug else 0.0, 3),
        "ce_velocity_p90": round(ce_vel_p90, 3),
    }
