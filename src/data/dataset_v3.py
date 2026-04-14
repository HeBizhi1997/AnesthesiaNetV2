"""
dataset_v3.py — 多模态序列数据集（MERIDIAN-v9）

在 SequenceDataset 基础上扩展，增加：
  - drug_ce    : (T, 6)  PK/PD 特征（CE_prop/rftn/mac/eq_norm/lagged/velocity）
  - vitals     : (T, 5)  生命体征快照（HR/SpO2/MBP/ETCO2/BT）
  - stim_cv    : (T,)    心血管刺激标签（新，替代旧 stim_events）
  - mask_drug  : (T,)    药物数据可用性
  - mask_vital : (T,)    生命体征可用性
  - ce_velocity: (T,)    过渡期加权因子（= drug_ce[:, 5]）

推理时（EEG-only）：drug_ce / vitals / mask_drug / mask_vital 全部置零，
模型 forward 接受 training_mode=False 时忽略这些输入。

HDF5 格式（向后兼容）：
  原有字段：waves, features, sqi, labels, times, phases, stim_events
  新增字段：drug_ce, vitals, stim_cv, mask_drug, mask_vital

build_dataset_v3() 是主要入口，负责：
  1. 调用 stim_labeler 和 pk_model 为 HDF5 文件增加多模态标签
  2. 提取生命体征并写入 HDF5
  3. 返回 SequenceDatasetV3 实例
"""

from __future__ import annotations
import os
import random
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import SequenceDataset, _filter_cases_by_std


# ── 生命体征提取 ───────────────────────────────────────────────────────────────
# 优先级：有创 > 无创，缺失用 NaN 填充

VITAL_TRACKS = {
    # (主要轨道, 备用轨道, 归一化均值, 归一化标准差)
    "HR":    ("Solar8000/HR",          None,                   75.0,  15.0),
    "SpO2":  ("Solar8000/PLETH_SPO2",  None,                   98.0,   2.0),
    "MBP":   ("Solar8000/ART_MBP",    "Solar8000/NIBP_MBP",   80.0,  15.0),
    "ETCO2": ("Solar8000/ETCO2",       None,                   35.0,   5.0),
    "BT":    ("Solar8000/BT",          None,                   36.5,   0.5),
}
VITAL_NAMES = ["HR", "SpO2", "MBP", "ETCO2", "BT"]   # 顺序固定，对应 vitals 列


def _extract_vitals_1hz(vf, n_sec: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取 5 个生命体征的 1Hz 快照，返回归一化后的特征矩阵。

    Returns
    -------
    vitals   : (n_sec, 5) float32，已全局归一化（μ/σ 基于典型生理范围）
    mask_vit : (n_sec,) float32，1=至少有一个生命体征可用
    """
    tracks = list(vf.trks.keys())
    vitals = np.full((n_sec, 5), 0.0, dtype=np.float32)   # 缺失填 0（全局归一化后的均值）
    avail  = np.zeros(n_sec, dtype=bool)

    for vi, (name, (prim, backup, mu, sigma)) in enumerate(VITAL_TRACKS.items()):
        arr = np.full(n_sec, np.nan)

        # 主要轨道
        if prim in tracks:
            d = vf.to_numpy([prim], 1.0)
            if d is not None:
                n = min(n_sec, len(d))
                arr[:n] = d[:n, 0]

        # 备用轨道（仅填充 NaN 位置）
        if backup and backup in tracks:
            d2 = vf.to_numpy([backup], 1.0)
            if d2 is not None:
                n2 = min(n_sec, len(d2))
                fill_idx = np.isnan(arr[:n2]) & ~np.isnan(d2[:n2, 0])
                arr[:n2][fill_idx] = d2[:n2, 0][fill_idx]

        # 全局归一化（使用典型生理范围的 μ/σ）
        valid = ~np.isnan(arr)
        if valid.any():
            arr_norm = (arr - mu) / sigma
            arr_norm = np.clip(arr_norm, -5.0, 5.0)
            vitals[:, vi] = np.where(valid, arr_norm, 0.0)
            avail |= valid

    mask_vit = avail.astype(np.float32)
    return vitals, mask_vit


def augment_hdf5_with_vitals(
    h5_path: str,
    raw_data_dir: str,
    verbose: bool = True,
) -> None:
    """
    为 HDF5 数据集中每个 case 添加生命体征特征。

    新增数据集：
      vitals     : (N_win, 5)  归一化后的生命体征快照
      mask_vital : (N_win,)    是否有生命体征可用（已由 stim_labeler 写入时合并）
    """
    import vitaldb
    from tqdm import tqdm

    with h5py.File(h5_path, "a") as f:
        case_ids = list(f.keys())
        n_done = 0

        for cid in tqdm(case_ids, desc="[3/3] Vitals", disable=not verbose):
            grp = f[cid]
            if "vitals" in grp:
                continue

            vital_path = os.path.join(raw_data_dir, f"{cid}.vital")
            if not os.path.exists(vital_path):
                continue

            try:
                vf = vitaldb.VitalFile(vital_path)
            except Exception:
                continue

            if "times" not in grp:
                continue

            times = grp["times"][:]
            n_sec = int(times.max()) + 10

            vit_1hz, _ = _extract_vitals_1hz(vf, n_sec)

            # 向量化 1Hz → window 索引映射
            t_arr  = times.astype(int)
            valid  = (t_arr >= 0) & (t_arr < n_sec)
            t_safe = np.clip(t_arr, 0, n_sec - 1)
            vit_win = vit_1hz[t_safe].astype(np.float32)   # (N_win, 5)
            vit_win[~valid] = 0.0

            grp.create_dataset("vitals", data=vit_win, compression="gzip")
            n_done += 1

        if verbose:
            print(f"[Vitals] Added vitals to {n_done} cases")


def build_multimodal_hdf5(
    h5_path: str,
    raw_data_dir: str,
    verbose: bool = True,
) -> None:
    """
    一次性为 HDF5 添加所有多模态标签（PK + Stim + Vitals）。
    幂等：已存在的数据集不会重复添加。
    """
    from .stim_labeler import augment_hdf5_with_stim_cv
    from .pk_model import augment_hdf5_with_pk

    if verbose:
        print("=== Building multimodal HDF5 annotations ===")

    if verbose:
        print("\n[1/3] Cardiovascular stimulation labels...")
    augment_hdf5_with_stim_cv(h5_path, raw_data_dir, verbose=verbose)

    if verbose:
        print("\n[2/3] PK/PD features...")
    augment_hdf5_with_pk(h5_path, raw_data_dir, verbose=verbose)

    if verbose:
        print("\n[3/3] Vital signs features...")
    augment_hdf5_with_vitals(h5_path, raw_data_dir, verbose=verbose)

    if verbose:
        print("\n=== Multimodal annotation complete ===")


# ── 主 Dataset 类 ─────────────────────────────────────────────────────────────

class SequenceDatasetV3(Dataset):
    """
    多模态序列数据集（MERIDIAN-v9）。

    在 SequenceDataset 基础上增加药物/生命体征输入，训练时使用；
    推理时调用者只需提供 EEG 字段，药物/生命体征字段全部为零张量。

    新增返回字段：
      "drug_ce"    : (T, 6)  PK 特征
      "vitals"     : (T, 5)  生命体征
      "stim_cv"    : (T,)    CV 刺激标签
      "mask_drug"  : (T,)    药物可用性
      "mask_vital" : (T,)    生命体征可用性
      "ce_velocity": (T,)    过渡期加权因子
    """

    def __init__(
        self,
        h5_path: str,
        case_ids: List[str],
        seq_len: int = 300,
        seq_stride: Optional[int] = None,
        augment: bool = False,
        noise_std: float = 0.05,
        cache_in_memory: bool = True,
        min_seq_std: float = 0.0,
    ):
        self.seq_len    = seq_len
        self.seq_stride = seq_stride if seq_stride is not None else 1
        self.augment    = augment
        self.noise_std  = noise_std
        self.cache_in_memory = cache_in_memory

        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._index: List[Tuple[str, int]] = []
        self._seq_bis: List[float] = []

        with h5py.File(h5_path, "r") as f:
            for cid in case_ids:
                if cid not in f:
                    continue
                grp = f[cid]
                n = int(grp.attrs.get("n_windows", 0))
                if n < seq_len:
                    continue

                if cache_in_memory:
                    entry = {
                        "waves":    grp["waves"][:].astype(np.float32),
                        "features": grp["features"][:].astype(np.float32),
                        "sqi":      grp["sqi"][:].astype(np.float32),
                        "labels":   grp["labels"][:].astype(np.float32),
                    }
                    # 原有多任务标签
                    if "phases" in grp:
                        entry["phases"]      = grp["phases"][:].astype(np.int64)
                    if "stim_events" in grp:
                        entry["stim_events"] = grp["stim_events"][:].astype(np.float32)
                    # v3 新增：CV 刺激标签
                    if "stim_cv" in grp:
                        entry["stim_cv"]     = grp["stim_cv"][:].astype(np.float32)
                    else:
                        entry["stim_cv"]     = np.zeros(n, dtype=np.float32)
                    # v3 新增：PK 特征
                    if "drug_ce" in grp:
                        entry["drug_ce"]     = grp["drug_ce"][:].astype(np.float32)
                        entry["mask_drug"]   = grp["mask_drug"][:].astype(np.float32)
                    else:
                        entry["drug_ce"]     = np.zeros((n, 6), dtype=np.float32)
                        entry["mask_drug"]   = np.zeros(n, dtype=np.float32)
                    # v3 新增：生命体征
                    if "vitals" in grp:
                        entry["vitals"]      = grp["vitals"][:].astype(np.float32)
                    else:
                        entry["vitals"]      = np.zeros((n, 5), dtype=np.float32)
                    # v3 新增：生命体征可用性（来自 stim_labeler）
                    if "mask_vital" in grp:
                        entry["mask_vital"]  = grp["mask_vital"][:].astype(np.float32)
                    else:
                        entry["mask_vital"]  = np.zeros(n, dtype=np.float32)

                    self._cache[cid] = entry

                labels_arr = (self._cache[cid]["labels"]
                              if cache_in_memory else f[cid]["labels"][:])

                # 构建有效序列起点索引
                if "times" in grp:
                    times = grp["times"][:]
                    diffs = np.diff(times)
                    for i in range(0, n - seq_len + 1, self.seq_stride):
                        if not np.all(diffs[i: i + seq_len - 1] == 1):
                            continue
                        seg = labels_arr[i: i + seq_len]
                        if min_seq_std > 0 and seg.std() < min_seq_std:
                            continue
                        self._index.append((cid, i))
                        self._seq_bis.append(float(seg[-1]))
                else:
                    for i in range(0, n - seq_len + 1, self.seq_stride):
                        seg = labels_arr[i: i + seq_len]
                        if min_seq_std > 0 and seg.std() < min_seq_std:
                            continue
                        self._index.append((cid, i))
                        self._seq_bis.append(float(seg[-1]))

        self._h5: Optional[h5py.File] = None
        self.is_cached = cache_in_memory

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cid, start = self._index[idx]
        end = start + self.seq_len
        c   = self._cache[cid]

        waves  = c["waves"][start:end].copy()
        feats  = c["features"][start:end].copy()
        sqis   = c["sqi"][start:end].copy()
        labels = c["labels"][start:end].copy()
        phases = c["phases"][start:end].copy()    if "phases"      in c else np.zeros(self.seq_len, dtype=np.int64)
        stim_cv = c["stim_cv"][start:end].copy()
        drug_ce  = c["drug_ce"][start:end].copy()
        vitals   = c["vitals"][start:end].copy()
        mask_drug  = c["mask_drug"][start:end].copy()
        mask_vital = c["mask_vital"][start:end].copy()

        if self.augment:
            noise = np.random.randn(*waves.shape).astype(np.float32) * self.noise_std
            scale = np.float32(np.random.uniform(0.85, 1.15))
            waves = waves * scale + noise
            if waves.shape[1] == 2 and np.random.rand() < 0.5:
                waves = waves[:, [1, 0], :]

        last_label = float(labels[-1])
        return {
            # EEG 核心（训练+推理都需要）
            "wave":        torch.from_numpy(waves),
            "features":    torch.from_numpy(feats),
            "sqi":         torch.from_numpy(sqis),
            "label":       torch.tensor(last_label / 100.0, dtype=torch.float32),
            "label_raw":   torch.tensor(last_label, dtype=torch.float32),
            "label_seq":   torch.from_numpy(labels / 100.0),
            # 分类标签
            "phases":      torch.from_numpy(phases),
            "stim_cv":     torch.from_numpy(stim_cv),
            # 多模态（仅训练）
            "drug_ce":     torch.from_numpy(drug_ce),
            "vitals":      torch.from_numpy(vitals),
            "mask_drug":   torch.from_numpy(mask_drug),
            "mask_vital":  torch.from_numpy(mask_vital),
            # 过渡期权重因子（drug_ce[:, 5]）
            "ce_velocity": torch.from_numpy(drug_ce[:, 5]),
        }

    def get_sample_weights(self, n_bins: int = 10) -> np.ndarray:
        """逆频率加权（基于 BIS 分布），用于 WeightedRandomSampler。"""
        bis = np.array(self._seq_bis, dtype=np.float32)
        bin_edges = np.linspace(0, 100, n_bins + 1)
        bin_idx = np.digitize(bis, bin_edges[1:-1])
        counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float32)
        counts = np.maximum(counts, 1)
        bin_weights = 1.0 / counts
        return bin_weights[bin_idx]

    def boost_transition_sequences(self, factor: int) -> None:
        """
        对包含高 CE velocity（PK 过渡期）的序列进行额外采样。
        替代旧的 induction_boost（固定相位复制），更精细地针对动态变化时期。
        """
        if factor <= 0 or not self.cache_in_memory:
            return
        if getattr(self, "_transition_boost_applied", False):
            return
        self._transition_boost_applied = True

        extra_idx, extra_bis = [], []
        for cid, start in self._index:
            c = self._cache[cid]
            seg_vel = c["drug_ce"][start: start + self.seq_len, 5]
            if float(seg_vel.max()) > 1.0:   # 高 CE velocity（归一化 > 1σ）
                bis_val = float(c["labels"][start + self.seq_len - 1])
                extra_idx.extend([(cid, start)] * factor)
                extra_bis.extend([bis_val] * factor)

        if extra_idx:
            self._index   = self._index   + extra_idx
            self._seq_bis = self._seq_bis + extra_bis
            print(f"Transition boost x{factor}: added {len(extra_idx):,} sequences "
                  f"(total {len(self._index):,})")
        else:
            print("Transition boost: no high-velocity sequences found "
                  "(drug data may be missing)")

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# ── 工厂函数 ─────────────────────────────────────────────────────────────────

def build_datasets_v3(
    h5_path: str,
    raw_data_dir: str,
    val_split: float    = 0.15,
    test_split: float   = 0.10,
    seq_len: int        = 300,
    seq_stride: Optional[int] = 150,
    seed: int           = 42,
    noise_std: float    = 0.05,
    cache_in_memory: bool = True,
    min_seq_std: float  = 0.0,
    case_std_pct_low: float  = 10.0,
    case_std_pct_high: float = 90.0,
    transition_boost: int    = 0,
    rebuild_multimodal: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    构建多模态训练/验证/测试数据集（患者级别划分，无数据泄漏）。

    Parameters
    ----------
    rebuild_multimodal : 强制重新计算多模态标签（否则跳过已存在的数据集）
    transition_boost   : 对高 CE velocity 序列的额外采样倍数（替代 induction_boost）
    """
    # 先确保 HDF5 中有多模态标签
    if rebuild_multimodal or not _check_multimodal_ready(h5_path):
        build_multimodal_hdf5(h5_path, raw_data_dir, verbose=True)

    # 案例质量过滤 + 患者级别划分
    with h5py.File(h5_path, "r") as f:
        all_cases = sorted(f.keys())

    all_cases = _filter_cases_by_std(
        h5_path, all_cases, case_std_pct_low, case_std_pct_high
    )

    rng = random.Random(seed)
    rng.shuffle(all_cases)
    n = len(all_cases)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    test_ids  = all_cases[:n_test]
    val_ids   = all_cases[n_test: n_test + n_val]
    train_ids = all_cases[n_test + n_val:]

    print(f"Patient split — train: {len(train_ids)}, "
          f"val: {len(val_ids)}, test: {len(test_ids)}")

    if cache_in_memory:
        print("Loading multimodal dataset into RAM…")

    kw_train = dict(seq_len=seq_len, seq_stride=seq_stride, augment=True,
                    noise_std=noise_std, cache_in_memory=cache_in_memory,
                    min_seq_std=min_seq_std)
    kw_eval  = dict(seq_len=seq_len, seq_stride=seq_stride, augment=False,
                    cache_in_memory=cache_in_memory, min_seq_std=0.0)

    train_ds = SequenceDatasetV3(h5_path, train_ids, **kw_train)
    val_ds   = SequenceDatasetV3(h5_path, val_ids,   **kw_eval)
    test_ds  = SequenceDatasetV3(h5_path, test_ids,  **kw_eval)

    if transition_boost > 0:
        train_ds.boost_transition_sequences(transition_boost)

    print(f"  train: {len(train_ds):,}, val: {len(val_ds):,}, test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


def _check_multimodal_ready(h5_path: str) -> bool:
    """检查 HDF5 是否已经包含多模态标签（至少第一个 case 有 drug_ce 和 stim_cv）。"""
    try:
        with h5py.File(h5_path, "r") as f:
            cids = list(f.keys())
            if not cids:
                return False
            grp = f[cids[0]]
            return "drug_ce" in grp and "stim_cv" in grp and "vitals" in grp
    except Exception:
        return False
