"""
upgrade_features_v11.py — 从现有 dataset_v3.h5 升级特征到 v11 (feature_dim=28)

v10 → v11 特征变化：
  旧 (24 维, v10): [ch0(11) | ch1(11) | asym | sqi]
    ch*(11) = [δ θ α β γ | PE | SEF95 | LZC | BSR×3]

  新 (28 维, v11): [ch0(13) | ch1(13) | asym | sqi]
    ch*(13) = [δ θ α β γ | PE | SEF95 | LZC | BSR×3 | spectral_slope | gamma_emg_ratio]

策略：
  - 读取已存储的 waves (N, 2, 512)，仅重新计算 spectral_slope + gamma_emg_ratio
  - 保留所有其他数据（labels, sqi, drug_ce, vitals, stim_cv 等）
  - 写入新 HDF5 dataset_v3_v11.h5

速度：无需重读 30GB raw .vital，只需 FFT 运算
  估算：2000 case × 10000 windows / 16 workers ≈ 5-10 分钟

Usage:
    python scripts/upgrade_features_v11.py
    python scripts/upgrade_features_v11.py --src outputs/preprocessed/dataset_v3.h5
                                            --dst outputs/preprocessed/dataset_v3_v11.h5
                                            --workers 8
                                            --batch 4096
"""

from __future__ import annotations
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
from tqdm import tqdm

from src.data.batch_processor import _batch_welch, _batch_spectral_slope, _batch_gamma_emg_ratio


# ── 常量 ─────────────────────────────────────────────────────────────────────

OLD_FEATS_PER_CH = 11   # v10: δθαβγ + PE + SEF95 + LZC + BSR×3
NEW_FEATS_PER_CH = 13   # v11: + spectral_slope + gamma_emg_ratio
N_CH             = 2
OLD_TOTAL        = OLD_FEATS_PER_CH * N_CH + 2   # 24
NEW_TOTAL        = NEW_FEATS_PER_CH * N_CH + 2   # 28
FS               = 128.0


# ── 特征升级（单块）─────────────────────────────────────────────────────────

def _upgrade_features_block(
    waves_block: np.ndarray,    # (M, 2, 512) float32
    feats_block: np.ndarray,    # (M, 24) float32
) -> np.ndarray:
    """
    给一个 window 块的 features 追加 spectral_slope + gamma_emg_ratio。
    返回 (M, 28) float32。
    """
    M = waves_block.shape[0]
    new_feats = np.zeros((M, NEW_TOTAL), dtype=np.float32)

    # 对每个通道计算新特征
    new_features_per_ch = []   # list of (M,) arrays for [slope, gamma_ratio]
    for ch in range(N_CH):
        flat = waves_block[:, ch, :].astype(np.float64)   # (M, T)
        freqs, pxx = _batch_welch(flat, FS, nperseg=min(256, flat.shape[1]))
        slope = _batch_spectral_slope(pxx, freqs)          # (M,)
        gamma = _batch_gamma_emg_ratio(pxx, freqs)         # (M,)
        new_features_per_ch.append((slope, gamma))

    # 拼装新特征向量：
    #   [ch0_old(11) | ch0_slope | ch0_gamma | ch1_old(11) | ch1_slope | ch1_gamma | asym | sqi]
    col = 0
    for ch in range(N_CH):
        old_start = ch * OLD_FEATS_PER_CH
        old_end   = old_start + OLD_FEATS_PER_CH
        # 复制原 11 个特征
        new_feats[:, col: col + OLD_FEATS_PER_CH] = feats_block[:, old_start:old_end]
        col += OLD_FEATS_PER_CH
        # 追加 spectral_slope + gamma_emg_ratio
        slope, gamma = new_features_per_ch[ch]
        new_feats[:, col]     = slope
        new_feats[:, col + 1] = gamma
        col += 2

    # 末尾两维 (alpha_asym + mean_sqi) 保持不变
    inter_start = OLD_FEATS_PER_CH * N_CH   # 22
    new_feats[:, col: col + 2] = feats_block[:, inter_start: inter_start + 2]

    return new_feats


def _upgrade_case(args):
    """
    子进程工作函数：升级单个 case 并写入临时 HDF5。
    返回 (case_id, n_windows, tmp_h5_path_or_error)。
    """
    case_id, src_path, tmp_dir, batch_size = args
    tmp_h5 = str(Path(tmp_dir) / f"{case_id}.h5")
    try:
        with h5py.File(src_path, "r") as src:
            grp = src[case_id]
            waves = grp["waves"][:]       # (N, 2, 512)
            feats = grp["features"][:]    # (N, 24)

            # 分块计算新特征（避免峰值内存过高）
            N = waves.shape[0]
            new_feats = np.empty((N, NEW_TOTAL), dtype=np.float32)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                new_feats[start:end] = _upgrade_features_block(
                    waves[start:end], feats[start:end])

            # 写临时 HDF5：复制所有数据集，用新 features 替换
            with h5py.File(tmp_h5, "w") as dst:
                grp_dst = dst.create_group(case_id)
                # 复制所有非 features 数据集
                for key in grp.keys():
                    if key == "features":
                        continue
                    src_ds = grp[key]
                    grp_dst.create_dataset(key, data=src_ds[:],
                                           compression="gzip", compression_opts=4)
                # 写升级后的 features
                grp_dst.create_dataset("features", data=new_feats,
                                       compression="gzip", compression_opts=4)
                # 复制 attributes
                for attr_key, attr_val in grp.attrs.items():
                    grp_dst.attrs[attr_key] = attr_val
                # 更新 feature_dim attribute
                grp_dst.attrs["feature_dim"] = NEW_TOTAL

        return case_id, N, tmp_h5
    except Exception as e:
        if Path(tmp_h5).exists():
            Path(tmp_h5).unlink(missing_ok=True)
        return case_id, 0, f"ERROR: {e}"


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",     default="outputs/preprocessed/dataset_v3.h5")
    parser.add_argument("--dst",     default="outputs/preprocessed/dataset_v3_v11.h5")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行 worker 数（默认: min(16, cpu-2)）")
    parser.add_argument("--batch",   type=int, default=4096,
                        help="每 worker 内分块处理的 window 数")
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        print(f"ERROR: 源文件不存在: {src_path}")
        sys.exit(1)

    print(f"源 HDF5: {src_path}")
    print(f"目标 HDF5: {dst_path}")
    print(f"特征升级: {OLD_TOTAL}维 → {NEW_TOTAL}维 (+ spectral_slope + gamma_emg_ratio)")

    import os
    cpu = os.cpu_count() or 4
    n_workers = args.workers if args.workers else min(16, max(1, cpu - 2))
    print(f"并行 workers: {n_workers}")

    # 读取 case 列表
    with h5py.File(str(src_path), "r") as f:
        case_ids = sorted(f.keys())
    print(f"共 {len(case_ids)} 个 case")

    # 跳过已完成的 case（dst 已有的）
    done_cases: set = set()
    if dst_path.exists():
        with h5py.File(str(dst_path), "r") as f:
            done_cases = set(f.keys())
        if done_cases:
            print(f"跳过已完成: {len(done_cases)} 个 case（使用 --overwrite 强制重算）")
    todo = [c for c in case_ids if c not in done_cases]
    print(f"待处理: {len(todo)} 个 case")

    if not todo:
        print("全部完成。")
        return

    # 临时目录
    tmp_dir = dst_path.parent / "tmp_upgrade_v11"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (cid, str(src_path), str(tmp_dir), args.batch)
        for cid in todo
    ]

    # ── 并行处理 ──────────────────────────────────────────────────────────────
    t0 = time.time()
    ok_results  = []
    err_results = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_upgrade_case, wa): wa[0] for wa in worker_args}
        with tqdm(total=len(futures), desc="升级特征", unit="case") as pbar:
            for fut in as_completed(futures):
                cid, n, path_or_err = fut.result()
                if isinstance(path_or_err, str) and path_or_err.startswith("ERROR:"):
                    tqdm.write(f"  FAIL {cid}: {path_or_err}")
                    err_results.append((cid, n, path_or_err))
                else:
                    ok_results.append((cid, n, path_or_err))
                pbar.update(1)

    elapsed = time.time() - t0
    print(f"\n处理完成: {len(ok_results)} ok, {len(err_results)} 失败  "
          f"({elapsed:.1f}s / {elapsed/60:.1f} min)")

    # ── 合并到目标 HDF5 ───────────────────────────────────────────────────────
    print("合并到目标 HDF5 ...")
    t1 = time.time()
    total_windows = 0

    with h5py.File(str(dst_path), "a") as dst:
        for cid, n, tmp_h5 in tqdm(ok_results, desc="合并", unit="case"):
            if cid in dst:
                del dst[cid]
            with h5py.File(tmp_h5, "r") as src:
                src.copy(cid, dst)
            total_windows += n
            Path(tmp_h5).unlink(missing_ok=True)

        # 顺带统计已有 case 的 windows
        for cid in done_cases:
            if cid in dst:
                total_windows += int(dst[cid].attrs.get("n_windows", 0))

    # 清理临时目录
    import shutil
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    merge_elapsed = time.time() - t1
    print(f"合并耗时: {merge_elapsed:.1f}s")
    print(f"\n升级完成！")
    print(f"  目标文件: {dst_path}")
    print(f"  总 windows: {total_windows:,}")
    print(f"  feature_dim: {OLD_TOTAL} → {NEW_TOTAL}")
    print(f"  总耗时: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

    # ── 快速验证 ──────────────────────────────────────────────────────────────
    print("\n验证前 3 个 case ...")
    with h5py.File(str(dst_path), "r") as f:
        for cid in sorted(f.keys())[:3]:
            feats = f[cid]["features"]
            assert feats.shape[1] == NEW_TOTAL, \
                f"Case {cid}: feature_dim={feats.shape[1]} != {NEW_TOTAL}"
            # 检查 spectral_slope 范围 [0, 1]
            slope_ch0 = feats[:, OLD_FEATS_PER_CH]
            slope_ch1 = feats[:, OLD_FEATS_PER_CH + 2 + OLD_FEATS_PER_CH]
            assert slope_ch0.min() >= 0.0 and slope_ch0.max() <= 1.0, \
                f"Case {cid} ch0 slope out of [0,1]"
            print(f"  {cid}: shape={feats.shape} ✓  "
                  f"slope_ch0∈[{slope_ch0.min():.2f},{slope_ch0.max():.2f}]  "
                  f"slope_ch1∈[{slope_ch1.min():.2f},{slope_ch1.max():.2f}]")
    print("验证通过 ✓")


if __name__ == "__main__":
    main()
