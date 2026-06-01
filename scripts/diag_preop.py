"""
diag_preop.py — 诊断 pre_op MAE=21 到底是「模型真坍缩」还是「标签/归一化伪迹」。

复用 evaluate.main_v3 的患者级划分与 EEG-only 推理路径，但按相位拆开
pred / label / sqi 的分布，重点回答：
  1. pre_op 真值 BIS 分布是什么？（清醒应 ~90-100）
  2. 模型在 pre_op 预测分布是什么？（坍缩则集中在维持区 ~45-60）
  3. 残差有无系统性符号（欠预测 / 过预测）？
  4. pre_op 的 SQI 如何？（电极刚贴，可能坏窗占多 → 标签不可信）
  5. 误差是否由少数患者主导？
用法：
  python scripts/diag_preop.py --config configs/pipeline_v14.yaml \
      --data outputs/preprocessed/dataset_v3_v13.h5
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import _filter_cases_by_std
from src.data.dataset_v3 import SequenceDatasetV3
from src.models.anesthesia_net_v3 import AnesthesiaNetV3

PHASES = {0: "pre_op", 1: "induction", 2: "maintenance", 3: "recovery"}


def _causal_rolling_mean(arr: np.ndarray, window: int = 15) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    idx = np.arange(len(arr))
    start = np.maximum(0, idx - window + 1)
    return (cs[idx + 1] - cs[start]) / (idx - start + 1)


def _pct(a: np.ndarray) -> str:
    if a.size == 0:
        return "(empty)"
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return (f"mean={a.mean():6.2f}  p5={q[0]:5.1f}  p25={q[1]:5.1f}  "
            f"p50={q[2]:5.1f}  p75={q[3]:5.1f}  p95={q[4]:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline_v14.yaml")
    ap.add_argument("--data",   default="outputs/preprocessed/dataset_v3_v13.h5")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = tcfg["seq_len"]

    ckpt_path = args.checkpoint or str(
        Path(cfg["paths"]["checkpoints"]) / "best_model_v3.pt")

    # ── 复现 build_datasets_v3 的患者级划分 ────────────────────────────────
    with h5py.File(args.data, "r") as f:
        all_cases = sorted(f.keys())
    all_cases = _filter_cases_by_std(
        args.data, all_cases,
        tcfg.get("case_std_pct_low", 10.0), tcfg.get("case_std_pct_high", 90.0))
    rng = random.Random(tcfg["seed"])
    rng.shuffle(all_cases)
    n = len(all_cases)
    n_test = max(1, int(n * tcfg["test_split"]))
    n_val = max(1, int(n * tcfg["val_split"]))
    if args.split == "test":
        ids = all_cases[:n_test]
    else:
        ids = all_cases[n_test:n_test + n_val]
    print(f"Checkpoint : {ckpt_path}")
    print(f"Split      : {args.split}  ({len(ids)} patients)")

    ds = SequenceDatasetV3(args.data, ids, seq_len=seq_len, seq_stride=seq_len,
                           augment=False, cache_in_memory=True, min_seq_std=0.0)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))

    model = AnesthesiaNetV3.from_config(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded epoch {ckpt.get('epoch','?')}  "
          f"val_MAE={ckpt.get('val_mae', float('nan')):.2f}\n")

    # ── 推理收集 ───────────────────────────────────────────────────────────
    P, L, PH, SQ = [], [], [], []   # pred_bis, label_bis, phase, sqi_mean (all per-timestep)
    # 每序列 pre_op 误差（用于查是否少数患者主导）
    seq_preop_mae = []
    with torch.no_grad():
        for batch in loader:
            wave = batch["wave"].to(device)
            feats = batch["features"].to(device)
            sqi = batch["sqi"].to(device)
            out = model(wave, feats, sqi)
            pred = out["pred_bis"].cpu().float().numpy()        # (B,T,1)
            lab = batch["label_seq"].cpu().float().numpy()      # (B,T)
            ph = batch["phases"].cpu().numpy() if "phases" in batch else None
            sq = batch["sqi"].cpu().float().numpy().mean(-1)    # (B,T)
            for b in range(pred.shape[0]):
                sm = _causal_rolling_mean(pred[b, :, 0], 15) * 100.0
                lb = lab[b] * 100.0
                P.append(sm); L.append(lb); SQ.append(sq[b])
                if ph is not None:
                    PH.append(ph[b])
                    m = ph[b] == 0
                    if m.sum() > 0:
                        seq_preop_mae.append(np.abs(sm[m] - lb[m]).mean())

    P = np.concatenate(P); L = np.concatenate(L); SQ = np.concatenate(SQ)
    PH = np.concatenate(PH) if PH else None
    err = np.abs(P - L)
    resid = P - L   # 正=过预测，负=欠预测

    print("="*72)
    print(f"  整体 MAE = {err.mean():.2f}   n={len(err):,}")
    print("="*72)
    if PH is None:
        print("无 phase 标签，无法分相位诊断。")
        return

    print(f"\n{'phase':<13}{'n':>9}{'frac':>7}  {'MAE':>6}  {'resid(P-L)':>11}  "
          f"{'SQI>0.5%':>8}")
    for pid, pname in PHASES.items():
        m = PH == pid
        if m.sum() == 0:
            continue
        sqi_ok_pct = 100.0 * (SQ[m] > 0.5).mean()
        print(f"{pname:<13}{m.sum():>9,}{100*m.mean():>6.1f}%  "
              f"{err[m].mean():>6.2f}  {resid[m].mean():>+11.2f}  {sqi_ok_pct:>7.1f}%")

    # ── pre_op 深挖 ─────────────────────────────────────────────────────────
    m = PH == 0
    print("\n" + "─"*72)
    print("pre_op 深挖")
    print("─"*72)
    print(f"  真值 BIS    : {_pct(L[m])}")
    print(f"  预测 BIS    : {_pct(P[m])}")
    print(f"  SQI         : {_pct(SQ[m])}")
    # 只看 SQI 好的 pre_op 时步（标签可信）
    mg = m & (SQ > 0.5)
    print(f"\n  [仅 SQI>0.5 的 pre_op]  n={mg.sum():,}  ({100*mg.sum()/max(m.sum(),1):.1f}% of pre_op)")
    if mg.sum() > 0:
        print(f"    真值 BIS  : {_pct(L[mg])}")
        print(f"    预测 BIS  : {_pct(P[mg])}")
        print(f"    MAE       : {err[mg].mean():.2f}   resid(P-L) {resid[mg].mean():+.2f}")
    # 真值高 BIS (>80, 清醒) 的 pre_op：模型能否输出高值？
    mh = m & (L > 80)
    print(f"\n  [pre_op 且真值 BIS>80 (清醒)]  n={mh.sum():,}  "
          f"({100*mh.sum()/max(m.sum(),1):.1f}% of pre_op)")
    if mh.sum() > 0:
        print(f"    预测 BIS  : {_pct(P[mh])}")
        print(f"    MAE       : {err[mh].mean():.2f}   resid(P-L) {resid[mh].mean():+.2f}")
        print(f"    预测 max  : {P[mh].max():.1f}  (模型输出上限探测)")

    # 患者级分布
    spm = np.array(seq_preop_mae)
    if spm.size:
        print(f"\n  序列级 pre_op MAE 分布 (n={spm.size} 序列): {_pct(spm)}")
        print(f"    MAE>15 的序列占比: {100*(spm>15).mean():.1f}%")

    # 全局：模型预测的整体上限（是否根本学不会输出高 BIS）
    print("\n" + "─"*72)
    print(f"  全 split 预测 BIS p99 = {np.percentile(P,99):.1f}  max = {P.max():.1f}")
    print(f"  全 split 真值 BIS p99 = {np.percentile(L,99):.1f}  max = {L.max():.1f}")
    print("─"*72)


if __name__ == "__main__":
    main()
