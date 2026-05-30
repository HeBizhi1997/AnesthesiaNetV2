"""
evaluate.py — 全面评估训练好的模型（支持 AnesthesiaNet V1 和 V2）。

修复的Bug：
  - 原版硬编码 AnesthesiaNet (V1)，V2 模型直接崩溃（返回5值被3值解包）
  - 原版用 BIS 阈值(60/40)定义相位，实际应用 phase_labels 列标签
  - 原版缺少相位分类准确率、刺激检测敏感度/特异性/AUROC

Usage:
    python scripts/evaluate.py --config configs/pipeline_v6.yaml
    python scripts/evaluate.py --config configs/pipeline_v6.yaml \\
        --checkpoint outputs/checkpoints/v6/best_model_v2.pt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets, _filter_cases_by_std
from src.models.anesthesia_net    import AnesthesiaNet
from src.models.anesthesia_net_v2 import AnesthesiaNetV2
from src.models.anesthesia_net_v3 import AnesthesiaNetV3


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(arr[mask].mean())


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """计算二分类 AUROC (Wilcoxon-Mann-Whitney 统计量)。"""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # 向量化 AUROC
    pos_exp = pos[:, None]     # (P, 1)
    neg_exp = neg[None, :]     # (1, N)
    auc = ((pos_exp > neg_exp).sum() + 0.5 * (pos_exp == neg_exp).sum())
    return float(auc) / (len(pos) * len(neg))


def _threshold_metrics(scores: np.ndarray, labels: np.ndarray,
                        threshold: float = 0.5) -> dict:
    """给定阈值计算敏感度/特异性/精确率/F1。"""
    pred_bin = (scores >= threshold).astype(int)
    tp = ((pred_bin == 1) & (labels == 1)).sum()
    tn = ((pred_bin == 0) & (labels == 0)).sum()
    fp = ((pred_bin == 1) & (labels == 0)).sum()
    fn = ((pred_bin == 0) & (labels == 1)).sum()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision   = tp / max(tp + fp, 1)
    f1          = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision":   float(precision),
        "f1":          float(f1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_v2(model: AnesthesiaNetV2, loader: DataLoader,
                device: torch.device) -> dict:
    """
    V2 模型评估：同时收集 BIS 预测、相位分类、刺激检测的全部指标。

    V2 forward 签名：
        pred_bis, phase_logits, stim_logits, correction, h
        = model(wave, features, sqi)

    使用 dataset 中实际的 phase_labels（不再用BIS阈值划分相位）。
    """
    model.eval()
    all_pred_bis, all_label_bis = [], []
    all_pred_phase, all_true_phase = [], []
    all_pred_stim, all_true_stim = [], []

    with torch.no_grad():
        for batch in loader:
            wave     = batch["wave"].to(device)       # (B, T, n_ch, W)
            features = batch["features"].to(device)   # (B, T, F)
            sqi      = batch["sqi"].to(device)        # (B, T, n_ch)

            # V2 forward: 5 return values
            pred_bis, phase_logits, stim_logits, _corr, _h = model(
                wave, features, sqi)

            B = wave.shape[0]

            # ── BIS (last timestep) ─────────────────────────────────────
            pred_bis_last = pred_bis[:, -1, 0].cpu().float().numpy() * 100.0
            label_bis_arr = batch["label_raw"].cpu().numpy()   # (B,) last window BIS
            all_pred_bis.append(pred_bis_last)
            all_label_bis.append(label_bis_arr)

            # ── Phase classification (last timestep) ────────────────────
            if "phases" in batch:
                ph_pred = phase_logits[:, -1, :].argmax(-1).cpu().numpy()   # (B,)
                ph_true = batch["phases"][:, -1].cpu().numpy()               # (B,)
                all_pred_phase.append(ph_pred)
                all_true_phase.append(ph_true)

            # ── Stimulation detection (all timesteps, flatten) ──────────
            if "stim_events" in batch:
                st_prob = torch.sigmoid(stim_logits[:, :, 0]).cpu().float().numpy()   # (B, T)
                st_true = batch["stim_events"].cpu().float().numpy()                   # (B, T)
                all_pred_stim.append(st_prob.ravel())
                all_true_stim.append(st_true.ravel())

    pred_bis  = np.concatenate(all_pred_bis)
    label_bis = np.concatenate(all_label_bis)
    err       = np.abs(pred_bis - label_bis)

    results: dict = {}

    # ── BIS 整体指标 ─────────────────────────────────────────────────────────
    results["mae_overall"] = float(err.mean())
    results["rmse"]        = float(np.sqrt(((pred_bis - label_bis) ** 2).mean()))
    results["pearson_r"]   = float(np.corrcoef(pred_bis, label_bis)[0, 1])
    results["n_samples"]   = len(pred_bis)

    # ── 按真实相位标签计算 BIS MAE ────────────────────────────────────────────
    # 使用真实相位标签（不是BIS阈值），这才是正确的方式
    if all_pred_phase:
        true_ph = np.concatenate(all_true_phase)
        phase_names = {0: "pre_op", 1: "induction", 2: "maintenance", 3: "recovery"}
        for ph_id, ph_name in phase_names.items():
            mask = (true_ph == ph_id)
            results[f"mae_{ph_name}"] = _safe_mean(err, mask)
            results[f"n_{ph_name}"]   = int(mask.sum())
    else:
        # 降级：用BIS阈值（当phase_labels不可用时）
        results["mae_induction_approx"]   = _safe_mean(err, label_bis >= 60)
        results["mae_maintenance_approx"] = _safe_mean(err, (label_bis >= 40) & (label_bis < 60))
        results["mae_recovery_approx"]    = _safe_mean(err, label_bis < 40)

    # ── 相位分类指标 ──────────────────────────────────────────────────────────
    if all_pred_phase:
        pred_ph = np.concatenate(all_pred_phase)
        true_ph = np.concatenate(all_true_phase)
        results["phase_accuracy"] = float((pred_ph == true_ph).mean())
        # 每类准确率
        phase_names = {0: "pre_op", 1: "induction", 2: "maintenance", 3: "recovery"}
        for ph_id, ph_name in phase_names.items():
            mask = (true_ph == ph_id)
            if mask.sum() > 0:
                results[f"phase_acc_{ph_name}"] = float((pred_ph[mask] == true_ph[mask]).mean())
            else:
                results[f"phase_acc_{ph_name}"] = float("nan")

    # ── 刺激检测指标 ──────────────────────────────────────────────────────────
    if all_pred_stim:
        pred_st = np.concatenate(all_pred_stim)
        true_st = np.concatenate(all_true_stim)
        results["stim_auroc"]    = _auroc(pred_st, true_st)
        results["stim_n_pos"]    = int(true_st.sum())
        results["stim_n_neg"]    = int((true_st == 0).sum())
        results["stim_pos_rate"] = float(true_st.mean())
        thr_m = _threshold_metrics(pred_st, true_st, threshold=0.5)
        results.update({f"stim_{k}": v for k, v in thr_m.items()})

    return results


def evaluate_v1(model: AnesthesiaNet, loader: DataLoader,
                device: torch.device) -> dict:
    """V1 模型评估（兼容）。"""
    model.eval()
    all_pred, all_label = [], []

    with torch.no_grad():
        for batch in loader:
            wave = batch["wave"].to(device)
            feat = batch["features"].to(device)
            sqi  = batch["sqi"].to(device)
            # V1 返回 (pred, pred_seq, h)
            pred, _, _ = model(wave, feat, sqi)
            pred_bis  = pred.squeeze(-1).cpu().numpy() * 100.0
            label_bis = batch["label_raw"].cpu().numpy()
            all_pred.append(pred_bis)
            all_label.append(label_bis)

    pred_arr  = np.concatenate(all_pred)
    label_arr = np.concatenate(all_label)
    err = np.abs(pred_arr - label_arr)
    return {
        "mae_overall": float(err.mean()),
        "rmse":        float(np.sqrt(((pred_arr - label_arr) ** 2).mean())),
        "pearson_r":   float(np.corrcoef(pred_arr, label_arr)[0, 1]),
        "n_samples":   len(pred_arr),
        # BIS阈值相位（V1无phase_labels）
        "mae_induction_approx":   float(_safe_mean(err, label_arr >= 60)),
        "mae_maintenance_approx": float(_safe_mean(err, (label_arr >= 40) & (label_arr < 60))),
        "mae_recovery_approx":    float(_safe_mean(err, label_arr < 40)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# V3 helpers (rank-sum AUROC + causal smoothing — match trainer_v3 val metric)
# ─────────────────────────────────────────────────────────────────────────────

def _auroc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via rank-sum — O(n log n), no P×N matrix (avoids OOM on full test set)."""
    n_pos = int((labels == 1).sum())
    n_neg = int(len(labels)) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    rank_sum_pos = float(ranks[labels == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _causal_rolling_mean(arr: np.ndarray, window: int = 15) -> np.ndarray:
    """15-step causal rolling mean — identical to trainer_v3 val smoothing."""
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    idx = np.arange(len(arr))
    start = np.maximum(0, idx - window + 1)
    return (cs[idx + 1] - cs[start]) / (idx - start + 1)


def evaluate_v3(model: AnesthesiaNetV3, loader: DataLoader,
                device: torch.device) -> dict:
    """
    V3 (MERIDIAN) 评估 — EEG-only 推理路径（drug_ce/vitals=None）。

    全时步 BIS + 每序列 15 步因果平滑（与 trainer_v3.val_epoch 完全一致，
    使 test MAE 与训练上报的 vMAE 可直接对比）。分相位 MAE 用真实 phase 标签。
    """
    model.eval()
    all_pred_bis, all_label_bis, all_phase = [], [], []
    all_pred_phase, all_true_phase = [], []
    all_pred_stim, all_true_stim = [], []

    with torch.no_grad():
        for batch in loader:
            wave     = batch["wave"].to(device)
            features = batch["features"].to(device)
            sqi      = batch["sqi"].to(device)
            out = model(wave, features, sqi)          # dict, EEG-only

            pred_np  = out["pred_bis"].cpu().float().numpy()        # (B, T, 1)
            label_np = batch["label_seq"].cpu().float().numpy()     # (B, T)
            phase_np = (batch["phases"].cpu().numpy()
                        if "phases" in batch else None)

            for b in range(pred_np.shape[0]):
                smoothed = _causal_rolling_mean(pred_np[b, :, 0], 15) * 100.0
                all_pred_bis.append(smoothed)
                all_label_bis.append(label_np[b] * 100.0)
                if phase_np is not None:
                    all_phase.append(phase_np[b])

            if "phases" in batch:
                all_pred_phase.append(out["phase_logits"].argmax(-1).cpu().numpy().ravel())
                all_true_phase.append(batch["phases"].cpu().numpy().ravel())

            stim_key = "stim_cv" if "stim_cv" in batch else (
                "stim_events" if "stim_events" in batch else None)
            if stim_key is not None:
                st_prob = torch.sigmoid(out["stim_logits"][:, :, 0]).cpu().float().numpy()
                all_pred_stim.append(st_prob.ravel())
                all_true_stim.append(batch[stim_key].cpu().float().numpy().ravel())

    pred_bis  = np.concatenate(all_pred_bis)
    label_bis = np.concatenate(all_label_bis)
    err       = np.abs(pred_bis - label_bis)

    results: dict = {
        "mae_overall": float(err.mean()),
        "rmse":        float(np.sqrt(((pred_bis - label_bis) ** 2).mean())),
        "pearson_r":   float(np.corrcoef(pred_bis, label_bis)[0, 1]),
        "n_samples":   len(pred_bis),
    }

    if all_phase:
        phase_arr = np.concatenate(all_phase)
        phase_names = {0: "pre_op", 1: "induction", 2: "maintenance", 3: "recovery"}
        for ph_id, ph_name in phase_names.items():
            mask = (phase_arr == ph_id)
            results[f"mae_{ph_name}"] = _safe_mean(err, mask)
            results[f"n_{ph_name}"]   = int(mask.sum())

    if all_pred_phase:
        pred_ph = np.concatenate(all_pred_phase)
        true_ph = np.concatenate(all_true_phase)
        results["phase_accuracy"] = float((pred_ph == true_ph).mean())
        phase_names = {0: "pre_op", 1: "induction", 2: "maintenance", 3: "recovery"}
        for ph_id, ph_name in phase_names.items():
            mask = (true_ph == ph_id)
            results[f"phase_acc_{ph_name}"] = (
                float((pred_ph[mask] == true_ph[mask]).mean()) if mask.sum() else float("nan"))

    if all_pred_stim:
        pred_st = np.concatenate(all_pred_stim)
        true_st = np.concatenate(all_true_stim)
        results["stim_auroc"]    = _auroc_rank(pred_st, true_st)
        results["stim_n_pos"]    = int(true_st.sum())
        results["stim_n_neg"]    = int((true_st == 0).sum())
        results["stim_pos_rate"] = float(true_st.mean())
        results.update({f"stim_{k}": v
                        for k, v in _threshold_metrics(pred_st, true_st, 0.5).items()})

    return results


def analyze_checkpoint_selection(history: dict, ema_alpha: float = 0.3) -> dict:
    """
    EMA vs 原始 val_mae 选点对比（量化 v13 "4.57 噪声峰" 问题）。

    原始选点：argmin(val_mae) —— 抓住每个真实最优点，但在抖动的验证曲线上
              挑中的往往是噪声峰，高估泛化。
    EMA 选点：argmin(EMA_α(val_mae)) —— 稳定值，更接近真实泛化水平。

    返回两种策略的 (epoch, mae) 及差距；差距越大说明原始策略越虚高。
    """
    vm = history.get("val_mae", []) if history else []
    if not vm:
        return {}
    vm = np.asarray(vm, dtype=np.float64)
    ema = np.empty_like(vm)
    ema[0] = vm[0]
    for i in range(1, len(vm)):
        ema[i] = ema_alpha * vm[i] + (1 - ema_alpha) * ema[i - 1]
    raw_ep = int(np.argmin(vm))
    ema_ep = int(np.argmin(ema))
    return {
        "n_epochs":      len(vm),
        "raw_best_epoch": raw_ep + 1,
        "raw_best_mae":   float(vm[raw_ep]),
        "ema_best_epoch": ema_ep + 1,
        "ema_best_mae":   float(ema[ema_ep]),
        # 原始最优 epoch 处的 EMA 值 = 该点的"诚实"水平
        "raw_best_ema":   float(ema[raw_ep]),
        "overstatement":  float(ema[raw_ep] - vm[raw_ep]),
    }


def print_selection_analysis(sel: dict) -> None:
    if not sel:
        print("\n── 选点分析 ─────────────────────────────────────")
        print("  (checkpoint 无 history，跳过)")
        return
    print(f"\n── 选点分析 (EMA α=0.3 vs 原始 val_mae) ──────────")
    print(f"  训练 epoch 数      : {sel['n_epochs']}")
    print(f"  原始选点 (旧策略)  : ep{sel['raw_best_epoch']}  vMAE={sel['raw_best_mae']:.2f}")
    print(f"  EMA  选点 (诚实)   : ep{sel['ema_best_epoch']}  vMAE={sel['ema_best_mae']:.2f}")
    print(f"  原始最优点的 EMA值 : {sel['raw_best_ema']:.2f}  "
          f"(虚高 {sel['overstatement']:+.2f} BIS)")
    if sel["overstatement"] > 0.3:
        print(f"  [!] 原始 val_mae 选点高估了 ~{sel['overstatement']:.2f} BIS（噪声峰）")
    else:
        print(f"  [OK] 验证曲线稳定，原始与 EMA 选点基本一致")


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def print_results(metrics: dict, model_version: str) -> None:
    print(f"\n{'='*60}")
    print(f"  评估结果 (AnesthesiaNet {model_version.upper()})")
    print(f"{'='*60}")

    print(f"\n── BIS 回归指标 ─────────────────────────────────")
    print(f"  样本数       : {metrics.get('n_samples', '?'):,}")
    print(f"  MAE (整体)   : {metrics['mae_overall']:.2f} BIS points")
    print(f"  RMSE         : {metrics['rmse']:.2f}")
    print(f"  Pearson r    : {metrics['pearson_r']:.4f}")

    print(f"\n── 分相位 BIS MAE ───────────────────────────────")
    if "mae_pre_op" in metrics:
        for ph in ["pre_op", "induction", "maintenance", "recovery"]:
            mae = metrics.get(f"mae_{ph}", float("nan"))
            n   = metrics.get(f"n_{ph}", 0)
            mae_s = f"{mae:.2f}" if mae == mae else "  N/A"
            print(f"  {ph:<15}: {mae_s}  (n={n:,})")
    else:
        # V1 降级
        for key, label in [
            ("mae_induction_approx",   "诱导 (BIS≥60)"),
            ("mae_maintenance_approx", "维持 (40-60)"),
            ("mae_recovery_approx",    "恢复 (BIS<40)"),
        ]:
            v = metrics.get(key, float("nan"))
            s = f"{v:.2f}" if v == v else "  N/A"
            print(f"  {label:<20}: {s}")

    if "phase_accuracy" in metrics:
        print(f"\n── 相位分类指标 ─────────────────────────────────")
        print(f"  整体准确率   : {metrics['phase_accuracy']*100:.1f}%")
        for ph in ["pre_op", "induction", "maintenance", "recovery"]:
            acc = metrics.get(f"phase_acc_{ph}", float("nan"))
            acc_s = f"{acc*100:.1f}%" if acc == acc else "  N/A"
            print(f"  {ph:<15}: {acc_s}")

    if "stim_auroc" in metrics:
        print(f"\n── 刺激检测指标 ─────────────────────────────────")
        print(f"  正例数/负例数: {metrics['stim_n_pos']:,} / {metrics['stim_n_neg']:,}"
              f"  ({metrics['stim_pos_rate']*100:.2f}% 阳性)")
        auc = metrics.get("stim_auroc", float("nan"))
        print(f"  AUROC        : {auc:.4f}" if auc == auc else "  AUROC: N/A")
        print(f"  Sensitivity  : {metrics.get('stim_sensitivity', 0)*100:.1f}%")
        print(f"  Specificity  : {metrics.get('stim_specificity', 0)*100:.1f}%")
        print(f"  Precision    : {metrics.get('stim_precision', 0)*100:.1f}%")
        print(f"  F1           : {metrics.get('stim_f1', 0):.4f}")

    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main_v3(args, cfg) -> dict:
    """V3 (MERIDIAN) 评估：EEG-only test 指标 + EMA vs raw 选点分析。"""
    from src.data.dataset_v3 import SequenceDatasetV3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]
    seq_len = args.seq_len or tcfg["seq_len"]

    # 数据路径：默认用 cfg 的 multimodal_h5
    h5_path = args.data
    if h5_path == "outputs/preprocessed/dataset.h5":
        h5_path = cfg.get("paths", {}).get(
            "multimodal_h5", "outputs/preprocessed/dataset_v3.h5")

    # checkpoint 路径
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_dir = cfg.get("paths", {}).get("checkpoints", "outputs/checkpoints")
        ckpt_path = str(Path(ckpt_dir) / "best_model_v3.pt")
    print(f"Device: {device}  |  Model: AnesthesiaNetV3  |  h5={h5_path}")
    print(f"Checkpoint: {ckpt_path}")

    # ── 复现 build_datasets_v3 的患者级划分，仅构建 test 集（省 RAM）──────────
    import h5py
    with h5py.File(h5_path, "r") as f:
        all_cases = sorted(f.keys())
    all_cases = _filter_cases_by_std(
        h5_path, all_cases,
        tcfg.get("case_std_pct_low", 10.0), tcfg.get("case_std_pct_high", 90.0))
    rng = random.Random(tcfg["seed"])
    rng.shuffle(all_cases)
    n = len(all_cases)
    n_test = max(1, int(n * tcfg["test_split"]))
    test_ids = all_cases[:n_test]
    print(f"Test patients: {len(test_ids)} (held-out, same split as training)")

    test_ds = SequenceDatasetV3(
        h5_path, test_ids, seq_len=seq_len, seq_stride=seq_len,
        augment=False, cache_in_memory=True, min_seq_std=0.0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"Test sequences: {len(test_ds):,}  seq_len={seq_len}")

    model = AnesthesiaNetV3.from_config(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Checkpoint from epoch {ckpt.get('epoch','?')},  "
          f"val_MAE={ckpt.get('val_mae', float('nan')):.2f} BIS "
          f"(EMA-weight if v14)")

    metrics = evaluate_v3(model, test_loader, device)
    print_results(metrics, "v3")
    print_selection_analysis(analyze_checkpoint_selection(ckpt.get("history", {})))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/pipeline_v6.yaml")
    parser.add_argument("--data",       default="outputs/preprocessed/dataset.h5")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 路径，不提供则自动从 cfg.paths.checkpoints 推断")
    parser.add_argument("--seq_len",    type=int, default=None,
                        help="覆盖评估序列长度（默认用配置值）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_version = cfg["training"].get("model_version", "v1")
    if model_version == "v3":
        return main_v3(args, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: AnesthesiaNet {model_version.upper()}")

    # ── 确定 checkpoint 路径 ──────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_dir  = cfg.get("paths", {}).get("checkpoints", "outputs/checkpoints")
        ckpt_file = "best_model_v2.pt" if model_version == "v2" else "best_model.pt"
        ckpt_path = str(Path(ckpt_dir) / ckpt_file)
    print(f"Checkpoint: {ckpt_path}")

    # ── 构建测试集 ────────────────────────────────────────────────────────────
    seq_len = args.seq_len or cfg["training"]["seq_len"]
    tcfg = cfg["training"]
    _, _, test_ds = build_datasets(
        h5_path=args.data,
        val_split=tcfg["val_split"],
        test_split=tcfg["test_split"],
        seq_len=seq_len,
        seq_stride=seq_len,    # 非重叠，加快评估
        seed=tcfg["seed"],
        noise_std=0.0,
        cache_in_memory=True,
        case_std_pct_low=tcfg.get("case_std_pct_low", 0.0),
        case_std_pct_high=tcfg.get("case_std_pct_high", 100.0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Test sequences: {len(test_ds):,}  seq_len={seq_len}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    if model_version == "v2":
        model = AnesthesiaNetV2.from_config(cfg)
    else:
        model = AnesthesiaNet.from_config(cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    epoch = ckpt.get("epoch", "?")
    val_mae = ckpt.get("val_mae", float("nan"))
    print(f"Checkpoint from epoch {epoch},  val_MAE={val_mae:.2f} BIS")

    # ── 运行评估 ──────────────────────────────────────────────────────────────
    if model_version == "v2":
        metrics = evaluate_v2(model, test_loader, device)
    else:
        metrics = evaluate_v1(model, test_loader, device)

    print_results(metrics, model_version)
    return metrics


if __name__ == "__main__":
    main()
