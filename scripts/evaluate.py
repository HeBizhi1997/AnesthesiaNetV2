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

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets
from src.models.anesthesia_net    import AnesthesiaNet
from src.models.anesthesia_net_v2 import AnesthesiaNetV2


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
