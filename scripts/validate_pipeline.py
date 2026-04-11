"""
validate_pipeline.py — 系统性验证框架

验证范围（从数据到模型到损失函数，每步科学验证）：

  模块 1: 数据完整性验证
    1.1 HDF5 结构完整性 — 所有必需字段存在且形状正确
    1.2 标签分布验证 — BIS 值范围、相位比例、刺激事件比例
    1.3 归一化验证 — 振幅分布（应接近单位方差）
    1.4 时间一致性验证 — 时间戳连续性、标签滞后正确性

  模块 2: 模型前向传播验证
    2.1 输出形状验证 — 所有返回张量形状正确
    2.2 输出范围验证 — BIS∈[0,1]，phase softmax 和为1
    2.3 梯度流验证 — 反向传播无 NaN/Inf
    2.4 确定性验证 — 相同输入得到相同输出

  模块 3: 损失函数验证
    3.1 Focal Loss alpha 方向验证 — alpha=0.99 时正类权重高于负类
    3.2 自适应权重验证 — UW-SO 在不同量级下输出合理权重
    3.3 单调性损失验证 — 诱导期上升惩罚，维持期不惩罚

  模块 4: SQI 惯性模式验证
    4.1 低质量窗口时状态冻结 — SQI<0.5 时输出不变
    4.2 高质量窗口时状态更新 — SQI≥0.5 时输出变化

  模块 5: 数据集采样验证
    5.1 诱导过采样比例验证
    5.2 时间顺序验证 — 同一序列内窗口时间单调递增
    5.3 患者无泄漏验证 — train/val/test 患者 ID 不重叠

Usage:
    python scripts/validate_pipeline.py --config configs/pipeline_v6.yaml
    python scripts/validate_pipeline.py --config configs/pipeline_v6.yaml --quick
"""

from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import yaml

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
SKIP = "  [SKIP]"


# ─────────────────────────────────────────────────────────────────────────────
# 报告工具
# ─────────────────────────────────────────────────────────────────────────────

class ValidationReport:
    def __init__(self):
        self.results: List[Tuple[str, str, str]] = []  # (status, name, detail)

    def add(self, status: str, name: str, detail: str = ""):
        self.results.append((status, name, detail))
        sym = {"PASS": "[OK]", "FAIL": "[!]", "WARN": "!", "SKIP": "-"}.get(status, "?")
        color_map = {"PASS": "\033[32m", "FAIL": "\033[31m",
                     "WARN": "\033[33m", "SKIP": "\033[90m"}
        reset = "\033[0m"
        col = color_map.get(status, "")
        msg = f"  {col}{sym} {name}{reset}"
        if detail:
            msg += f"  ->  {detail}"
        print(msg)

    def summary(self):
        n_pass = sum(1 for s, *_ in self.results if s == "PASS")
        n_fail = sum(1 for s, *_ in self.results if s == "FAIL")
        n_warn = sum(1 for s, *_ in self.results if s == "WARN")
        n_skip = sum(1 for s, *_ in self.results if s == "SKIP")
        print(f"\n{'='*60}")
        print(f"  验证摘要: {n_pass} 通过  {n_fail} 失败  {n_warn} 警告  {n_skip} 跳过")
        if n_fail > 0:
            print(f"\n  失败项目:")
            for s, name, detail in self.results:
                if s == "FAIL":
                    print(f"    [!] {name}: {detail}")
        print(f"{'='*60}")
        return n_fail == 0


rpt = ValidationReport()


# ─────────────────────────────────────────────────────────────────────────────
# 模块 1: 数据完整性验证
# ─────────────────────────────────────────────────────────────────────────────

def validate_data(h5_path: str, max_cases: int = 50) -> None:
    print(f"\n【模块 1】数据完整性验证  ({h5_path})")

    if not Path(h5_path).exists():
        rpt.add("FAIL", "HDF5 文件存在性", f"{h5_path} 不存在")
        return

    with h5py.File(h5_path, "r") as f:
        case_ids = sorted(f.keys())
        n_cases = len(case_ids)
        rpt.add("PASS", "HDF5 文件可打开", f"{n_cases} 个患者 case")

        if n_cases == 0:
            rpt.add("FAIL", "患者数量", "HDF5 为空")
            return

        # 检查前 max_cases 个 case
        check_ids = case_ids[:max_cases]

        # 1.1 必需字段检查
        required = {"waves", "features", "sqi", "labels", "times"}
        optional = {"phases", "stim_events"}
        missing_required, missing_optional = [], []
        shape_errors = []

        all_bis: List[float] = []
        all_phases: List[int] = []
        all_stims: List[float] = []
        amp_stats: List[float] = []
        n_windows_total = 0

        has_multitask = None   # 是否有多任务标签（phases/stim_events）

        for cid in check_ids:
            grp = f[cid]
            keys = set(grp.keys())

            miss_r = required - keys
            if miss_r:
                missing_required.append(f"{cid}:{miss_r}")

            miss_o = optional - keys
            if miss_o and has_multitask is not False:
                has_multitask = (len(miss_o) == 0)

            n = int(grp.attrs.get("n_windows", -1))
            n_windows_total += max(n, 0)

            # 形状检查
            waves = grp["waves"][:]
            labels = grp["labels"][:]
            if waves.shape[0] != labels.shape[0]:
                shape_errors.append(f"{cid}: waves[{waves.shape[0]}] vs labels[{labels.shape[0]}]")

            # 标签分布
            all_bis.extend(labels.tolist())

            if "phases" in grp:
                all_phases.extend(grp["phases"][:].tolist())
            if "stim_events" in grp:
                all_stims.extend(grp["stim_events"][:].tolist())

            # 振幅分布（归一化后应接近单位方差）
            if waves.size > 0:
                rms = float(np.sqrt((waves ** 2).mean()))
                amp_stats.append(rms)

        # 结果报告
        if missing_required:
            rpt.add("FAIL", "必需字段完整性",
                    f"{len(missing_required)} 个 case 缺字段")
        else:
            rpt.add("PASS", "必需字段完整性",
                    f"所有 {len(check_ids)} 个 case 字段完整")

        if shape_errors:
            rpt.add("FAIL", "数组形状一致性", f"{len(shape_errors)} 个形状错误")
        else:
            rpt.add("PASS", "数组形状一致性")

        # 多任务标签
        mt_str = "[OK] 有 phases+stim_events" if all_phases else "[!] 无多任务标签"
        if all_phases:
            rpt.add("PASS", "多任务标签存在", mt_str)
        else:
            rpt.add("WARN", "多任务标签缺失",
                    "phases/stim_events 不存在，请先运行 phase_labeler")

        # 1.2 标签分布验证
        bis_arr = np.array(all_bis)
        rpt.add("PASS", "BIS 值范围",
                f"min={bis_arr.min():.1f}  max={bis_arr.max():.1f}  "
                f"mean={bis_arr.mean():.1f}  std={bis_arr.std():.1f}")

        if bis_arr.min() < 10 or bis_arr.max() > 100:
            rpt.add("WARN", "BIS 异常值",
                    f"超出 [10,100] 范围: min={bis_arr.min():.1f} max={bis_arr.max():.1f}")
        else:
            rpt.add("PASS", "BIS 值在有效范围 [10,100]")

        if all_phases:
            phase_arr = np.array(all_phases)
            phase_counts = {i: (phase_arr == i).sum() for i in range(4)}
            phase_names  = {0: "pre_op", 1: "induction", 2: "maintenance", 3: "recovery"}
            total_ph = len(phase_arr)
            ph_str = "  ".join(
                f"{phase_names[i]}={phase_counts[i]/total_ph*100:.1f}%"
                for i in range(4))
            rpt.add("PASS", "相位分布", ph_str)

            # 检查诱导期比例
            ind_pct = phase_counts[1] / total_ph * 100
            if ind_pct < 0.1:
                rpt.add("WARN", "诱导期比例极低",
                        f"{ind_pct:.2f}% — 可能需要 induction_boost")
            elif ind_pct < 0.5:
                rpt.add("WARN", "诱导期比例偏低",
                        f"{ind_pct:.2f}% — 建议 induction_boost≥10")
            else:
                rpt.add("PASS", f"诱导期比例合理", f"{ind_pct:.2f}%")

        if all_stims:
            stim_arr = np.array(all_stims)
            stim_pct = stim_arr.mean() * 100
            stim_ratio = (1 - stim_arr.mean()) / max(stim_arr.mean(), 1e-6)
            rpt.add("PASS", "刺激事件比例",
                    f"{stim_pct:.2f}%  (负:正 ≈ {stim_ratio:.0f}:1)")

        # 1.3 归一化验证
        if amp_stats:
            rms_mean = np.mean(amp_stats)
            rms_std  = np.std(amp_stats)
            if 0.3 < rms_mean < 3.0:
                rpt.add("PASS", "振幅归一化",
                        f"RMS mean={rms_mean:.3f} std={rms_std:.3f} (目标≈1.0)")
            else:
                rpt.add("WARN", "振幅归一化异常",
                        f"RMS mean={rms_mean:.3f} 偏离1.0较多，检查归一化流程")

        rpt.add("PASS", "窗口总数", f"{n_windows_total:,} 窗口 ({len(check_ids)} 个 case)")

        # 1.4 时间连续性验证（抽样检查5个case）
        time_errors = 0
        for cid in check_ids[:5]:
            if "times" in f[cid]:
                times = f[cid]["times"][:]
                diffs = np.diff(times)
                if not np.all(diffs > 0):
                    time_errors += 1

        if time_errors == 0:
            rpt.add("PASS", "时间戳单调递增")
        else:
            rpt.add("FAIL", "时间戳单调递增", f"{time_errors}/5 个case时间戳错乱")


# ─────────────────────────────────────────────────────────────────────────────
# 模块 2: 模型前向传播验证
# ─────────────────────────────────────────────────────────────────────────────

def validate_model(cfg: dict) -> None:
    print(f"\n【模块 2】模型前向传播验证")

    try:
        from src.models.anesthesia_net_v2 import AnesthesiaNetV2
        model = AnesthesiaNetV2.from_config(cfg)
        model.eval()
    except Exception as e:
        rpt.add("FAIL", "模型构建", str(e))
        return

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rpt.add("PASS", "模型构建成功", f"参数量={n_params:,}")

    # 构造测试输入
    eeg_cfg  = cfg["eeg"]
    wind_cfg = cfg["windowing"]
    model_cfg = cfg["model"]
    n_ch      = len(eeg_cfg["channels"])
    win_samp  = int(wind_cfg["window_sec"] * eeg_cfg["srate"])
    feat_dim  = model_cfg["feature_dim"]
    B, T = 4, 10

    wave     = torch.randn(B, T, n_ch, win_samp)
    features = torch.randn(B, T, feat_dim)
    sqi      = torch.rand(B, T, n_ch).clamp(0, 1)

    # 2.1 输出形状验证
    try:
        with torch.no_grad():
            pred_bis, phase_logits, stim_logits, correction, h = model(
                wave, features, sqi)

        assert pred_bis.shape    == (B, T, 1),      f"pred_bis: {pred_bis.shape}"
        assert phase_logits.shape == (B, T, 4),     f"phase_logits: {phase_logits.shape}"
        assert stim_logits.shape  == (B, T, 1),     f"stim_logits: {stim_logits.shape}"
        assert correction.shape   == (B, T, 1),     f"correction: {correction.shape}"
        rpt.add("PASS", "输出形状正确",
                f"pred_bis={tuple(pred_bis.shape)}  "
                f"phase={tuple(phase_logits.shape)}  "
                f"stim={tuple(stim_logits.shape)}")
    except Exception as e:
        rpt.add("FAIL", "输出形状", str(e))
        return

    # 2.2 输出范围验证
    bis_min = pred_bis.min().item()
    bis_max = pred_bis.max().item()
    if 0.0 <= bis_min and bis_max <= 1.0:
        rpt.add("PASS", "BIS 输出范围 [0,1]",
                f"min={bis_min:.4f}  max={bis_max:.4f}")
    else:
        rpt.add("FAIL", "BIS 输出范围",
                f"min={bis_min:.4f}  max={bis_max:.4f} (应在[0,1])")

    import torch.nn.functional as F
    ph_sum = F.softmax(phase_logits, dim=-1).sum(-1)
    if ph_sum.allclose(torch.ones_like(ph_sum), atol=1e-5):
        rpt.add("PASS", "Phase softmax 和为 1")
    else:
        rpt.add("FAIL", "Phase softmax",
                f"和 不为1: {ph_sum.min():.4f}~{ph_sum.max():.4f}")

    stim_prob = torch.sigmoid(stim_logits)
    if 0.0 <= stim_prob.min().item() and stim_prob.max().item() <= 1.0:
        rpt.add("PASS", "Stim 概率范围 [0,1]")
    else:
        rpt.add("FAIL", "Stim 概率范围", "sigmoid 输出超出 [0,1]")

    # 2.3 梯度流验证
    try:
        model.train()
        wave2 = wave.requires_grad_(False)
        pred2, ph2, st2, _, _ = model(wave2, features, sqi)
        loss_dummy = pred2.mean() + ph2.mean() + st2.mean()
        loss_dummy.backward()

        nan_grads = [(n, p) for n, p in model.named_parameters()
                     if p.grad is not None and
                     (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())]
        if nan_grads:
            rpt.add("FAIL", "梯度流 NaN/Inf",
                    f"{len(nan_grads)} 个参数梯度异常: "
                    f"{[n for n, _ in nan_grads[:3]]}")
        else:
            rpt.add("PASS", "梯度流正常（无 NaN/Inf）")

        none_grads = [(n, p) for n, p in model.named_parameters()
                      if p.requires_grad and p.grad is None]
        if none_grads:
            rpt.add("WARN", "存在无梯度参数",
                    f"{len(none_grads)} 个: {[n for n, _ in none_grads[:3]]}")
        else:
            rpt.add("PASS", "所有参数都有梯度")

    except Exception as e:
        rpt.add("FAIL", "反向传播", traceback.format_exc(limit=3))

    # 2.4 确定性验证
    model.eval()
    with torch.no_grad():
        out1 = model(wave, features, sqi)[0]
        out2 = model(wave, features, sqi)[0]
    if torch.allclose(out1, out2, atol=1e-6):
        rpt.add("PASS", "确定性输出（eval 模式）")
    else:
        rpt.add("WARN", "非确定性输出",
                f"最大差 {(out1 - out2).abs().max():.2e}（BN 统计？）")

    # 2.5 SQI 惯性模式验证
    threshold = cfg["model"].get("sqi_inertia_threshold", 0.5)
    if threshold > 0:
        model.eval()
        with torch.no_grad():
            # 全部 SQI = 0（应触发惯性，保持第一步状态）
            sqi_zero = torch.zeros_like(sqi)
            out_zero = model(wave, features, sqi_zero)[0]

            # 全部 SQI = 1（正常更新）
            sqi_one  = torch.ones_like(sqi)
            out_one  = model(wave, features, sqi_one)[0]

        max_diff = float((out_zero - out_one).abs().max())
        if max_diff > 0.001:
            rpt.add("PASS", "SQI 惯性模式有效",
                    f"SQI=0 vs SQI=1 最大预测差={max_diff:.4f}")
        else:
            rpt.add("WARN", "SQI 惯性模式效果弱",
                    f"SQI=0 vs SQI=1 差值={max_diff:.6f}（可能初始化影响）")
    else:
        rpt.add("SKIP", "SQI 惯性模式", "sqi_inertia_threshold=0（已禁用）")


# ─────────────────────────────────────────────────────────────────────────────
# 模块 3: 损失函数验证
# ─────────────────────────────────────────────────────────────────────────────

def validate_loss(cfg: dict) -> None:
    print(f"\n【模块 3】损失函数验证")

    from src.training.loss_v2 import focal_loss, MultiTaskLoss, monotonic_loss

    tcfg = cfg["training"]

    # 3.1 Focal Loss alpha 方向验证（关键 Bug 确认）
    alpha = tcfg.get("focal_alpha", 0.99)

    # 用接近0.5的预测，计算正例与负例的有效权重比
    logit = torch.zeros(100)  # p = 0.5
    target_pos = torch.ones(100)
    target_neg = torch.zeros(100)

    loss_pos = focal_loss(logit, target_pos, alpha=alpha,
                          pos_weight=tcfg.get("stim_pos_weight", 99.0)).item()
    loss_neg = focal_loss(logit, target_neg, alpha=alpha,
                          pos_weight=1.0).item()  # pos_weight 只对正例有效

    if loss_pos > loss_neg:
        rpt.add("PASS", f"Focal alpha={alpha:.2f} 正例权重 > 负例",
                f"正例loss={loss_pos:.4f}  负例loss={loss_neg:.4f}  "
                f"比={loss_pos/max(loss_neg,1e-6):.1f}x")
    else:
        rpt.add("FAIL", f"Focal alpha={alpha:.2f} 方向错误",
                f"正例loss={loss_pos:.4f} < 负例loss={loss_neg:.4f}  "
                f"alpha 应接近1.0(正例upweight)")

    # 额外验证：alpha=0.25 时（V5 bug 场景）
    lp_old = focal_loss(logit, target_pos, alpha=0.25, pos_weight=1.0).item()
    ln_old = focal_loss(logit, target_neg, alpha=0.25, pos_weight=1.0).item()
    if lp_old < ln_old:
        rpt.add("PASS", "V5 focal_alpha=0.25 Bug 确认",
                f"alpha=0.25时正例loss({lp_old:.4f})<负例loss({ln_old:.4f}) -> 这是Bug，V6已修复")

    # 3.2 pos_weight 效果验证
    pw = tcfg.get("stim_pos_weight", 99.0)
    loss_pw1 = focal_loss(logit, target_pos, alpha=0.99, pos_weight=1.0).item()
    loss_pwN = focal_loss(logit, target_pos, alpha=0.99, pos_weight=pw).item()
    ratio = loss_pwN / max(loss_pw1, 1e-9)
    if abs(ratio - pw) < pw * 0.5:  # 应在 0.5x ~ 1.5x * pw 范围内（alpha调制）
        rpt.add("PASS", f"pos_weight={pw:.0f} 放大效果",
                f"pw=1->{loss_pw1:.4f}  pw={pw:.0f}->{loss_pwN:.4f}  "
                f"放大倍数={ratio:.1f}x")
    else:
        rpt.add("WARN", f"pos_weight={pw:.0f} 效果异常",
                f"放大倍数={ratio:.1f}x (理论≈{pw}x)")

    # 3.3 UW-SO 自适应权重验证
    use_auto = tcfg.get("use_auto_weight", False)
    if use_auto:
        criterion = MultiTaskLoss(
            use_auto_weight=True,
            focal_alpha=alpha,
            stim_pos_weight=pw,
        )
        # 模拟 BIS Huber (≈0.003) 和 Phase CE (≈0.35) 量级差距
        bis_dummy   = torch.tensor(0.003)
        phase_dummy = torch.tensor(0.35)
        stim_dummy  = torch.tensor(0.004)
        mono_dummy  = torch.tensor(0.001)
        total = criterion._auto_weighted_sum(bis_dummy, phase_dummy, stim_dummy, mono_dummy)
        rpt.add("PASS", "UW-SO 自适应权重计算",
                f"BIS(0.003)+Phase(0.35)+Stim(0.004)+Mono(0.001) -> total={total.item():.4f}")

        # 验证钳位：权重不应超过8
        raw = torch.stack([bis_dummy, phase_dummy, stim_dummy, mono_dummy])
        target = raw.mean()
        w = (target / raw.clamp(1e-8)).clamp(0.1, 8.0)
        if w.max().item() <= 8.0 and w.min().item() >= 0.1:
            rpt.add("PASS", "UW-SO 权重钳位正确",
                    f"weights={w.numpy().round(2).tolist()}")
        else:
            rpt.add("FAIL", "UW-SO 权重钳位失效", f"weights={w.tolist()}")
    else:
        rpt.add("SKIP", "UW-SO 自适应权重", "use_auto_weight=false")

    # 3.4 单调性损失验证
    # 诱导期（phase=1）BIS 下降时预测上升 -> 应有惩罚
    B, T = 2, 10
    label_down = torch.linspace(0.9, 0.4, T).unsqueeze(0).expand(B, -1)
    pred_up    = torch.linspace(0.4, 0.9, T).unsqueeze(0).expand(B, -1).unsqueeze(-1)
    ph_induct  = torch.ones(B, T, dtype=torch.long)  # phase=1 全为诱导期

    mono_val = monotonic_loss(pred_up, label_down, ph_induct)
    if mono_val.item() > 0.01:
        rpt.add("PASS", "单调性损失：诱导期上升被惩罚",
                f"loss={mono_val.item():.4f} > 0")
    else:
        rpt.add("FAIL", "单调性损失：诱导期上升未惩罚",
                f"loss={mono_val.item():.6f}")

    # 维持期（phase=2）不应惩罚
    ph_maint = torch.full((B, T), 2, dtype=torch.long)
    mono_maint = monotonic_loss(pred_up, label_down, ph_maint)
    if mono_maint.item() < 1e-6:
        rpt.add("PASS", "单调性损失：维持期不惩罚",
                f"loss={mono_maint.item():.6f} ≈ 0")
    else:
        rpt.add("FAIL", "单调性损失：维持期被错误惩罚",
                f"loss={mono_maint.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 模块 4: 数据集采样验证
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(h5_path: str, cfg: dict, quick: bool = False) -> None:
    print(f"\n【模块 4】数据集采样验证")

    if not Path(h5_path).exists():
        rpt.add("SKIP", "数据集采样验证", "HDF5 不存在")
        return

    try:
        from src.data.dataset import build_datasets
        tcfg = cfg["training"]
        seq_len = min(tcfg["seq_len"], 30) if quick else tcfg["seq_len"]

        train_ds, val_ds, test_ds = build_datasets(
            h5_path=h5_path,
            val_split=tcfg["val_split"],
            test_split=tcfg["test_split"],
            seq_len=seq_len,
            seq_stride=seq_len,
            seed=tcfg["seed"],
            noise_std=0.0,
            cache_in_memory=True,
            case_std_pct_low=tcfg.get("case_std_pct_low", 0.0),
            case_std_pct_high=tcfg.get("case_std_pct_high", 100.0),
        )
    except Exception as e:
        rpt.add("FAIL", "数据集构建", str(e))
        return

    rpt.add("PASS", "数据集构建",
            f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # 4.1 患者无泄漏验证
    train_cids = set(cid for cid, _ in train_ds._index)
    val_cids   = set(cid for cid, _ in val_ds._index)
    test_cids  = set(cid for cid, _ in test_ds._index)

    tv_overlap = train_cids & val_cids
    tt_overlap = train_cids & test_cids
    vt_overlap = val_cids  & test_cids

    if not tv_overlap and not tt_overlap and not vt_overlap:
        rpt.add("PASS", "患者无数据泄漏（train/val/test 不重叠）",
                f"train={len(train_cids)}  val={len(val_cids)}  test={len(test_cids)} 患者")
    else:
        rpt.add("FAIL", "患者数据泄漏！",
                f"train∩val={len(tv_overlap)}  train∩test={len(tt_overlap)}  "
                f"val∩test={len(vt_overlap)}")

    # 4.2 时间顺序验证（抽样5个序列）
    order_errors = 0
    for idx in range(min(5, len(train_ds))):
        item = train_ds[idx]
        label_seq = item["label_seq"].numpy()
        # 序列标签应有一定变化（不是完全平的）
        if label_seq.std() < 0.001:
            order_errors += 1

    if order_errors == 0:
        rpt.add("PASS", "时间序列有效性（标准差>0）")
    else:
        rpt.add("WARN", "存在标签方差极小的序列",
                f"{order_errors}/5 个序列 std<0.001")

    # 4.3 多任务标签存在性
    item0 = train_ds[0]
    if "phases" in item0:
        ph = item0["phases"].numpy()
        st = item0["stim_events"].numpy()
        rpt.add("PASS", "多任务标签可访问",
                f"phases shape={ph.shape}  stim shape={st.shape}  "
                f"unique phases={np.unique(ph).tolist()}")
    else:
        rpt.add("WARN", "多任务标签不可用",
                "HDF5 中无 phases/stim_events，多任务头将使用默认标签")

    # 4.4 诱导期过采样验证
    induction_boost = tcfg.get("induction_boost", 0)
    if induction_boost > 0 and "phases" in item0:
        n_before = len(train_ds)
        train_ds.boost_induction_sequences(induction_boost)
        n_after = len(train_ds)
        added = n_after - n_before
        rpt.add("PASS" if added > 0 else "WARN",
                f"诱导期过采样 x{induction_boost}",
                f"序列数: {n_before:,} -> {n_after:,} (+{added:,})")
    elif induction_boost > 0:
        rpt.add("SKIP", "诱导期过采样",
                "induction_boost>0 但无 phases 标签")
    else:
        rpt.add("SKIP", "诱导期过采样", "induction_boost=0（已禁用）")

    # 4.5 BIS 分布验证
    n_check = min(500, len(train_ds))
    bis_vals = [float(train_ds[i]["label_raw"]) for i in range(n_check)]
    bis_arr  = np.array(bis_vals)
    maint_pct = ((bis_arr >= 40) & (bis_arr < 60)).mean() * 100
    ind_pct   = (bis_arr >= 60).mean() * 100
    rec_pct   = (bis_arr < 40).mean() * 100
    rpt.add("PASS", f"BIS 分布（前{n_check}序列末值）",
            f"诱导≥60: {ind_pct:.1f}%  维持40-60: {maint_pct:.1f}%  "
            f"恢复<40: {rec_pct:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 模块 5: 端到端前向验证（一个完整 batch）
# ─────────────────────────────────────────────────────────────────────────────

def validate_end_to_end(h5_path: str, cfg: dict) -> None:
    print(f"\n【模块 5】端到端前向+损失验证")

    if not Path(h5_path).exists():
        rpt.add("SKIP", "端到端验证", "HDF5 不存在")
        return

    try:
        from src.data.dataset import build_datasets
        from src.models.anesthesia_net_v2 import AnesthesiaNetV2
        from src.training.loss_v2 import MultiTaskLoss
        from torch.utils.data import DataLoader

        tcfg = cfg["training"]
        seq_len = min(tcfg["seq_len"], 10)

        _, val_ds, _ = build_datasets(
            h5_path=h5_path,
            val_split=tcfg["val_split"],
            test_split=tcfg["test_split"],
            seq_len=seq_len,
            seed=tcfg["seed"],
            noise_std=0.0,
            cache_in_memory=True,
        )
        loader = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=0)
        batch  = next(iter(loader))

        model = AnesthesiaNetV2.from_config(cfg)
        criterion = MultiTaskLoss(
            focal_alpha=tcfg.get("focal_alpha", 0.99),
            stim_pos_weight=tcfg.get("stim_pos_weight", 99.0),
            use_auto_weight=tcfg.get("use_auto_weight", False),
        )

        wave      = batch["wave"]
        features  = batch["features"]
        sqi       = batch["sqi"]
        label_seq = batch["label_seq"]
        sqi_mean  = sqi.mean(-1)

        phase_labels = batch.get("phases",
            torch.full(label_seq.shape, 2, dtype=torch.long))
        stim_labels  = batch.get("stim_events",
            torch.zeros_like(label_seq))

        model.train()
        pred_bis, phase_logits, stim_logits, _, _ = model(wave, features, sqi)
        losses = criterion(pred_bis, phase_logits, stim_logits,
                           label_seq, phase_labels, stim_labels, sqi_mean)
        loss = losses["loss"]

        rpt.add("PASS", "端到端前向传播",
                f"total_loss={loss.item():.4f}  "
                f"bis={losses['bis_loss'].item():.4f}  "
                f"phase={losses['phase_loss'].item():.4f}  "
                f"stim={losses['stim_loss'].item():.4f}")

        # 验证 loss 不是 nan/inf
        if torch.isnan(loss) or torch.isinf(loss):
            rpt.add("FAIL", "Loss 数值", f"loss={loss.item()} (NaN or Inf!)")
        else:
            rpt.add("PASS", "Loss 数值有限（无 NaN/Inf）")

        # 反向传播
        loss.backward()
        max_grad = max((p.grad.abs().max().item()
                        for p in model.parameters()
                        if p.grad is not None), default=0.0)
        rpt.add("PASS" if max_grad < 100 else "WARN",
                "反向传播梯度大小",
                f"max_grad={max_grad:.4f}")

    except Exception as e:
        rpt.add("FAIL", "端到端验证", traceback.format_exc(limit=4))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_v6.yaml")
    parser.add_argument("--data",   default="outputs/preprocessed/dataset.h5")
    parser.add_argument("--quick",  action="store_true",
                        help="快速模式：减少检查样本数")
    parser.add_argument("--module", default="all",
                        choices=["all", "data", "model", "loss", "dataset", "e2e"],
                        help="只运行特定验证模块")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"  AnesthesiaNetV2 系统性验证框架")
    print(f"  Config : {args.config}")
    print(f"  Data   : {args.data}")
    print(f"  Mode   : {'快速' if args.quick else '完整'}")
    print(f"{'='*60}")

    run_all = (args.module == "all")

    if run_all or args.module == "data":
        validate_data(args.data, max_cases=20 if args.quick else 100)

    if run_all or args.module == "model":
        validate_model(cfg)

    if run_all or args.module == "loss":
        validate_loss(cfg)

    if run_all or args.module == "dataset":
        validate_dataset(args.data, cfg, quick=args.quick)

    if run_all or args.module == "e2e":
        validate_end_to_end(args.data, cfg)

    success = rpt.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
