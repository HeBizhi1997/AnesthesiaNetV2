# Task Plan: Bug修复 + 系统验证 + 重新训练

## 发现的全部Bug（按优先级）

### 🔴 严重 Bug（影响训练/评估结果）
1. **evaluate.py 硬编码 V1 模型** — 调用 `AnesthesiaNet`(V1) 并用3值解包，V2返回5值直接崩溃
2. **evaluate.py 相位划分错误** — 用BIS阈值(60/40)定义相位，实际应用`phase_labels`列标签
3. **trainer_v2.py val_phase_acc 从不计算** — history 里有但 val_epoch 从未填充，全是初始值
4. **tbptt_trainer.py 与 V2 模型不兼容** — PatientStore 不加载 phases/stim_events，model 调用解包错误

### 🟡 中等 Bug（影响可靠性）
5. **loader.py 归一化回退无警告** — 无清醒段时静默回退，无法感知数据质量
6. **trainer_v2.py 缺少刺激检测指标** — 无 AUROC/sensitivity/specificity 报告

### 🟢 设计改进
7. **validate_pipeline.py 缺失** — 无系统性验证框架
8. **训练日志不完整** — 缺少吞吐量、ETA、每相位指标实时显示
9. **pipeline_v7.yaml** — 集成所有修复的新配置

## 执行阶段

- [x] Phase 0: 读取所有源文件，全面理解代码
- [ ] Phase 1: 修复严重Bug (evaluate.py, trainer_v2.py, tbptt_trainer.py, loader.py)
- [ ] Phase 2: 实现系统性验证框架 (scripts/validate_pipeline.py)
- [ ] Phase 3: 改进训练日志 + 创建 pipeline_v7.yaml
- [ ] Phase 4: 启动训练，监控进度

## Status
**Currently in Phase 1** — 修复严重Bug
