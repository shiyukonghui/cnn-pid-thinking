# CNN-PID 对照试验 Spec

## Why
灵感来源.md 提出"残差连接 ≈ PID 微分、层间插入 softmax ≈ PID 积分、网络权重 ≈ 比例系数"的类比。需要通过 2×2 因子对照试验（残差 × 层间 softmax 的有无）量化两种组件对 CNN 性能的贡献，验证该类比是否成立。

## What Changes
- 新建 PyTorch 实验工程（当前仓库仅有灵感来源.md）
- 实现 4 个模型变体（同一骨干网络，通过开关控制组件）：
  1. Baseline：无残差、无层间 softmax
  2. 仅残差
  3. 仅多 softmax 插入（每个卷积阶段之间插入 Softmax）
  4. 残差 + 多 softmax 同时插入
- 实现统一训练/评估脚本：相同数据集、超参数、随机种子，保证对照公平
- 输出每个变体的训练曲线与最终精度，生成汇总对比表

## Impact
- Affected specs: 无（全新能力）
- Affected code:
  - `models.py`：4 个变体的网络定义
  - `train.py`：训练/评估入口（支持 `--variant` 参数）
  - `results/`：运行产生的日志、CSV 汇总表

## ADDED Requirements

### Requirement: 模型变体定义
系统 SHALL 在 `models.py` 中提供一个统一的 CNN 骨干（3 个卷积阶段），并通过 `use_residual`、`use_softmax` 两个布尔开关生成 4 个变体：
- 残差：每个阶段输出与输入做相加（通道数变化时用 1×1 卷积投影对齐）
- 层间 softmax：在每个卷积阶段输出后插入 `nn.Softmax(dim=1)`（沿通道维归一化）

#### Scenario: 构造 4 个变体
- **WHEN** 分别以 (False,False)、(True,False)、(False,True)、(True,True) 实例化模型并输入 CIFAR-10 形状张量 (1,3,32,32)
- **THEN** 4 个变体均输出形状 (1,10) 的 logits，无形状错误

#### Scenario: softmax 变体输出为概率分布
- **WHEN** 对启用 softmax 的变体输出特征沿通道维求和
- **THEN** 每个空间位置的和近似为 1（仅对插入 softmax 的位置）

### Requirement: 统一对照训练
系统 SHALL 提供训练脚本 `train.py`：
- 数据集：CIFAR-10（train/test 官方划分）
- 4 个变体使用完全相同的超参数与固定随机种子
- 支持命令行选择变体，训练过程中记录每 epoch 的 train loss、test accuracy
- 结果写入 `results/<variant>.csv`（逐 epoch 记录）

#### Scenario: 单变体完整训练
- **WHEN** 运行 `python train.py --variant baseline`
- **THEN** 训练正常完成并在 `results/` 生成 `baseline.csv`，包含逐 epoch 指标

### Requirement: 汇总对比
系统 SHALL 在 4 个变体全部训练完成后生成汇总表 `results/summary.csv`，包含每个变体的最优 test accuracy。

#### Scenario: 生成汇总
- **WHEN** 依次运行 4 个变体后运行 `python train.py --summarize`
- **THEN** `results/summary.csv` 存在且包含 4 行变体数据

## REMOVED Requirements
（无）
