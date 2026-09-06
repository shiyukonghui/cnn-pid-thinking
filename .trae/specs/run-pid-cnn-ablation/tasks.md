# Tasks

- [x] Task 1: 实现模型定义 `models.py`
  - [x] SubTask 1.1: 统一 CNN 骨干（3 个卷积阶段：3→32→64→128，每阶段 Conv-BN-ReLU ×2）
  - [x] SubTask 1.2: `use_residual` 开关：每阶段输出经 1×1 卷积投影后与阶段输入相加
  - [x] SubTask 1.3: `use_softmax` 开关：每阶段输出后插入 `nn.Softmax(dim=1)`
  - [x] SubTask 1.4: 提供工厂函数 `build_model(variant)`，映射 baseline / residual / softmax / both
  - [x] SubTask 1.5: 验证：脚本内对 4 个变体做形状冒烟测试（输入 (1,3,32,32) 输出 (1,10)）
- [x] Task 2: 实现训练/评估脚本 `train.py`
  - [x] SubTask 2.1: CIFAR-10 数据加载（标准增强：RandomCrop+Flip，Normalize；test 不增强）
  - [x] SubTask 2.2: 固定随机种子（torch/numpy/random），Adam 或 SGD 优化器、固定 epoch 数与 lr
  - [x] SubTask 2.3: 训练循环：逐 epoch 记录 train loss、test acc，写入 `results/<variant>.csv`
  - [x] SubTask 2.4: `--summarize` 模式：聚合 4 个 csv 生成 `results/summary.csv`
- [x] Task 3: 运行对照试验并产出结果
  - [x] SubTask 3.1: 依次运行 4 个变体训练（CPU 上用较小 epoch 数保证可完成，如 3 epoch）
  - [x] SubTask 3.2: 生成 `results/summary.csv` 并向用户报告对比结论

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 1 内部 SubTask 可并行；Task 3 各变体运行相互独立
