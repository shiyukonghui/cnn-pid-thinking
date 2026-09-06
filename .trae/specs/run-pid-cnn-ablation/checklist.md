# Checklist

- [x] `models.py` 中 4 个变体（baseline/residual/softmax/both）均可实例化且前向输出形状为 (1,10)
- [x] 残差变体包含阶段输出与输入相加的结构（通道不一致时用 1×1 卷积对齐）
- [x] softmax 变体在每个卷积阶段后插入了 `nn.Softmax(dim=1)`
- [x] `train.py` 固定随机种子，4 个变体使用相同超参数与数据集（CIFAR-10）
- [x] 运行单变体训练后在 `results/` 生成对应 `<variant>.csv`
- [x] `--summarize` 生成 `results/summary.csv` 且包含 4 行结果
- [x] 最终向用户报告 4 个变体的精度对比结论
