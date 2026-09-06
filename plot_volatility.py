"""deep8 × 60 epoch 各变体 test_acc 波动分析图（含两轮全部 8 个变体）。

从 results/deep8_<variant>.csv 读取精度曲线，绘制：
  左图：8 个变体的 test_acc 随 epoch 变化（原始曲线）
  右图：滚动窗口波动率（标准差）随 epoch 变化
分组样式：第一轮 2x2 因子（细线）；第二轮积分修正方案 A/B/C（粗线）
输出保存到 results/deep8_volatility.png
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")  # 无界面后端，仅保存图片
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 全部 8 个变体：第一轮 4 个 + 第二轮积分修正 4 个
VARIANTS = ("baseline", "residual", "softmax", "both",
            "integral", "pid_full", "sumnorm", "sumnorm_residual")
# 各变体配色：第一轮用冷/灰系，第二轮方案A/B/C用高饱和色
COLORS = {
    "baseline":         "#999999",
    "residual":         "#1f77b4",
    "softmax":          "#ff7f0e",
    "both":             "#2ca02c",
    "integral":         "#d62728",   # 方案 A：泄漏积分器
    "pid_full":         "#9467bd",   # 方案 A：完整 PID
    "sumnorm":          "#8c564b",   # 方案 C：求和归一化
    "sumnorm_residual": "#e377c2",   # 方案 C：求和归一化 + 残差
}
# 第二轮变体（方案 A/C）用粗线突出显示
BOLD_VARIANTS = ("integral", "pid_full", "sumnorm", "sumnorm_residual")
WINDOW = 5  # 滚动波动率窗口（epoch）


def load_acc(path):
    """读取 csv，返回 (epochs, accs) 两个列表。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["epoch"]), float(r["test_acc"])))
            except (ValueError, KeyError, TypeError):
                continue
    rows.sort()
    return [e for e, _ in rows], [a for _, a in rows]


def rolling_std(xs, window=WINDOW):
    """滚动窗口标准差（窗口不足时用可得数据）。"""
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        seg = xs[lo:i + 1]
        out.append(st.pstdev(seg) if len(seg) >= 2 else 0.0)
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---- 左图：精度曲线 ----
    ax = axes[0]
    for v in VARIANTS:
        eps, accs = load_acc(os.path.join(RESULTS_DIR, f"deep8_{v}.csv"))
        lw = 2.2 if v in BOLD_VARIANTS else 1.1  # 第二轮变体用粗线
        ax.plot(eps, accs, label=v, color=COLORS[v], linewidth=lw, alpha=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test_acc")
    ax.set_title("deep8 x 60ep: test accuracy curves (round1 + round2)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # ---- 右图：滚动波动率 ----
    ax = axes[1]
    for v in VARIANTS:
        eps, accs = load_acc(os.path.join(RESULTS_DIR, f"deep8_{v}.csv"))
        lw = 2.2 if v in BOLD_VARIANTS else 1.1
        ax.plot(eps, rolling_std(accs), label=v, color=COLORS[v],
                linewidth=lw, alpha=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"rolling std (window={WINDOW})")
    ax.set_title("deep8 x 60ep: accuracy volatility (round1 + round2)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "deep8_volatility.png")
    fig.savefig(out_path, dpi=150)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
