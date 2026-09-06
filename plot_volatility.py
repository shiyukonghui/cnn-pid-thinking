"""deep8 × 60 epoch 各变体 test_acc 波动分析图。

从 results/deep8_<variant>.csv 读取精度曲线，绘制：
  左图：4 个变体的 test_acc 随 epoch 变化（原始曲线）
  右图：滚动窗口波动率（标准差）随 epoch 变化
输出保存到 results/deep8_volatility.png
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")  # 无界面后端，仅保存图片
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
VARIANTS = ("baseline", "residual", "softmax", "both")
COLORS = {"baseline": "#888888", "residual": "#1f77b4",
          "softmax": "#ff7f0e", "both": "#2ca02c"}
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- 左图：精度曲线 ----
    ax = axes[0]
    for v in VARIANTS:
        eps, accs = load_acc(os.path.join(RESULTS_DIR, f"deep8_{v}.csv"))
        ax.plot(eps, accs, label=v, color=COLORS[v], linewidth=1.2, alpha=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test_acc")
    ax.set_title("deep8 x 60ep: test accuracy curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- 右图：滚动波动率 ----
    ax = axes[1]
    for v in VARIANTS:
        eps, accs = load_acc(os.path.join(RESULTS_DIR, f"deep8_{v}.csv"))
        ax.plot(eps, rolling_std(accs), label=v, color=COLORS[v],
                linewidth=1.2, alpha=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"rolling std (window={WINDOW})")
    ax.set_title("deep8 x 60ep: accuracy volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "deep8_volatility.png")
    fig.savefig(out_path, dpi=150)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
