"""对照试验训练脚本（2x2 因子：残差连接 x 层间 softmax，CIFAR-10）。

用法：
    python train.py --variant baseline --epochs 3
    python train.py --summarize   # 汇总 results/ 下 4 个变体的最优精度
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import build_model

# CIFAR-10 各通道均值 / 标准差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

VARIANTS = ("baseline", "residual", "softmax", "both")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEEP_MODE = False  # --summarize 时标记是否汇总 5 阶段（deep5_ 前缀）结果


def set_seed(seed: int):
    """固定 random / numpy / torch（含 cuda）随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser(description="残差/层间softmax 2x2 因子对照试验（CIFAR-10）")
    parser.add_argument("--variant", choices=VARIANTS, default=None,
                        help="试验变体（--summarize 时可省略）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数（默认 3）")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小（默认 128）")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率（默认 0.001）")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam",
                        help="优化器（默认 adam）")
    parser.add_argument("--summarize", action="store_true",
                        help="汇总 results/ 下各变体 csv 的最优精度")
    parser.add_argument("--stages", type=int, default=3, choices=(3, 5),
                        help="骨干卷积阶段数（默认 3；5 为加深版）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()
    # variant 必填，除非使用 --summarize
    if not args.summarize and args.variant is None:
        parser.error("--variant 为必填参数（除非指定 --summarize）")
    return args


def build_dataloaders(batch_size: int):
    """构建 CIFAR-10 训练/测试 DataLoader（Windows 兼容：num_workers=0）。"""
    # 训练集：随机裁剪（四周补 4 像素）+ 随机水平翻转 + 标准化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    # 测试集：仅标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model, loader, device):
    """返回给定 loader 上的准确率（0~1）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_training(args):
    """执行单个变体的训练循环，逐 epoch 把结果追加写入 results/<variant>.csv。

    深度>3 时文件名带 deep_ 前缀（如 results/deep5_residual.csv），避免覆盖 3 阶段结果。
    """
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        else "cpu")
    print(f"使用设备: {device}  骨干: {args.stages} 阶段")

    # CUDA 可用时启用 cudnn 自动调优
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = build_model(args.variant, num_stages=args.stages).to(device)

    # 优化器
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = build_dataloaders(args.batch_size)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    name_prefix = f"deep{args.stages}_" if args.stages != 3 else ""
    csv_path = os.path.join(RESULTS_DIR, f"{name_prefix}{args.variant}.csv")
    # 追加写模式：若文件不存在则先写入表头
    need_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if need_header:
        writer.writerow(["epoch", "train_loss", "test_acc"])

    for epoch in range(1, args.epochs + 1):
        # ---- 训练一个 epoch，累计 train loss（按样本数平均）----
        model.train()
        total_loss, total_samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        train_loss = total_loss / total_samples

        # ---- 测试集评估 ----
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} test_acc={test_acc:.4f}")
        writer.writerow([epoch, f"{train_loss:.4f}", f"{test_acc:.4f}"])
        csv_file.flush()

    csv_file.close()
    print(f"结果已写入: {csv_path}")


def summarize():
    """读取 results/ 下 4 个变体 csv，取各自 test_acc 最大值，写 summary.csv 并打印。

    3 阶段与 5 阶段（deep5_ 前缀）结果分别汇总到 summary.csv / deep5_summary.csv。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    prefix = "deep5_" if DEEP_MODE else ""
    for variant in VARIANTS:
        path = os.path.join(RESULTS_DIR, f"{prefix}{variant}.csv")
        if not os.path.exists(path):
            print(f"警告: 缺少 {path}，跳过该变体")
            continue
        best_acc = 0.0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    best_acc = max(best_acc, float(row["test_acc"]))
                except (ValueError, KeyError):
                    continue
        rows.append((variant, best_acc))

    if not rows:
        print("results/ 下没有任何变体 csv，无法汇总。请先运行训练。")
        return

    summary_path = os.path.join(RESULTS_DIR, f"{prefix}summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "best_test_acc"])
        for variant, best_acc in rows:
            writer.writerow([variant, f"{best_acc:.4f}"])

    print("=== 汇总结果（各变体最优 test_acc）===")
    print(f"{'variant':<10} {'best_test_acc':>14}")
    for variant, best_acc in rows:
        print(f"{variant:<10} {best_acc:>14.4f}")
    print(f"已写入: {summary_path}")


def main():
    global DEEP_MODE
    args = get_args()
    DEEP_MODE = (args.stages == 5)
    if args.summarize:
        summarize()
    else:
        run_training(args)


if __name__ == "__main__":
    main()
