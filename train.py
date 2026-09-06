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

from models import VARIANT_MAPPING, build_model

# CIFAR-10 各通道均值 / 标准差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

VARIANTS = tuple(VARIANT_MAPPING)  # 从 models.py 动态获取全部变体（含第二轮积分修正变体）
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEEP_MODE = False  # --summarize 时标记是否汇总更深层（deep5_/deep8_ 前缀）结果
# 文件名前缀与 stages 的映射（3 阶段无前缀，保持向后兼容）
STAGE_PREFIX = {3: "", 5: "deep5_", 8: "deep8_"}


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
    parser.add_argument("--stages", type=int, default=3, choices=(3, 5, 8),
                        help="骨干卷积阶段数（默认 3；5/8 为加深版）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader 工作进程数（默认 4，Windows 下仅 CUDA/CPU 加速时生效）")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="启用 AMP 混合精度（CUDA 上默认开启，用 --no-amp 关闭）")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="禁用 AMP 混合精度")
    parser.add_argument("--cos", action="store_true", default=None,
                        help="启用余弦退火学习率（长训练默认建议开启；默认随 optimizer 自动决定）")
    parser.add_argument("--warmup", type=int, default=0,
                        help="学习率线性 warmup 轮数（默认 0，仅 cos 模式生效）")
    args = parser.parse_args()
    # variant 必填，除非使用 --summarize
    if not args.summarize and args.variant is None:
        parser.error("--variant 为必填参数（除非指定 --summarize）")
    return args


def build_dataloaders(batch_size: int, num_workers: int = 0):
    """构建 CIFAR-10 训练/测试 DataLoader。"""
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
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
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
    """执行单个变体的训练循环，逐 epoch 把结果追加写入 results/<prefix><variant>.csv。

    文件名前缀按 stages 决定（3 无前缀 / deep5_ / deep8_），避免不同深度结果互相覆盖。
    """
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        else "cpu")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # AMP：默认仅在 CUDA 上启用（--no-amp 可关闭）
    use_amp = (args.amp if args.amp is not None else use_cuda) and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"使用设备: {device}  骨干: {args.stages} 阶段  AMP: {'开' if use_amp else '关'}")

    model = build_model(args.variant, num_stages=args.stages).to(device)

    # 优化器：sgd 默认带动量与 weight_decay（长训练极限性能更稳定），adam 保持原样
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 学习率调度：--cos 显式开启；未指定时 sgd+多轮训练 自动启用余弦退火
    use_cos = args.cos if args.cos is not None else (args.optimizer == "sgd" and args.epochs >= 10)
    scheduler = None
    if use_cos:
        warmup = min(args.warmup, max(args.epochs - 1, 0))
        # 余弦退火到接近 0；warmup 轮内由下方循环手动线性提升
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - warmup, 1), eta_min=args.lr * 1e-3)

    train_loader, test_loader = build_dataloaders(args.batch_size, args.num_workers)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    name_prefix = STAGE_PREFIX.get(args.stages, f"deep{args.stages}_")
    csv_path = os.path.join(RESULTS_DIR, f"{name_prefix}{args.variant}.csv")
    # 追加写模式：若文件不存在则先写入表头
    need_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if need_header:
        writer.writerow(["epoch", "train_loss", "test_acc", "lr"])

    for epoch in range(1, args.epochs + 1):
        # ---- warmup：线性提升学习率到目标值（仅 cos 模式且 epoch<=warmup 时生效）----
        if scheduler is not None and epoch <= args.warmup:
            warm_lr = args.lr * epoch / max(args.warmup, 1)
            for g in optimizer.param_groups:
                g["lr"] = warm_lr

        # ---- 训练一个 epoch，累计 train loss（按样本数平均）----
        model.train()
        total_loss, total_samples = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        train_loss = total_loss / total_samples

        # ---- 测试集评估 ----
        test_acc = evaluate(model, test_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} train_loss={train_loss:.4f} test_acc={test_acc:.4f} lr={cur_lr:.6f}")
        writer.writerow([epoch, f"{train_loss:.4f}", f"{test_acc:.4f}", f"{cur_lr:.6f}"])
        csv_file.flush()

        # ---- 调度器步进：warmup 期间不退火 ----
        if scheduler is not None and epoch > args.warmup:
            scheduler.step()

    csv_file.close()
    print(f"结果已写入: {csv_path}")


def summarize():
    """读取 results/ 下 4 个变体 csv，取各自 test_acc 最大值，写 summary.csv 并打印。

    3 阶段 / 5 阶段（deep5_）/ 8 阶段（deep8_）分别汇总到对应 summary 文件。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    prefix = STAGE_PREFIX.get(args.stages_global, "")
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
    global DEEP_MODE, args
    args = get_args()
    # summarize() 里通过全局变量获取 stages 前缀
    args.stages_global = args.stages
    DEEP_MODE = (args.stages != 3)
    if args.summarize:
        summarize()
    else:
        run_training(args)


if __name__ == "__main__":
    main()
