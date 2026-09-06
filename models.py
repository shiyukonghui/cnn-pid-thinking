"""对照试验模型定义（2x2 因子：残差连接 x 层间 softmax）。

验证类比假设：
  - 残差连接   ≈ PID 中的"微分"项（为层间变化提供快速直通校正通路）
  - 层间 softmax ≈ PID 中的"积分"项（对通道维分布做归一化累积约束）

通过 (use_residual, use_softmax) 的有无组合出 4 个变体：
  baseline / residual / softmax / both
"""
import torch
import torch.nn as nn

# 各阶段通道数：3 阶段 / 5 阶段两种骨干
STAGE_CHANNELS = (3, 32, 64, 128)          # 3 阶段：32->16->8->4
STAGE_CHANNELS_DEEP = (3, 32, 32, 64, 128, 256)  # 5 阶段：32->16->8->4->2（通道翻倍前保持 32，便于残差）
NUM_CLASSES = 10


class ConvBlock(nn.Module):
    """基础卷积单元：Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Stage(nn.Module):
    """一个卷积阶段：两组 Conv-BN-ReLU + 可选残差/softmax + MaxPool 下采样。

    forward 顺序（对应 PID 类比）：
      1. 主卷积通路 conv_block(x)
      2. 残差相加（"微分"通路）：stage_out + proj(x)，在卷积之后、MaxPool 之前
      3. 层间 softmax（"积分"项）：对通道维（dim=1）归一化，同样在池化之前
      4. MaxPool2d(2) 下采样
    """

    def __init__(self, in_channels: int, out_channels: int,
                 use_residual: bool, use_softmax: bool):
        super().__init__()
        self.use_residual = use_residual
        self.use_softmax = use_softmax

        # 主通路：每阶段两组 (Conv3x3 -> BN -> ReLU)
        self.conv_block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

        # 残差通路：输入输出通道数不一致时，用 1x1 Conv(+BN) 投影对齐；
        # stride=1 即可，下采样由阶段末尾的 MaxPool 完成
        if use_residual and in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = None

        # 层间 softmax：作用于通道维，插入在残差相加后、池化前
        self.softmax = nn.Softmax(dim=1) if use_softmax else None

        # 阶段末尾下采样
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv_block(x)
        if self.use_residual:
            identity = self.proj(x) if self.proj is not None else x
            out = out + identity  # 残差相加：卷积之后、MaxPool 之前
        if self.use_softmax:
            out = self.softmax(out)
        return self.pool(out)


class PidAblationCNN(nn.Module):
    """统一 CNN 骨干，2x2 因子对照试验模型。

    参数:
        use_residual: 是否启用残差连接（类比 PID 微分项）
        use_softmax:  是否启用层间 softmax（类比 PID 积分项）

    结构：3 个阶段（32 -> 16 -> 8 -> 4 空间尺寸），分类头
          AdaptiveAvgPool2d(1) -> Flatten -> Linear(128, 10)
    """

    def __init__(self, use_residual: bool = False, use_softmax: bool = False,
                 num_stages: int = 3):
        super().__init__()
        self.use_residual = use_residual
        self.use_softmax = use_softmax

        channels = STAGE_CHANNELS_DEEP if num_stages == 5 else STAGE_CHANNELS
        self.stages = nn.ModuleList([
            Stage(channels[i], channels[i + 1], use_residual, use_softmax)
            for i in range(len(channels) - 1)
        ])

        # 分类头：全局平均池化 -> 展平 -> 全连接
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], NUM_CLASSES),
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


def build_model(variant: str, num_stages: int = 3) -> PidAblationCNN:
    """工厂函数：variant -> (use_residual, use_softmax)。

    variant 取值：
        baseline -> (False, False)
        residual -> (True,  False)
        softmax  -> (False, True)
        both     -> (True,  True)

    num_stages: 3 或 5，选择骨干深度
    """
    mapping = {
        "baseline": (False, False),
        "residual": (True, False),
        "softmax": (False, True),
        "both": (True, True),
    }
    if variant not in mapping:
        raise ValueError(f"未知变体: {variant!r}，可选: {sorted(mapping)}")
    if num_stages not in (3, 5):
        raise ValueError(f"num_stages 仅支持 3 或 5，实际: {num_stages}")
    use_residual, use_softmax = mapping[variant]
    return PidAblationCNN(use_residual=use_residual, use_softmax=use_softmax,
                          num_stages=num_stages)


if __name__ == "__main__":
    # 冒烟测试：4 个变体 x (3/5 阶段) 各前向一次，校验输出形状；softmax 变体额外校验通道和
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)

    for num_stages in (3, 5):
        for variant in ("baseline", "residual", "softmax", "both"):
            model = build_model(variant, num_stages=num_stages)
            model.eval()
            with torch.no_grad():
                y = model(x)
            assert y.shape == (1, 10), \
                f"{variant}(s={num_stages}): 期望输出形状 (1, 10)，实际 {tuple(y.shape)}"
            print(f"stages={num_stages} variant={variant:<8} 输出形状={tuple(y.shape)}")

            # 对启用 softmax 的变体：手动分步执行第一个 stage 的模块序列，
            # 校验 softmax 后（池化前）特征在通道维求和近似为 1
            if model.use_softmax:
                stage = model.stages[0]
                with torch.no_grad():
                    feats = stage.conv_block(x)
                    if stage.use_residual:
                        feats = feats + (stage.proj(x) if stage.proj is not None else x)
                    feats = stage.softmax(feats)  # softmax 之后、池化之前
                sums = feats.sum(dim=1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
                    f"{variant}: 通道和偏离 1，实际范围 [{sums.min():.6f}, {sums.max():.6f}]"
                print(f"  校验: 阶段1 softmax 后通道和 = {sums.mean().item():.6f} (期望 ≈ 1)")

    print("冒烟测试全部通过")
