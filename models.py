"""对照试验模型定义（残差连接 x 层间积分机制，CIFAR-10）。

PID 类比假设的演进：
  第一轮实验结论：层间 softmax 并非"积分"——它是无记忆的通道维竞争归一化，
  会抹掉幅值并压制梯度。真正的积分应当是对历史状态按时间（层深度）累积。

本轮新增三种"积分"实现，与既有残差（微分）组合成完整 PID 对照：
  方案 A（正统积分）：跨阶段泄漏积分器 S ← α·S + (1−α)·stage_out
      - integral      : 仅积分器
      - pid_full      : 残差微分 + 积分器 + 线性比例（权重可学习）
  方案 B（重解读）  : 残差本身就是积分（Neural ODE 视角），对应已有 residual 变体
  方案 C（求和归一化）: LRN 式 x/(k+mean(x²))^β，保留"求和参与"语义且不破坏幅值
      - sumnorm          : 仅求和归一化
      - sumnorm_residual : 求和归一化 + 残差

变体列表：
  baseline / residual / softmax / both          （第一轮 2x2 因子）
  integral / pid_full / sumnorm / sumnorm_residual （第二轮积分修正）
"""
import torch
import torch.nn as nn

# 各阶段通道数：3 阶段 / 5 阶段 / 8 阶段三种骨干
STAGE_CHANNELS = (3, 32, 64, 128)          # 3 阶段：32->16->8->4（每阶段池化）
STAGE_CHANNELS_DEEP = (3, 32, 32, 64, 128, 256)  # 5 阶段：32->16->8->4->2（通道翻倍前保持 32，便于残差）
# 8 阶段：前 3 阶段不下采样（32x32 空间、通道恒定 32），后 5 阶段池化：32->16->16->8->4
# 这样空间只池化 5 次（与 5 阶段一致），避免 32x32 图像被池化到 0
STAGE_CHANNELS_DEEP8 = (3, 32, 32, 32, 64, 64, 128, 256, 256)
# 8 阶段各阶段是否在末尾池化（False 表示该阶段保持空间尺寸）
STAGE_POOL_DEEP8 = (False, False, False, True, True, True, True, True)
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


class SumNorm(nn.Module):
    """方案 C：LRN 式求和归一化。

    x_out = x / (k + mean_c(x^2))^β，沿通道维先求和（积分语义）再归一化，
    保留幅值信息，不做 exp 竞争。k、β 为可学习参数。
    """

    def __init__(self, k_init: float = 1.0, beta_init: float = 0.75):
        super().__init__()
        # 偏置 k 保证小响应处分母不为零；β 控制归一化强度
        self.k = nn.Parameter(torch.tensor(float(k_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        # 通道维均方（求和语义的归一化形式）
        ms = x.pow(2).mean(dim=1, keepdim=True)          # (B,1,H,W)
        return x / (self.k + ms).pow(self.beta.abs() + 1e-6)


class Stage(nn.Module):
    """一个卷积阶段：两组 Conv-BN-ReLU + 可选残差/积分机制 + MaxPool 下采样。

    forward 顺序：
      1. 主卷积通路 conv_block(x)
      2. 残差相加（"微分"通路）：stage_out + proj(x)，在卷积之后、MaxPool 之前
      3. 积分/归一化机制（softmax 或 sumnorm），在池化之前
      4. MaxPool2d(2) 下采样（use_pool=False 时跳过）
    """

    def __init__(self, in_channels: int, out_channels: int,
                 use_residual: bool, use_softmax: bool, use_pool: bool = True,
                 use_sumnorm: bool = False):
        super().__init__()
        self.use_residual = use_residual
        self.use_softmax = use_softmax
        self.use_sumnorm = use_sumnorm

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

        # 第一轮：层间 softmax（通道维竞争归一化，实验证明非积分）
        self.softmax = nn.Softmax(dim=1) if use_softmax else None
        # 方案 C：求和归一化
        self.sumnorm = SumNorm() if use_sumnorm else None

        # 阶段末尾下采样（use_pool=False 的阶段保持空间尺寸，用于 8 阶段骨干浅层）
        self.pool = nn.MaxPool2d(2) if use_pool else None

    def forward(self, x):
        out = self.conv_block(x)
        if self.use_residual:
            identity = self.proj(x) if self.proj is not None else x
            out = out + identity  # 残差相加：卷积之后、MaxPool 之前
        if self.use_softmax:
            out = self.softmax(out)
        if self.use_sumnorm:
            out = self.sumnorm(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class LeakyIntegrator(nn.Module):
    """方案 A：跨阶段泄漏积分器。

    每个阶段输出后更新累积状态：
        S ← α·S + (1−α)·stage_out
    α 为逐通道可学习泄漏系数（经 sigmoid 约束在 (0,1)）。
    状态 S 与当前阶段输出同宽；传入下一阶段前由外部 bridge 投影对齐。
    """

    def __init__(self, width: int):
        super().__init__()
        # 泄漏系数：logit 参数经 sigmoid 得 alpha，初始化 0.5（logit=0）
        self.alpha_logit = nn.Parameter(torch.zeros(1, width, 1, 1))

    def forward(self, s, feat):
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * s + (1.0 - alpha) * feat


class PidAblationCNN(nn.Module):
    """统一 CNN 骨干，因子对照试验模型。

    参数:
        use_residual: 残差连接（PID 微分项 / 方案 B 的积分读法）
        use_softmax:  层间 softmax（第一轮的"伪积分"对照）
        use_sumnorm:  求和归一化（方案 C）
        use_integral: 泄漏积分器（方案 A，跨阶段状态累积）
        use_pid_head: 分类头前注入累积状态（pid_full：微分+积分+比例完整 PID）
    """

    def __init__(self, use_residual: bool = False, use_softmax: bool = False,
                 num_stages: int = 3, use_sumnorm: bool = False,
                 use_integral: bool = False, use_pid_head: bool = False):
        super().__init__()
        self.use_residual = use_residual
        self.use_softmax = use_softmax
        self.use_sumnorm = use_sumnorm
        self.use_integral = use_integral
        self.use_pid_head = use_pid_head

        channels = STAGE_CHANNELS_DEEP8 if num_stages == 8 else \
            (STAGE_CHANNELS_DEEP if num_stages == 5 else STAGE_CHANNELS)
        # 8 阶段骨干：按 STAGE_POOL_DEEP8 决定各阶段是否池化；3/5 阶段全部池化
        pool_flags = STAGE_POOL_DEEP8 if num_stages == 8 else (True,) * (len(channels) - 1)
        self.stages = nn.ModuleList([
            Stage(channels[i], channels[i + 1], use_residual, use_softmax,
                  use_pool=pool_flags[i], use_sumnorm=use_sumnorm)
            for i in range(len(channels) - 1)
        ])

        # 方案 A：每个阶段一个泄漏积分器；跨阶段桥接投影（S 注入下一阶段前对齐通道/尺寸）
        if use_integral:
            self.integrators = nn.ModuleList([
                LeakyIntegrator(channels[i + 1])
                for i in range(len(channels) - 1)
            ])
            # 桥接：积分更新前，把 S 从上一阶段宽度 channels[i] 投影到当前阶段输出宽度
            # channels[i+1]（1x1 conv + BN）；注入下一阶段时 S 已与其输入同宽，无需再投影
            self.bridges = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels[i + 1]),
                )
                for i in range(1, len(channels) - 1)
            ])
            # 积分状态注入下一阶段的门控（gated add），初始门接近 0 以稳定训练
            self.inj_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(len(channels) - 2)
            ])

        # 分类头：全局平均池化 -> 展平 -> 全连接
        head_in = channels[-1] * (2 if use_pid_head else 1)  # pid_full：拼接累积状态
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(head_in, NUM_CLASSES),
        )

    def forward(self, x):
        if self.use_integral:
            s = None  # 跨阶段累积状态
            for i, stage in enumerate(self.stages):
                # 注入：把上一阶段的累积状态门控加到本阶段输入（S 与输入天然同宽同尺寸）
                if s is not None:
                    x = x + torch.sigmoid(self.inj_gates[i - 1]) * s
                # 主通路前向
                x = stage(x)
                # 更新累积状态：S ← α·S + (1−α)·stage_out（首阶段 S 从零开始）
                if s is not None:
                    # 桥接：把 S 从上一阶段宽度投影到当前阶段输出宽度，并做空间对齐
                    s_proj = self.bridges[i - 1](s)
                    if s_proj.shape[-2:] != x.shape[-2:]:
                        s_proj = nn.functional.adaptive_avg_pool2d(
                            s_proj, x.shape[-2:])
                    s = self.integrators[i](s_proj, x)
                else:
                    s = self.integrators[i](torch.zeros_like(x), x)
            if self.use_pid_head:
                # 完整 PID：比例（主通路特征）+ 积分（累积状态）拼接后线性映射
                x = torch.cat([x, s], dim=1)
            return self.head(x)

        for stage in self.stages:
            x = stage(x)
        return self.head(x)


# 变体 -> 模型开关参数的映射（工厂函数使用）
VARIANT_MAPPING = {
    # ---- 第一轮 2x2 因子 ----
    "baseline":         dict(use_residual=False, use_softmax=False),
    "residual":         dict(use_residual=True,  use_softmax=False),
    "softmax":          dict(use_residual=False, use_softmax=True),
    "both":             dict(use_residual=True,  use_softmax=True),
    # ---- 第二轮积分修正 ----
    # 方案 A：泄漏积分器（仅积分）
    "integral":         dict(use_integral=True),
    # 方案 A 完整 PID：残差微分 + 泄漏积分 + 比例拼接头
    "pid_full":         dict(use_residual=True, use_integral=True, use_pid_head=True),
    # 方案 C：求和归一化
    "sumnorm":          dict(use_sumnorm=True),
    # 方案 C + 残差
    "sumnorm_residual": dict(use_residual=True, use_sumnorm=True),
}


def build_model(variant: str, num_stages: int = 3) -> PidAblationCNN:
    """工厂函数：variant -> PidAblationCNN。

    num_stages: 3 / 5 / 8，选择骨干深度
    """
    if variant not in VARIANT_MAPPING:
        raise ValueError(f"未知变体: {variant!r}，可选: {sorted(VARIANT_MAPPING)}")
    if num_stages not in (3, 5, 8):
        raise ValueError(f"num_stages 仅支持 3 / 5 / 8，实际: {num_stages}")
    return PidAblationCNN(num_stages=num_stages, **VARIANT_MAPPING[variant])


if __name__ == "__main__":
    # 冒烟测试：全部变体 x (3/5/8 阶段) 各前向一次，校验输出形状；
    # softmax 变体校验通道和；integral 变体校验泄漏系数范围
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)

    for num_stages in (3, 5, 8):
        for variant in VARIANT_MAPPING:
            model = build_model(variant, num_stages=num_stages)
            model.eval()
            with torch.no_grad():
                y = model(x)
            assert y.shape == (2, 10), \
                f"{variant}(s={num_stages}): 期望输出形状 (2, 10)，实际 {tuple(y.shape)}"
            print(f"stages={num_stages} variant={variant:<17} 输出形状={tuple(y.shape)}")

            # softmax 变体：校验 softmax 后（池化前）特征通道和近似为 1
            if model.use_softmax:
                stage = model.stages[0]
                with torch.no_grad():
                    feats = stage.conv_block(x)
                    if stage.use_residual:
                        feats = feats + (stage.proj(x) if stage.proj is not None else x)
                    feats = stage.softmax(feats)
                sums = feats.sum(dim=1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
                    f"{variant}: 通道和偏离 1，实际范围 [{sums.min():.6f}, {sums.max():.6f}]"
                print(f"  校验: 阶段1 softmax 后通道和 = {sums.mean().item():.6f} (期望 ≈ 1)")

            # integral 变体：校验泄漏系数逐通道均在 (0,1)
            if model.use_integral:
                for integ in model.integrators:
                    a = torch.sigmoid(integ.alpha_logit)
                    assert (a > 0.0).all() and (a < 1.0).all(), \
                        f"泄漏系数越界: min={a.min().item()}, max={a.max().item()}"

    print("冒烟测试全部通过")
