import torch
import torch.nn as nn
import time


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Focus, self).__init__()
        self.conv = Conv(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(torch.concat((x[:, :, ::2, ::2], x[:, :, ::2, 1::2],
                                       x[:, :, 1::2, ::2], x[:, :, 1::2, 1::2]), dim=1))


# 残差层，张量形状输入与输出相同
class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels):
        super(ResBlock, self).__init__()
        self.conv1 = Conv(channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(hidden_channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        output = self.conv2(self.conv1(x))
        return residual + output


# 卷积核全部都是1*1，用来整合特征
class CSPNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_resBlock):
        super(CSPNet, self).__init__()
        # 残差边，通道数减半
        self.conv1 = Conv(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        # 残差块堆叠前卷积，通道数减半
        self.conv2 = Conv(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        # 最后整合特征卷积层
        self.conv3 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # 残差块堆叠
        self.res_block = nn.Sequential(*[ResBlock(in_channels // 2, in_channels) for _ in range(num_resBlock)])

    def forward(self, x):
        return self.conv3(torch.concat((self.conv1(x), self.res_block(self.conv2(x))), dim=1))


class SPPNet_Paralleling(nn.Module):
    def __init__(self, channels, kernel=[5, 9, 13]):
        super(SPPNet_Paralleling, self).__init__()
        self.conv = Conv(channels, channels, kernel_size=3, stride=1, padding=1)
        self.maxPool = nn.Sequential(*[nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel])
        self.conv_combine = Conv(channels * 4, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv_combine(torch.concat([x] + [m(x) for m in self.maxPool], dim=1))


class SPPNet_Series(nn.Module):
    def __init__(self, channels, kernel=[5, 9, 13]):
        super(SPPNet_Series, self).__init__()
        self.conv = Conv(channels, channels, kernel_size=3, stride=1, padding=1)
        self.maxPool1 = nn.MaxPool2d(kernel_size=kernel[0], stride=1, padding=kernel[0] // 2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=kernel[1], stride=1, padding=kernel[1] // 2)
        self.maxPool3 = nn.MaxPool2d(kernel_size=kernel[2], stride=1, padding=kernel[2] // 2)
        self.conv_combine = Conv(channels * 4, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        output1 = self.conv(x)
        output2 = self.maxPool1(output1)
        output3 = self.maxPool2(output2)
        output4 = self.maxPool3(output3)
        return self.conv_combine(torch.concat([output1, output2, output3, output4], dim=1))


class CSPDarkNet(nn.Module):
    def __init__(self, base_depth=3, base_channels=64):
        super(CSPDarkNet, self).__init__()
        self.Focus = Focus(in_channels=3, out_channels=base_channels)
        self.dark1 = nn.Sequential(
            # 下采样
            Conv(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            CSPNet(base_channels * 2, base_channels * 2, base_depth)
        )
        self.dark2 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            CSPNet(base_channels * 4, base_channels * 4, base_depth * 3)
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            CSPNet(base_channels * 8, base_channels * 8, base_depth * 3)
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1),
            SPPNet_Paralleling(channels=base_channels * 16),
            CSPNet(base_channels * 16, base_channels * 16, base_depth * 3)
        )

    def forward(self, x):
        output1 = self.dark2(self.dark1(self.Focus(x)))
        output2 = self.dark3(output1)
        output3 = self.dark4(output2)
        return output1, output2, output3


if __name__ == "__main__":
    img = torch.randn(1, 3, 640, 640)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CSPDarkNet()
    model = model.to(device)
    img = img.to(device)
