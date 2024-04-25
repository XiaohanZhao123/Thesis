import torch
from torch import nn
from spikingjelly.activation_based import neuron, functional


class ShallowEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, T) -> None:
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            16, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * T,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * T)
        self.act3 = neuron.GatedLIFNode(T=T)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # reshape x, from (N, C*T, H, W) to (N, T, C, H, W)
        x = x.split(x.size(1) // self.T, dim=1)
        x = torch.stack(x, dim=0)
        x = self.act3(x)

        return x


class ShallowEncoderRepat(nn.Module):
    def __init__(self, in_channels, out_channels, T) -> None:
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            16, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = neuron.GatedLIFNode(T=T, inplane=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.stack([x] * self.T, dim=0)
        x = self.act2(x)

        return x


class ShallowEncoderGLIF(nn.Module):
    def __init__(self, in_channels, out_channels, T) -> None:
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            16, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = neuron.GatedLIFNode(T=T)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.stack([x] * self.T, dim=0)
        x = self.act2(x)

        return x


class LSTMEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, T, img_size) -> None:
        super().__init__()
        self.T = T
        self.img_size = img_size
        self.out_channels = out_channels
        self.lstm = nn.LSTM(
            in_channels * img_size * img_size,
            512,
            batch_first=True,
        )
        self.fc = nn.Linear(512, out_channels * img_size * img_size)
        self.act = neuron.LIFNode()

    def forward(self, x):
        # reshape x,
        # from (N, C, H, W) to (N, 1, C*H*W)
        x = x.view(x.size(0), 1, -1)
        # repeat x in time dimension
        x = torch.repeat_interleave(x, self.T, dim=1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        x = self.act(x)
        x = x.reshape(
            x.size(0), x.size(1), self.out_channels, self.img_size, self.img_size
        )

        return x


if __name__ == "__main__":
    # test shallow encoder
    encoder = ShallowEncoderRepat(in_channels=3, out_channels=8, T=10)
    x = torch.randn(2, 3, 32, 32)
    spike_data = encoder(x)
    print(spike_data.shape)
