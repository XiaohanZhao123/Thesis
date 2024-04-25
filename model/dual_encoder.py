import torch
from torch import nn
from spikingjelly.activation_based import neuron


class ShallowEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, T) -> None:
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, out_channels * T, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * T)
        self.act2 = neuron.LIFNode()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # reshape x, from (N, C*T, H, W) to (N, T, C, H, W)
        x = x.split(x.size(1) // self.T, dim=1)
        x = torch.stack(x, dim=0)
        x = self.act2(x)
        
        return x
    
    
    
if __name__ == '__main__':
    # test 
    encoder = ShallowEncoder(in_channels=3, out_channels=8, T=10)
    x = torch.randn(2, 3, 32, 32)
    spike_data = encoder(x)
    print(spike_data.shape)