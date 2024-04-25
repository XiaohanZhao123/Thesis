import torch
from torch import nn
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import neuron


class ShallowDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, T) -> None:
        super().__init__()
        # conv over the time dimension, making input (N, T, C, H, W) to (N, 1, C, H, W)
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=(T, 1, 1), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 3, kernel_size=out_channels, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.act2 = nn.Tanh()
        
    def forward(self, x):
        x = x.permute(1, 2, 0, 3, 4) # change the shape from (T, N, C, H, W) to (N, C, T, H, W)
        x = self.conv1(x)
        # change the shape from (N, 16, 1, H, W) to (N, 16, H, W)
        x = x.squeeze(2)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        return x
    
if __name__ == '__main__':
    # test 
    decoder = ShallowDecoder(in_channels=16, out_channels=3, T=10)
    x = torch.randn(10, 2, 16, 32, 32)
    spike_data = decoder(x)
    print(spike_data.shape)
    
        
        