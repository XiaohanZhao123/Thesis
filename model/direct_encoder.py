import torch
from torch import nn


class DirecEncoder(nn.Module):
    def __init__(self, T: int):
        super(DirecEncoder, self).__init__()
        self.T = T

    def forward(self, x):
        x = torch.stack([x] * self.T, dim=0)
        return x
    
    
if __name__ == '__main__':
    encoder = DirecEncoder(T=1000)
    x = torch.randn(1, 784)
    spike_data = encoder(x)
    print(spike_data.shape)