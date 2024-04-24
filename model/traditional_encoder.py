from snntorch import spikegen
from torch import nn
import torch

class RateEncoder(nn.Module):
    def __init__(self, T: int):
        super(RateEncoder, self).__init__()
        self.T = T
        
    def forward(self, x):
        x = _standarlize(x)
        spike_data = spikegen.rate(x, num_steps=self.T)
        return spike_data
    
    
def _standarlize(x):
    return (x - x.min()) / (x.max() - x.min())


class LatencyEncoder(nn.Module):
    def __init__(self, T, tau, threshold) -> None:
        super().__init__()
        self.T = T
        self.tau = tau
        self.threshold = threshold
        
    def forward(self, x):
        x = _standarlize(x)
        spike_data = spikegen.latency(x, num_steps=self.T, tau=self.tau, threshold=self.threshold, normalize=True)
        return spike_data
    
