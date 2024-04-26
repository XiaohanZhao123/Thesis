from spikingjelly.activation_based import layer, neuron, surrogate
from torch import nn
import torch

class SimpleNet(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs) -> None:
        super().__init__()
        surrogate_fn = surrogate.ATan()
        self.conv1 = layer.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(16)
        self.act1 = neuron.LIFNode(v_threshold=1.0, surrogate_function=surrogate_fn)
        self.pool1 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = layer.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(32)
        self.act2 = neuron.LIFNode(v_threshold=1.0, surrogate_function=surrogate_fn)
        self.pool2 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = layer.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = layer.BatchNorm2d(64)
        self.act3 = neuron.LIFNode(v_threshold=1.0, surrogate_function=surrogate_fn)
        self.pool3 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = layer.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = layer.BatchNorm2d(128)
        self.act4 = neuron.LIFNode(v_threshold=1.0, surrogate_function=surrogate_fn)
        self.pool4 = layer.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(in_features=128, out_features=num_classes, bias=True)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    

if __name__ == '__main__':
    model = SimpleNet(3, 10)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
        