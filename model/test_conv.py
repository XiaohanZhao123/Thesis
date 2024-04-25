import torch.nn as nn
import torch

# Parameters
C = 3   # Number of input channels, example value
out_channels = 8  # Example value for number of output channels
T = 10  # Example value for the size of the time dimension
kH, kW = 3, 3  # Kernel size in height and width
sH, sW = 1, 1  # Stride in height and width
pH, pW = 1, 1  # Padding in height and width

# Convolutional layer
conv3d = nn.Conv3d(in_channels=C, out_channels=out_channels, 
                   kernel_size=(T, kH, kW), 
                   stride=(1, sH, sW), 
                   padding=(0, pH, pW))

# Example input tensor (N, T, C, H, W)
input_tensor = torch.randn(2, C, T, 32, 32)  # N=2, H=W=32 (example sizes)

# Apply convolution
output = conv3d(input_tensor)
print(output.shape)
