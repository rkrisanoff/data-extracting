import torch
import torch.nn as nn


class Conv2dResidual(nn.Module):
    INPUT_SHAPE = (1, 16, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x + residual
        x = self.relu(x)
        return x


if __name__ == "__main__":
    model = Conv2dResidual()
    x = torch.randn(1, 16, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


