import torch
import torch.nn as nn


class Conv2dBottleneck(nn.Module):
    INPUT_SHAPE = (1, 64, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.compress = nn.Conv2d(64, 16, kernel_size=1)
        self.expand = nn.Conv2d(16, 64, kernel_size=1)
    
    def forward(self, x):
        x = self.compress(x)
        x = self.expand(x)
        return x


if __name__ == "__main__":
    model = Conv2dBottleneck()
    x = torch.randn(1, 64, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


