import torch
import torch.nn as nn


class BatchNorm2dAtomic(nn.Module):
    INPUT_SHAPE = (1, 16, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(16)
    
    def forward(self, x):
        return self.bn(x)


if __name__ == "__main__":
    model = BatchNorm2dAtomic()
    x = torch.randn(1, 16, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
