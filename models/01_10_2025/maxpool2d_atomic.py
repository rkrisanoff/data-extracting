import torch
import torch.nn as nn


class MaxPool2dAtomic(nn.Module):
    INPUT_SHAPE = (1, 16, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.pool(x)


if __name__ == "__main__":
    model = MaxPool2dAtomic()
    x = torch.randn(1, 16, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


