import torch
import torch.nn as nn


class MaxPool2dSingle(nn.Module):
    INPUT_SHAPE = (1, 16, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(16 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = MaxPool2dSingle()
    x = torch.randn(1, 16, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


