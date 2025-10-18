import torch
import torch.nn as nn


class Conv2dPyramid(nn.Module):
    INPUT_SHAPE = (1, 1, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 28 * 28, 10)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = Conv2dPyramid()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


