import torch
import torch.nn as nn


class Conv2dSequential(nn.Module):
    INPUT_SHAPE = (1, 1, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 28 * 28, 10)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = Conv2dSequential()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

