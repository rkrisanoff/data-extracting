import torch
import torch.nn as nn


class LinearBottleneck(nn.Module):
    INPUT_SHAPE = (1, 784)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'features'}
    
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(784, 128)
        self.expand = nn.Linear(128, 784)
    
    def forward(self, x):
        x = self.compress(x)
        x = self.expand(x)
        return x


if __name__ == "__main__":
    model = LinearBottleneck()
    x = torch.randn(1, 784)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


