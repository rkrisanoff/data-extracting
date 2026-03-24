import torch
import torch.nn as nn


class Conv2dParallel(nn.Module):
    INPUT_SHAPE = (1, 1, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        out_a = self.conv_a(x)
        out_b = self.conv_b(x)
        
        out = torch.cat([out_a, out_b], dim=1)
        return out


if __name__ == "__main__":
    model = Conv2dParallel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


