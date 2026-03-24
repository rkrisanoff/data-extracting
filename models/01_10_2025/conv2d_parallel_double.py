import torch
import torch.nn as nn


class Conv2dParallelDouble(nn.Module):
    INPUT_SHAPE = (1, 1, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size', 2: 'height', 3: 'width'}
    STATIC_AXES = {1: 'channels'}
    
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv_c = nn.Conv2d(1, 16, kernel_size=7, padding=3)
        self.linear = nn.Linear(48 * 28 * 28, 10)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        out_a = self.conv_a(x)
        out_b = self.conv_b(x)
        out_c = self.conv_c(x)
        
        out = torch.cat([out_a, out_b, out_c], dim=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    model = Conv2dParallelDouble()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


