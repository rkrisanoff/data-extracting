import torch
import torch.nn as nn


class SoftmaxAtomic(nn.Module):
    INPUT_SHAPE = (1, 10)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'features'}
    
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        return self.softmax(x)


if __name__ == "__main__":
    model = SoftmaxAtomic()
    x = torch.randn(1, 10)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


