import torch
import torch.nn as nn


class SigmoidAtomic(nn.Module):
    INPUT_SHAPE = (1, 784)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'features'}
    
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(x)


if __name__ == "__main__":
    model = SigmoidAtomic()
    x = torch.randn(1, 784)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


