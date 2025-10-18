import torch
import torch.nn as nn


class LinearResidual(nn.Module):
    INPUT_SHAPE = (1, 784)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'features'}
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        
        residual = x
        x = self.linear2(x)
        x = x + residual
        x = self.relu2(x)
        
        x = self.linear3(x)
        return x


if __name__ == "__main__":
    model = LinearResidual()
    x = torch.randn(1, 784)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


