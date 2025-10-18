import torch
import torch.nn as nn


class LinearParallel(nn.Module):
    INPUT_SHAPE = (1, 784)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'features'}
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        
        self.branch_a = nn.Linear(256, 128)
        self.branch_b = nn.Linear(256, 128)
        self.relu_a = nn.ReLU()
        self.relu_b = nn.ReLU()
        
        self.linear_out = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        
        x1, x2 = torch.split(x, 256, dim=1)
        
        branch_a = self.relu_a(self.branch_a(x1))
        branch_b = self.relu_b(self.branch_b(x2))
        
        x = torch.cat([branch_a, branch_b], dim=1)
        x = self.linear_out(x)
        return x


if __name__ == "__main__":
    model = LinearParallel()
    x = torch.randn(1, 784)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


