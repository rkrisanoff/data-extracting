import torch
import torch.nn as nn


class LinearParallelTriple(nn.Module):
    INPUT_SHAPE = (1, 784)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'features'}
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        
        self.branch_a = nn.Linear(170, 128)
        self.branch_b = nn.Linear(171, 128)
        self.branch_c = nn.Linear(171, 128)
        self.relu_a = nn.ReLU()
        self.relu_b = nn.ReLU()
        self.relu_c = nn.ReLU()
        
        self.linear_out = nn.Linear(384, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        
        x1, x2, x3 = torch.split(x, [170, 171, 171], dim=1)
        
        branch_a = self.relu_a(self.branch_a(x1))
        branch_b = self.relu_b(self.branch_b(x2))
        branch_c = self.relu_c(self.branch_c(x3))
        
        x = torch.cat([branch_a, branch_b, branch_c], dim=1)
        x = self.linear_out(x)
        return x


if __name__ == "__main__":
    model = LinearParallelTriple()
    x = torch.randn(1, 784)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


