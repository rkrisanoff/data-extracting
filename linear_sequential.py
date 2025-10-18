import torch
import torch.nn as nn


class LinearSequential(nn.Module):
    INPUT_SHAPE = (1, 784)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'features'}
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


if __name__ == "__main__":
    model = LinearSequential()
    x = torch.randn(1, 784)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


