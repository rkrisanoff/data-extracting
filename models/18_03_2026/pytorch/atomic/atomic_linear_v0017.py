import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 32)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'feature_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.linear_0 = nn.Linear(in_features=32, out_features=32)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = self.linear_0(tensor)
        return x0
