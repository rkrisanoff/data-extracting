import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 512)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'feature_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.linear_0 = nn.Linear(in_features=512, out_features=512)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = self.linear_0(tensor)
        return x0
