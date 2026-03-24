import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 2048)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'feature_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.sigmoid_0 = nn.Sigmoid()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = self.sigmoid_0(tensor)
        return x0
