import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 16)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'feature_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.tanh_0 = nn.Tanh()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = self.tanh_0(tensor)
        return x0
