import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 128)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'feature_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.softmax_0 = nn.Softmax(dim=-1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = self.softmax_0(tensor)
        return x0
