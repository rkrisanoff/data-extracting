import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 8, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'channels', 2: 'height', 3: 'width'}

    def __init__(self) -> None:
        super().__init__()


    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = tensor
        x1 = x0 + tensor
        return x1
