import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 1, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'channels', 2: 'height', 3: 'width'}

    def __init__(self) -> None:
        super().__init__()
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = self.maxpool2d_0(tensor)
        return x0
