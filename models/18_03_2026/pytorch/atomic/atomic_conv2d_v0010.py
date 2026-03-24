import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 1, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'channels', 2: 'height', 3: 'width'}

    def __init__(self) -> None:
        super().__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = self.conv2d_0(tensor)
        return x0
