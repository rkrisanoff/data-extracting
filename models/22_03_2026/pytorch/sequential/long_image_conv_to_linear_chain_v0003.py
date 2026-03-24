import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 1, 28, 28)
    DYNAMIC_AXES = {0: 'batch_size'}
    STATIC_AXES = {1: 'channels', 2: 'height', 3: 'width'}

    def __init__(self) -> None:
        super().__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu_0 = nn.ReLU()
        self.conv2d_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.sigmoid_0 = nn.Sigmoid()
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool2d_0 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=32)
        self.adaptiveavgpool2d_0 = nn.AdaptiveAvgPool2d(output_size=[1, 1])
        self.flatten_0 = nn.Flatten()
        self.linear_0 = nn.Linear(in_features=32, out_features=64)
        self.tanh_0 = nn.Tanh()
        self.linear_1 = nn.Linear(in_features=64, out_features=64)
        self.softmax_0 = nn.Softmax(dim=-1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = self.conv2d_0(tensor)
        x1 = self.relu_0(x0)
        x2 = self.conv2d_1(x1)
        x3 = self.sigmoid_0(x2)
        x4 = self.maxpool2d_0(x3)
        x5 = self.avgpool2d_0(x4)
        x6 = self.batchnorm2d_0(x5)
        x7 = self.adaptiveavgpool2d_0(x6)
        x8 = self.flatten_0(x7)
        x9 = self.linear_0(x8)
        x10 = self.tanh_0(x9)
        x11 = self.linear_1(x10)
        x12 = self.softmax_0(x11)
        return x12
