import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 16, 32)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'sequence_length'}
    STATIC_AXES = {2: 'embedding_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.lstm_0 = nn.LSTM(batch_first=True, input_size=32, hidden_size=128, num_layers=2)
        self.linear_0 = nn.Linear(in_features=128, out_features=128)
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(in_features=128, out_features=128)
        self.layernorm_0 = nn.LayerNorm(normalized_shape=[128])
        self.sigmoid_0 = nn.Sigmoid()
        self.layernorm_1 = nn.LayerNorm(normalized_shape=[128])
        self.softmax_0 = nn.Softmax(dim=-1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0, _ = self.lstm_0(tensor)
        x1 = self.linear_0(x0)
        x2 = self.relu_0(x1)
        x3 = self.linear_1(x2)
        x4 = self.layernorm_0(x3)
        x5 = self.sigmoid_0(x4)
        x6 = self.layernorm_1(x5)
        x7 = self.softmax_0(x6)
        return x7
