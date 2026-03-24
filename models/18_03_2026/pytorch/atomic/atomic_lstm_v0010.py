import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 32, 32)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'sequence_length'}
    STATIC_AXES = {2: 'embedding_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.lstm_0 = nn.LSTM(batch_first=True, input_size=32, hidden_size=64, num_layers=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0, _ = self.lstm_0(tensor)
        return x0
