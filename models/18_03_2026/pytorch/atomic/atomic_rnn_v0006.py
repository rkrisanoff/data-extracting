import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 8, 64)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'sequence_length'}
    STATIC_AXES = {2: 'embedding_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.rnn_0 = nn.RNN(batch_first=True, input_size=64, hidden_size=128)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0, _ = self.rnn_0(tensor)
        return x0
