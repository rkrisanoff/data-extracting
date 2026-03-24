import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 8, 128)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'sequence_length'}
    STATIC_AXES = {2: 'embedding_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.rnn_0 = nn.RNN(batch_first=True, input_size=128, hidden_size=32)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0, _ = self.rnn_0(tensor)
        return x0
