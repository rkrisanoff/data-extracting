import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 32, 128)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'sequence_length'}
    STATIC_AXES = {2: 'embedding_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.multiheadattention_0 = nn.MultiheadAttention(batch_first=True, embed_dim=128, num_heads=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0, _ = self.multiheadattention_0(tensor, tensor, tensor)
        return x0
