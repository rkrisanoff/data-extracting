import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 32, 64)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'sequence_length'}
    STATIC_AXES = {2: 'embedding_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.multiheadattention_0 = nn.MultiheadAttention(batch_first=True, embed_dim=64, num_heads=4)
        self.layernorm_0 = nn.LayerNorm(normalized_shape=[64])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0, _ = self.multiheadattention_0(tensor, tensor, tensor)
        x1 = self.layernorm_0(x0)
        return x1
