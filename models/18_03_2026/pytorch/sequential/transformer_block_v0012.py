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
        self.linear_0 = nn.Linear(in_features=64, out_features=256)
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(in_features=256, out_features=64)
        self.layernorm_1 = nn.LayerNorm(normalized_shape=[64])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0, _ = self.multiheadattention_0(tensor, tensor, tensor)
        x1 = x0 + tensor
        x2 = self.layernorm_0(x1)
        x3 = self.linear_0(x2)
        x4 = self.relu_0(x3)
        x5 = self.linear_1(x4)
        x6 = x5 + x2
        x7 = self.layernorm_1(x6)
        return x7
