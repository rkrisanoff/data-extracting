import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SHAPE = (1, 32, 256)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'sequence_length'}
    STATIC_AXES = {2: 'embedding_dim'}

    def __init__(self) -> None:
        super().__init__()
        self.multiheadattention_0 = nn.MultiheadAttention(batch_first=True, embed_dim=256, num_heads=8)
        self.layernorm_0 = nn.LayerNorm(normalized_shape=[256])
        self.multiheadattention_1 = nn.MultiheadAttention(batch_first=True, embed_dim=256, num_heads=8)
        self.layernorm_1 = nn.LayerNorm(normalized_shape=[256])
        self.linear_0 = nn.Linear(in_features=256, out_features=128)
        self.sigmoid_0 = nn.Sigmoid()
        self.linear_1 = nn.Linear(in_features=128, out_features=256)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=[256])
        self.relu_0 = nn.ReLU()
        self.layernorm_3 = nn.LayerNorm(normalized_shape=[256])
        self.softmax_0 = nn.Softmax(dim=-1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x0, _ = self.multiheadattention_0(tensor, tensor, tensor)
        x1 = self.layernorm_0(x0)
        x2, _ = self.multiheadattention_1(x1, x1, x1)
        x3 = self.layernorm_1(x2)
        x4 = self.linear_0(x3)
        x5 = self.sigmoid_0(x4)
        x6 = self.linear_1(x5)
        x7 = self.layernorm_2(x6)
        x8 = self.relu_0(x7)
        x9 = self.layernorm_3(x8)
        x10 = self.softmax_0(x9)
        return x10
