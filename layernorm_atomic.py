import torch
import torch.nn as nn


class LayerNormAtomic(nn.Module):
    INPUT_SHAPE = (1, 28, 256)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'seq_len'}
    STATIC_AXES = {2: 'embed_dim'}
    
    def __init__(self):
        super().__init__()
        self.layernorm = nn.LayerNorm(256)
    
    def forward(self, x):
        return self.layernorm(x)


if __name__ == "__main__":
    model = LayerNormAtomic()
    x = torch.randn(1, 28, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
