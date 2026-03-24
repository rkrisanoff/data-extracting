import torch
import torch.nn as nn


class AttentionSelf(nn.Module):
    INPUT_SHAPE = (1, 28, 256)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'seq_len'}
    STATIC_AXES = {2: 'embed_dim'}
    
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(256, num_heads=1, batch_first=True)
    
    def forward(self, x):
        out, _ = self.attention(x, x, x)
        return out


if __name__ == "__main__":
    model = AttentionSelf()
    x = torch.randn(1, 28, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")


