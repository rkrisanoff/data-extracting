import torch
import torch.nn as nn


class GruAtomic(nn.Module):
    INPUT_SHAPE = (1, 7, 128)
    DYNAMIC_AXES = {1: 'seq_len'}
    STATIC_AXES = {0: 'batch_size', 2: 'features'}
    
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 256, batch_first=True)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 256, device=x.device)
        out, _ = self.gru(x, h0)
        return out


if __name__ == "__main__":
    model = GruAtomic()
    x = torch.randn(1, 28, 128)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
