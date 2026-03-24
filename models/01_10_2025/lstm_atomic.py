import torch
import torch.nn as nn


class LstmAtomic(nn.Module):
    INPUT_SHAPE = (1, 7, 128)
    DYNAMIC_AXES = {1: 'seq_len'}
    STATIC_AXES = {0: 'batch_size', 2: 'features'}
    
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, batch_first=True)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 256, device=x.device)
        c0 = torch.zeros(1, x.size(0), 256, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out


if __name__ == "__main__":
    model = LstmAtomic()
    x = torch.randn(1, 28, 128)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
