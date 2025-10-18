import torch
import torch.nn as nn


class RnnSingle(nn.Module):
    INPUT_SHAPE = (1, 7, 28)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'seq_len'}
    STATIC_AXES = {2: 'features'}
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(28, 128)
        self.rnn = nn.RNN(128, 256, batch_first=True)
        self.linear = nn.Linear(256, 10)
    
    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.view(batch_size, 7, -1)
        
        x = self.embedding(x)
        
        h0 = torch.zeros(1, batch_size, 256, device=x.device)
        rnn_out, _ = self.rnn(x, h0)
        
        x = rnn_out[:, -1, :]
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = RnnSingle()
    x = torch.randn(1, 7, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
