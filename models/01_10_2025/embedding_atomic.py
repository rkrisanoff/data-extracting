import torch
import torch.nn as nn


class EmbeddingAtomic(nn.Module):
    INPUT_SHAPE = (1, 10)
    DYNAMIC_AXES = {0: 'batch_size', 1: 'seq_len'}
    STATIC_AXES = {}
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)
    
    def forward(self, x):
        # Embedding ожидает целочисленные индексы
        x = x.long()
        return self.embedding(x)


if __name__ == "__main__":
    model = EmbeddingAtomic()
    x = torch.randint(0, 1000, (1, 10))
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
