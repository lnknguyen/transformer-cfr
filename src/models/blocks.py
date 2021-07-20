import torch.nn as nn

class EmbeddingAdder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, dim=2):
        return x.sum(dim=dim)