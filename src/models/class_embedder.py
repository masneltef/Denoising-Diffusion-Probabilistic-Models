import torch
import torch.nn as nn
import math

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, cond_drop_rate=0.1):
        super().__init__()
        
        # TODO: implement the class embeddering layer for CFG using nn.Embedding
        self.embedding = nn.Embedding(n_classes + 1, embed_dim)  # +1 for unconditional class
        self.cond_drop_rate = cond_drop_rate
        self.num_classes = n_classes

    def forward(self, x):
        b = x.shape[0]
        
        if self.cond_drop_rate > 0 and self.training:
            # TODO: implement class drop with unconditional class
            mask = torch.rand(b, device=x.device) < self.cond_drop_rate
            x = x.clone()
            x[mask] = self.num_classes  # unconditional class index
        
        # TODO: get embedding: N, embed_dim
        c = self.embedding(x)
        return c