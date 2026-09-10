# implementing attention mechanism 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout=0.1):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.Q = nn.Linear(d_in, d_out, bias=False)
        self.K = nn.Linear(d_in, d_out, bias=False)
        self.V = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  

        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)

        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores = scores / (self.d_out ** 0.5)

        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        attention = F.softmax(scores, dim=2)
        attention = self.dropout(attention)

        hidden_states = torch.bmm(attention, values)
        return hidden_states

