import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
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


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_in, d_out, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(d_in=d_in, d_out=d_out, block_size=block_size, dropout=dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(num_heads * d_out, num_heads * d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

