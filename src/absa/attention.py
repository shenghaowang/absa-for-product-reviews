# The implementation of the attention layer is taken from
# Tunstall, L., Werra, L. von, &amp; Wolf, T. (2022).
# Natural language processing with transformers, Revised edition.
# O'Reilly Media, Inc.

from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Calculate attention scores using the dot product"""
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)

    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        """Single attention head projects token embeddings
        into query, key, value and generate attention scores.
        Attention heads work in a similar way to the filters
        in CNN.

        Parameters
        ----------
        embed_dim : int
            number of embedding dimensions of the tokens
        head_dim : int
            number of dimensions to project into

        """
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )

        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """Implement multi-head attention layer

        Parameters
        ----------
        embed_dim : int
            embedding dimension of the tokens
        num_heads : int
            number of attention heads
        """
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)[:, -1, :]

        return x
