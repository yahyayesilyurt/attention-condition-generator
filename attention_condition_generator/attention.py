import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, query, key, value, return_attn: bool = False):
        """
        query: (B, 1, D)
        key/value: (B, 1, D)
        """

        attn_out, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            need_weights=return_attn,
            average_attn_weights=False
        )

        out = self.norm(query + self.dropout(attn_out))

        if return_attn:
            return out, attn_weights

        return out
