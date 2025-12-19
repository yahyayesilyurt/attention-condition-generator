import torch
import torch.nn as nn
from .attention import CrossAttentionBlock
from .domain_embedding import DomainEmbedding


class AttentionConditioner(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_domains: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.domain_embedding = DomainEmbedding(num_domains, emb_dim)
        self.attention = CrossAttentionBlock(
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, user_emb, domain_id, return_attn: bool = False):
        """
        user_emb: (B, D)
        domain_id: int or (B,)
        """

        B, D = user_emb.shape

        domain_emb = self.domain_embedding(domain_id)

        if domain_emb.dim() == 2:
            domain_emb = domain_emb.unsqueeze(1)

        if domain_emb.size(0) == 1:
            domain_emb = domain_emb.expand(B, -1, -1)

        query = user_emb.unsqueeze(1)

        if return_attn:
            out, attn_weights = self.attention(
                query=query,
                key=domain_emb,
                value=domain_emb,
                return_attn=True
            )
            return out.squeeze(1), attn_weights

        out = self.attention(
            query=query,
            key=domain_emb,
            value=domain_emb
        )

        return out.squeeze(1)
