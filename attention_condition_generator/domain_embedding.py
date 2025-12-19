import torch
import torch.nn as nn


class DomainEmbedding(nn.Module):
    """
    Learnable domain / global embeddings.

    Maps domain IDs to embedding vectors.
    """

    def __init__(self, num_domains: int, emb_dim: int):
        """
        Args:
            num_domains (int): Number of domains
            emb_dim (int): Embedding dimension
        """
        super().__init__()

        self.num_domains = num_domains
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(num_domains, emb_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init is standard and stable
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, domain_id):
        """
        Args:
            domain_id (int or LongTensor):
                - int: single domain
                - LongTensor of shape (B,)

        Returns:
            domain_emb (Tensor): (B, emb_dim)
        """

        if isinstance(domain_id, int):
            domain_id = torch.tensor([domain_id], dtype=torch.long, device=self.embedding.weight.device)

        elif isinstance(domain_id, torch.Tensor):
            domain_id = domain_id.long()

        else:
            raise TypeError("domain_id must be int or torch.LongTensor")

        domain_emb = self.embedding(domain_id)

        return domain_emb
