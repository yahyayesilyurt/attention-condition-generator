import torch
from attention_condition_generator.conditioner import AttentionConditioner


def test_attention_weight_logging():
    B, D = 4, 16

    model = AttentionConditioner(
        emb_dim=D,
        num_domains=2,
        num_heads=4
    )

    user_emb = torch.randn(B, D)
    domain_id = 1

    condition, attn_weights = model(
        user_emb,
        domain_id,
        return_attn=True
    )

    assert condition.shape == (B, D)
    assert attn_weights.shape == (B, 4, 1, 1)
