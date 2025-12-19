import torch

from attention_condition_generator.conditioner import AttentionConditioner


def test_batch_domain_ids():
    B, D = 4, 64
    num_domains = 3

    user_emb = torch.randn(B, D)
    domain_id = torch.tensor([0, 1, 2, 1])

    conditioner = AttentionConditioner(
        emb_dim=D,
        num_heads=4,
        num_domains=num_domains,
    )

    condition = conditioner(user_emb, domain_id)

    assert condition.shape == (B, D)


def test_single_domain_broadcast():
    B, D = 8, 128
    num_domains = 2

    user_emb = torch.randn(B, D)
    domain_id = 1  # single domain

    conditioner = AttentionConditioner(
        emb_dim=D,
        num_heads=8,
        num_domains=num_domains,
    )

    condition = conditioner(user_emb, domain_id)

    assert condition.shape == (B, D)


def test_backward_pass():
    B, D = 2, 32
    num_domains = 2

    user_emb = torch.randn(B, D, requires_grad=True)
    domain_id = torch.tensor([0, 1])

    conditioner = AttentionConditioner(
        emb_dim=D,
        num_heads=4,
        num_domains=num_domains,
    )

    condition = conditioner(user_emb, domain_id)
    loss = condition.mean()
    loss.backward()

    assert user_emb.grad is not None
