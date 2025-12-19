import logging
import torch
from attention_condition_generator import AttentionConditioner

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

logger = logging.getLogger(__name__)

model = AttentionConditioner(
    emb_dim=32,
    num_domains=2
)

user_emb = torch.randn(8, 32)

condition, attn_weights = model(
    user_emb,
    domain_id=1,
    return_attn=True
)

attn_mean = attn_weights.mean(dim=0).squeeze().tolist()

logger.info({
    "event": "attention_logging",
    "attn_mean": attn_mean
})
