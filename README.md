# Attention Condition Generator

An attention-based conditioning module that generates domain-aware
condition vectors from user embeddings using multi-head cross-attention.

## Motivation

In cross-domain recommendation settings, user representations often need to be
adapted or conditioned based on domain-level context (e.g., _book_, _movie_).

Instead of relying on simple concatenation or static domain features, this module
uses **multi-head cross-attention** to dynamically fuse:

- **User embeddings** (queries)
- **Learnable domain embeddings** (keys & values)

The output is a **condition vector** that captures how a user should attend to
domain-specific information.

## Key Features

- Multi-head **cross-attention** (Query ≠ Key/Value)
- Learnable domain / global embeddings
- Clean and minimal public API
- Fully differentiable and end-to-end trainable
- Plug-and-play design (no dependency on diffusion or graph models)
- Shape-safe with broadcast support
  Project Structure

## High-Level Overview

Given:

- User embeddings `u ∈ R^{B×D}`
- Domain IDs `d`

The module computes:

- Q = u
- K = E_domain(d)
- V = E_domain(d)

condition = CrossAttention(Q, K, V)

where `E_domain` is a learnable embedding table and the attention mechanism follows the standard Transformer formulation.

## Project Structure

```bash
attention-condition-generator/
│
├── attention_condition_generator/
│   ├── __init__.py
│   │
│   ├── conditioner.py          # Public API: AttentionConditioner
│   ├── attention.py            # Multi-head cross-attention block
│   ├── domain_embedding.py     # Learnable domain embeddings
│   └── utils.py                # (optional) helper utilities
│
├── tests/
│   └── test_shapes.py          # Shape & backward-pass tests
│
├── examples/
│   └── simple_usage.py         # Minimal usage example
│
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the module in editable mode:

```bash
pip install -e .
```

This makes the module importable as a standard Python package while allowing local code changes without reinstallation.

## API Reference

### AttentionConditioner

```bash
AttentionConditioner(
    emb_dim: int,
    num_heads: int,
    num_domains: int,
    dropout: float = 0.0
)
```

Arguments:

- emb_dim: Dimensionality of user and domain embeddings
- num_heads: Number of attention heads
- num_domains: Total number of domains
- dropout: Dropout rate applied inside attention

### Forward

```bash
condition = conditioner(user_emb, domain_id)
```

Inputs:

- user_emb: Tensor of shape (B, D)
- domain_id: int or LongTensor of shape (B,)

Output:

- condition: Tensor of shape (B, D)

## Testing

Run shape and gradient tests using:

```bash
pytest tests/
```

These tests verify:

- Output shape correctness
- Domain ID broadcasting
- Gradient flow through the module
