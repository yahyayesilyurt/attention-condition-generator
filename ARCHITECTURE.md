# 🏗️ Architecture Overview

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAT (Heterogeneous Graph)                     │
│                                                                   │
│    User ←──interacts──→ Movie                                   │
│     │                                                             │
│     └───interacts──→ Book                                        │
│                                                                   │
│  Output: User Embeddings (27785, 32)                            │
│          ↓                                                        │
│  Contains information about BOTH movie and book interactions    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              AttentionConditioner (This Module)                  │
│                                                                   │
│  Input:  User Embedding (32-dim)                                │
│          Domain Label: "movie" or "book"                         │
│                                                                   │
│  Process:                                                         │
│    Query: User Embedding                                         │
│    Key/Value: Learnable Global Domain Embedding                 │
│               ├─ Domain "movie": Represents movie domain         │
│               └─ Domain "book": Represents book domain           │
│                                                                   │
│  Multi-Head Cross-Attention:                                     │
│    - 4 attention heads                                           │
│    - Each head learns different aspects of domain conditioning  │
│                                                                   │
│  Output: Domain-Conditioned User Embedding (32-dim)             │
│          ├─ For "movie": User representation tuned for movies   │
│          └─ For "book": User representation tuned for books     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Points

### 1. User Embeddings (Input)

- **Source**: GAT model on heterogeneous graph
- **Shape**: (27785, 32)
- **Content**: Contains user's interactions with both movies and books
- **Interpretation**: Rich representation of user preferences across domains

### 2. Domain Embeddings (Learnable)

Two global vectors learned during training:

```python
Domain Embedding "movie" (32-dim):
  - Represents characteristics of movie domain
  - Learned to extract movie-relevant features from user embeddings
  - Used when generating movie recommendations

Domain Embedding "book" (32-dim):
  - Represents characteristics of book domain
  - Learned to extract book-relevant features from user embeddings
  - Used when generating book recommendations
```

### 3. Attention Mechanism

```
User Embedding (mixed movie+book info)
         ↓ [Query]

Cross-Attention with Domain Embedding
         ↓ [Key/Value: movie OR book]

Attention learns:
  - Which aspects of user embedding are relevant for this domain
  - How to weight different user preferences for movie vs book

         ↓

Domain-Conditioned Embedding
  - If domain="movie": Emphasizes movie-relevant user preferences
  - If domain="book": Emphasizes book-relevant user preferences
```

### 4. Why This Works

**Problem**: User embeddings contain information about both domains mixed together.

**Solution**: Attention mechanism + domain embeddings act as a "filter":

- Domain embedding "movie" → Extracts movie-relevant aspects
- Domain embedding "book" → Extracts book-relevant aspects

**Result**: Same user → Different representations for different domains

## Data Flow Example

### User 123's Embeddings:

```python
# Original GAT embedding (mixed)
user_123 = [0.5, -0.3, 0.8, 0.1, ...]  # (32-dim)
# Contains both movie AND book interaction patterns

# After conditioning for MOVIE domain
condition_movie = model(user_123, domain="movie")
# → [0.6, -0.2, 0.9, 0.05, ...]  # (32-dim)
# Emphasizes movie preferences

# After conditioning for BOOK domain
condition_book = model(user_123, domain="book")
# → [0.4, -0.4, 0.7, 0.15, ...]  # (32-dim)
# Emphasizes book preferences

# These are DIFFERENT because attention focused on different aspects
```

### Cosine Similarity Analysis:

```python
similarity(user_123, condition_movie) = 0.92  # High: preserves user info
similarity(user_123, condition_book) = 0.89   # High: preserves user info
similarity(condition_movie, condition_book) = 0.75  # Lower: domains differ
```

## Training Process

### Phase 1: Prepare Dataset

```python
For each user embedding (27785 total):
  Create 2 samples:
    1. (user_embedding, domain="movie")
    2. (user_embedding, domain="book")

Total training samples: 27785 × 2 = 55,570
```

### Phase 2: Train AttentionConditioner

```python
For each batch:
  Input: user_embeddings, domain_labels
  Output: conditioned_embeddings

  Loss:
    - Reconstruction: Keep user information
    - Diversity: Make movie ≠ book representations
    - Cosine: Preserve semantic meaning
```

### Phase 3: Generate Condition Vectors

```python
For each user (27785):
  For each domain (movie, book):
    condition[user, domain] = model(user_emb, domain)

Save: (27785, 2, 32) tensor
```

## What The Model Learns

### Domain Embedding "movie":

- Learns to identify movie-specific preference patterns in user embeddings
- Might focus on: genre preferences, director tastes, movie popularity patterns
- Transforms user embedding to emphasize movie-relevant features

### Domain Embedding "book":

- Learns to identify book-specific preference patterns in user embeddings
- Might focus on: author preferences, topic interests, reading complexity
- Transforms user embedding to emphasize book-relevant features

### Attention Heads:

Each of the 4 heads might learn different aspects:

- Head 1: Content similarity (action movies ↔ thriller books)
- Head 2: Popularity patterns (mainstream vs niche)
- Head 3: Temporal patterns (new releases vs classics)
- Head 4: Social patterns (trending items)

## Expected Behavior After Training

### Good Signs:

✅ Diversity loss > 0: Movie and book conditions are different  
✅ Cosine similarity 0.85-0.95: User information preserved  
✅ Attention weights non-uniform: Model learned to focus  
✅ Validation loss decreasing: Model generalizing well

### What To Avoid:

❌ Diversity loss → 0: Movie and book conditions are identical (model not learning domain differences)  
❌ Cosine similarity < 0.7: Too much user information lost  
❌ All attention heads identical: Heads not specializing  
❌ Overfitting: Val loss increasing while train loss decreasing

## Summary

**Input**: User embeddings with mixed movie+book interactions  
**Process**: Attention-based domain conditioning  
**Output**: Separate conditions optimized for movie vs book generation  
**Usage**: Feed into diffusion model for domain-specific recommendations

The key insight: **Same user, different domain perspectives** → Better domain-specific recommendations!
