import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from attention_condition_generator.conditioner import AttentionConditioner


class UserEmbeddingDataset(Dataset):
    """
    Dataset for GAT user embeddings with domain labels.
    Each user embedding is paired with a domain label for conditioning.
    """
    
    def __init__(self, user_embeddings, domain_labels, domain_to_idx):
        """
        Args:
            user_embeddings: (N, D) tensor - GAT user embeddings only
            domain_labels: list of str - Domain label for each user sample
            domain_to_idx: dict - Domain string to index mapping
        """
        self.embeddings = user_embeddings
        self.domain_labels = domain_labels
        self.domain_to_idx = domain_to_idx
        
        assert len(self.embeddings) == len(self.domain_labels), \
            "Number of embeddings must match number of labels"
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        user_emb = self.embeddings[idx]
        domain_str = self.domain_labels[idx]
        domain_id = self.domain_to_idx[domain_str]
        
        return user_emb, domain_id, idx


class ConditionerTrainer:
    """
    Trainer class for AttentionConditioner model.
    Trains the model to generate domain-specific condition vectors from user embeddings.
    """
    
    def __init__(
        self,
        model,
        device,
        learning_rate=1e-3,
        weight_decay=1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        # AdamW optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Reduce learning rate when validation loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history for visualization
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def reconstruction_loss(self, conditioned, original):
        """
        Compute reconstruction loss to preserve original information
        while adding domain-specific conditioning.
        
        Args:
            conditioned: Domain-conditioned embeddings
            original: Original user embeddings
            
        Returns:
            total_loss: Combined loss
            mse: Mean squared error
            cosine_loss: Cosine similarity loss
        """
        # MSE loss: conditioned embedding should be close to original
        mse = nn.functional.mse_loss(conditioned, original)
        
        # Cosine similarity loss: preserve semantic information
        cos_sim = nn.functional.cosine_similarity(conditioned, original, dim=1).mean()
        cosine_loss = 1 - cos_sim
        
        # Combine losses: prioritize reconstruction, add cosine for semantics
        total_loss = mse + 0.1 * cosine_loss
        
        return total_loss, mse, cosine_loss
    
    def diversity_loss(self, conditioned_batch, domain_ids):
        """
        Encourage diversity between different domains.
        Embeddings from different domains should be distinguishable.
        
        Args:
            conditioned_batch: Batch of conditioned embeddings
            domain_ids: Domain IDs for each embedding in batch
            
        Returns:
            diversity_loss: Scalar loss value
        """
        if len(conditioned_batch) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Check if batch has multiple domains
        unique_domains = torch.unique(domain_ids)
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute mean embedding for each domain
        domain_means = []
        for domain_id in unique_domains:
            mask = domain_ids == domain_id
            if mask.sum() > 0:
                domain_mean = conditioned_batch[mask].mean(dim=0)
                domain_means.append(domain_mean)
        
        if len(domain_means) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Domain means should be different from each other
        domain_means = torch.stack(domain_means)
        normalized = nn.functional.normalize(domain_means, dim=1)
        sim_matrix = torch.mm(normalized, normalized.t())
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(len(domain_means), device=self.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        
        # Minimize similarity between domains (encourage diversity)
        diversity_loss = sim_matrix.abs().mean()
        
        return diversity_loss
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            metrics: Dictionary of average metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0
        total_mse = 0
        total_cosine = 0
        total_diversity = 0
        
        pbar = tqdm(dataloader, desc="Training")
        
        for user_emb, domain_id, _ in pbar:
            user_emb = user_emb.to(self.device)
            domain_id = domain_id.to(self.device)
            
            # Forward pass: generate domain-conditioned embeddings
            conditioned = self.model(user_emb, domain_id)
            
            # Compute losses
            loss, mse, cosine_loss = self.reconstruction_loss(conditioned, user_emb)
            div_loss = self.diversity_loss(conditioned, domain_id)
            
            # Total loss with diversity regularization
            total = loss + 0.1 * div_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += total.item()
            total_mse += mse.item()
            total_cosine += cosine_loss.item()
            total_diversity += div_loss.item()
            
            pbar.set_postfix({
                'loss': f'{total.item():.4f}',
                'mse': f'{mse.item():.4f}',
                'div': f'{div_loss.item():.4f}'
            })
        
        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'mse': total_mse / n,
            'cosine': total_cosine / n,
            'diversity': total_diversity / n
        }
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            metrics: Dictionary of average validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_mse = 0
        
        with torch.no_grad():
            for user_emb, domain_id, _ in dataloader:
                user_emb = user_emb.to(self.device)
                domain_id = domain_id.to(self.device)
                
                conditioned = self.model(user_emb, domain_id)
                
                loss, mse, _ = self.reconstruction_loss(conditioned, user_emb)
                
                total_loss += loss.item()
                total_mse += mse.item()
        
        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'mse': total_mse / n
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs,
        save_dir='checkpoints'
    ):
        """
        Full training loop with validation and checkpointing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            
        Returns:
            history: Training history dictionary
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print metrics
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"Cosine: {train_metrics['cosine']:.4f}, "
                  f"Diversity: {train_metrics['diversity']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"MSE: {val_metrics['mse']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'history': self.history
                }, save_dir / 'best_model.pt')
                print(f"✓ Best model saved (val_loss: {best_val_loss:.4f})")
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history


def generate_condition_vectors(
    model,
    user_embeddings,
    domain_to_idx,
    device,
    output_path='./outputs/condition_vectors_for_diffusion.pt'
):
    """
    Generate condition vectors for all users across all domains.
    These vectors will be used as conditioning input for the diffusion model.
    
    Args:
        model: Trained AttentionConditioner model
        user_embeddings: Original user embeddings from GAT
        domain_to_idx: Domain name to index mapping
        device: Device to run inference on
        output_path: Path to save condition vectors
        
    Returns:
        output_data: Dictionary containing condition vectors and metadata
    """
    model.eval()
    
    print("\n" + "="*60)
    print("Generating condition vectors for diffusion model")
    print("="*60 + "\n")
    
    num_users = len(user_embeddings)
    emb_dim = user_embeddings.shape[1]
    num_domains = len(domain_to_idx)
    
    # Create reverse mapping
    idx_to_domain = {v: k for k, v in domain_to_idx.items()}
    
    # Initialize storage for condition vectors: (num_users, num_domains, emb_dim)
    all_conditions = torch.zeros(num_users, num_domains, emb_dim)
    
    # Store attention weights for analysis
    all_attention_weights = {}
    
    with torch.no_grad():
        for domain_name, domain_id in domain_to_idx.items():
            print(f"Processing domain: '{domain_name}' (id={domain_id})")
            
            # Process in batches for efficiency
            batch_size = 256
            domain_attention_weights = []
            
            for i in tqdm(range(0, num_users, batch_size), desc=f"  Batches"):
                end_idx = min(i + batch_size, num_users)
                batch_emb = user_embeddings[i:end_idx].to(device)
                
                # Generate domain-conditioned vectors
                conditioned, attn_weights = model(
                    batch_emb,
                    domain_id,
                    return_attn=True
                )
                
                all_conditions[i:end_idx, domain_id] = conditioned.cpu()
                
                # Save attention weights from first batch for analysis
                if i == 0:
                    domain_attention_weights.append(attn_weights.cpu())
            
            all_attention_weights[domain_name] = torch.cat(domain_attention_weights, dim=0)
    
    # Save everything for diffusion model
    output_data = {
        'condition_vectors': all_conditions,      # (num_users, num_domains, emb_dim)
        'attention_weights': all_attention_weights,  # dict: {domain_name: weights}
        'domain_to_idx': domain_to_idx,           # {"user": 0, "movie": 1}
        'idx_to_domain': idx_to_domain,           # {0: "user", 1: "movie"}
        'num_users': num_users,
        'num_domains': num_domains,
        'embedding_dim': emb_dim,
        'model_state_dict': model.state_dict()
    }
    
    torch.save(output_data, output_path)
    
    print(f"\n✓ Condition vectors saved to '{output_path}'")
    print(f"  Shape: {all_conditions.shape}")
    print(f"  - {num_users} users")
    print(f"  - {num_domains} domains: {list(domain_to_idx.keys())}")
    print(f"  - {emb_dim} dimensions")
    
    # Print domain-wise statistics
    print(f"\nDomain-wise statistics:")
    for domain_name, domain_id in domain_to_idx.items():
        domain_conds = all_conditions[:, domain_id, :]
        print(f"  {domain_name}:")
        print(f"    Mean: {domain_conds.mean():.4f}")
        print(f"    Std:  {domain_conds.std():.4f}")
        print(f"    Min:  {domain_conds.min():.4f}")
        print(f"    Max:  {domain_conds.max():.4f}")
    
    return output_data


def load_gat_embeddings(path):
    """
    Load GAT embeddings from file.
    Expected format: {'embeddings': {'user': Tensor, 'movie': Tensor, 'book': Tensor}}
    
    We only use USER embeddings for conditioning.
    
    Args:
        path: Path to GAT embeddings file
        
    Returns:
        user_embeddings: User embeddings only
        domain_labels: Domain label for each user sample
        domain_to_idx: Domain name to index mapping
    """
    print(f"Loading GAT embeddings from '{path}'...")
    
    gat_data = torch.load(path)
    
    # Extract embeddings dict
    if isinstance(gat_data, dict) and 'embeddings' in gat_data:
        embeddings_dict = gat_data['embeddings']
        
        if not isinstance(embeddings_dict, dict):
            raise ValueError("embeddings should be a dict with domain keys")
        
        print(f"\nFound embedding types: {list(embeddings_dict.keys())}")
        for key, val in embeddings_dict.items():
            print(f"  {key}: {val.shape}")
        
        # Extract only USER embeddings
        if 'user' not in embeddings_dict:
            raise ValueError("'user' embeddings not found in file")
        
        user_embeddings = embeddings_dict['user']
        print(f"\n✓ Using USER embeddings only: {user_embeddings.shape}")
        
        # User embeddings contain interactions from BOTH movie and book domains
        # We need to condition these embeddings based on which domain we're generating for
        # Domain embeddings represent: MOVIE domain or BOOK domain
        
        # Available item domains for conditioning
        available_domains = ['movie', 'book']
        domain_to_idx = {name: idx for idx, name in enumerate(available_domains)}
        
        print(f"\nDomain conditioning setup:")
        print(f"  User embeddings contain interactions from: {available_domains}")
        print(f"  Global domain vectors represent: {available_domains}")
        print(f"  Domain mapping: {domain_to_idx}")
        print(f"\nPurpose:")
        print(f"  - 'movie' domain: Condition user for movie recommendation")
        print(f"  - 'book' domain: Condition user for book recommendation")
        
        # Create dataset: each user embedding will be used with BOTH domains
        # This creates 2x the data (each user x 2 domains)
        num_users = len(user_embeddings)
        
        # Replicate user embeddings for each domain
        all_user_embeddings = []
        all_domain_labels = []
        
        for domain_name in available_domains:
            all_user_embeddings.append(user_embeddings)
            all_domain_labels.extend([domain_name] * num_users)
        
        # Stack all embeddings
        all_user_embeddings = torch.cat(all_user_embeddings, dim=0)
        
        print(f"\n✓ Dataset created:")
        print(f"  Total samples: {len(all_user_embeddings)} ({num_users} users x {len(available_domains)} domains)")
        print(f"  Label distribution:")
        for domain_name in available_domains:
            count = all_domain_labels.count(domain_name)
            print(f"    {domain_name}: {count} ({count/len(all_domain_labels)*100:.1f}%)")
        
        return all_user_embeddings, all_domain_labels, domain_to_idx
    
    else:
        raise ValueError(
            "Expected format: {'embeddings': {'user': Tensor, 'movie': Tensor, ...}}"
        )


def main():
    """
    Main training pipeline:
    1. Load GAT user embeddings
    2. Create domain-labeled dataset
    3. Train AttentionConditioner model
    4. Generate condition vectors for diffusion model
    """
    
    # ========== Configuration ==========
    config = {
        'gat_embeddings_path': './embeddings/gat_embeddings.pt',
        'num_heads': 4,          # Number of attention heads
        'dropout': 0.1,          # Dropout rate for regularization
        'batch_size': 128,       # Batch size for training
        'num_epochs': 50,        # Number of training epochs
        'learning_rate': 1e-3,   # Initial learning rate
        'weight_decay': 1e-5,    # L2 regularization
        'train_split': 0.8,      # Train/validation split ratio
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Configuration:")
    print(json.dumps(config, indent=2))
    
    # ========== Load GAT Embeddings ==========
    user_embeddings, domain_labels, domain_to_idx = load_gat_embeddings(
        config['gat_embeddings_path']
    )
    
    emb_dim = user_embeddings.shape[1]
    num_samples = user_embeddings.shape[0]
    num_domains = len(domain_to_idx)
    num_actual_users = num_samples // num_domains  # Original number of unique users
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Unique users: {num_actual_users}")
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Domains: {num_domains} {list(domain_to_idx.keys())}")
    print(f"  Total training samples: {num_samples} (users x domains)")
    print(f"{'='*60}\n")
    
    # ========== Train/Val Split ==========
    train_size = int(config['train_split'] * num_samples)
    
    train_embeddings = user_embeddings[:train_size]
    train_labels = domain_labels[:train_size]
    
    val_embeddings = user_embeddings[train_size:]
    val_labels = domain_labels[train_size:]
    
    print(f"Data split:")
    print(f"  Train: {len(train_embeddings)} samples")
    print(f"  Val:   {len(val_embeddings)} samples")
    
    # ========== Create Datasets & DataLoaders ==========
    train_dataset = UserEmbeddingDataset(
        train_embeddings, 
        train_labels, 
        domain_to_idx
    )
    val_dataset = UserEmbeddingDataset(
        val_embeddings, 
        val_labels, 
        domain_to_idx
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # ========== Initialize Model ==========
    print(f"\nInitializing AttentionConditioner...")
    
    model = AttentionConditioner(
        emb_dim=emb_dim,
        num_domains=num_domains,
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created")
    print(f"  Architecture: Multi-Head Cross-Attention")
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Num domains: {num_domains}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ========== Train Model ==========
    trainer = ConditionerTrainer(
        model=model,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        save_dir='checkpoints'
    )
    
    # ========== Load Best Model ==========
    print("\nLoading best model...")
    checkpoint = torch.load('checkpoints/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")
    
    # ========== Generate Condition Vectors ==========
    # Use original user embeddings (before domain replication) for generation
    original_user_embeddings = user_embeddings[:num_actual_users]
    
    condition_data = generate_condition_vectors(
        model=model,
        user_embeddings=original_user_embeddings,
        domain_to_idx=domain_to_idx,
        device=config['device'],
        output_path='./outputs/condition_vectors_for_diffusion.pt'
    )
    
    # ========== Save Configuration ==========
    config['domain_to_idx'] = domain_to_idx
    config['num_domains'] = num_domains
    config['embedding_dim'] = emb_dim
    config['num_users'] = num_actual_users
    
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ Training pipeline completed successfully!")
    print("="*60)
    print("\nOutput files:")
    print("  - checkpoints/best_model.pt")
    print("  - ./outputs/condition_vectors_for_diffusion.pt")
    print("  - training_config.json")
    print(f"\nCondition vectors ready for diffusion model:")
    print(f"  Shape: ({num_actual_users}, {num_domains}, {emb_dim})")
    print(f"  Domains: {list(domain_to_idx.keys())}")
    print("\nDone!")


if __name__ == "__main__":
    main()