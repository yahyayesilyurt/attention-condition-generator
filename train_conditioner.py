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
    """
    def __init__(self, user_embeddings, domain_labels, domain_to_idx):
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
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    
    def reconstruction_loss(self, conditioned, original):
        # MSE loss: conditioned embedding should be close to original
        # CRITICAL UPDATE: Lowered MSE weight to allow domain transformation
        mse = nn.functional.mse_loss(conditioned, original)
        
        # Cosine similarity loss: preserve semantic information
        cos_sim = nn.functional.cosine_similarity(conditioned, original, dim=1).mean()
        cosine_loss = 1 - cos_sim
        
        # Combine losses
        total_loss = 0.5 * mse + 0.5 * cosine_loss
        return total_loss, mse, cosine_loss
    
    def diversity_loss(self, conditioned_batch, domain_ids):
        if len(conditioned_batch) < 2: return torch.tensor(0.0, device=self.device)
        
        unique_domains = torch.unique(domain_ids)
        if len(unique_domains) < 2: return torch.tensor(0.0, device=self.device)
        
        domain_means = []
        for domain_id in unique_domains:
            mask = domain_ids == domain_id
            if mask.sum() > 0:
                domain_means.append(conditioned_batch[mask].mean(dim=0))
        
        if len(domain_means) < 2: return torch.tensor(0.0, device=self.device)
        
        domain_means = torch.stack(domain_means)
        normalized = nn.functional.normalize(domain_means, dim=1)
        sim_matrix = torch.mm(normalized, normalized.t())
        
        mask = torch.eye(len(domain_means), device=self.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        
        diversity_loss = sim_matrix.abs().mean()
        return diversity_loss
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0; total_mse = 0; total_cosine = 0; total_diversity = 0
        
        pbar = tqdm(dataloader, desc="Training")
        
        for user_emb, domain_id, _ in pbar:
            user_emb = user_emb.to(self.device)
            domain_id = domain_id.to(self.device)
            
            conditioned = self.model(user_emb, domain_id)
            
            loss, mse, cosine_loss = self.reconstruction_loss(conditioned, user_emb)
            div_loss = self.diversity_loss(conditioned, domain_id)
            
            # CRITICAL UPDATE: Increased Diversity Weight from 0.1 to 1.0
            total = loss + 1.0 * div_loss
            
            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total.item()
            total_mse += mse.item()
            total_cosine += cosine_loss.item()
            total_diversity += div_loss.item()
            
            pbar.set_postfix({'loss': f'{total.item():.4f}', 'div': f'{div_loss.item():.4f}'})
        
        n = len(dataloader)
        return {
            'loss': total_loss / n, 'mse': total_mse / n, 
            'cosine': total_cosine / n, 'diversity': total_diversity / n
        }
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0; total_mse = 0
        with torch.no_grad():
            for user_emb, domain_id, _ in dataloader:
                user_emb = user_emb.to(self.device)
                domain_id = domain_id.to(self.device)
                conditioned = self.model(user_emb, domain_id)
                loss, mse, _ = self.reconstruction_loss(conditioned, user_emb)
                total_loss += loss.item()
                total_mse += mse.item()
        n = len(dataloader)
        return {'loss': total_loss / n, 'mse': total_mse / n}
    
    def train(self, train_loader, val_loader, num_epochs, save_dir='checkpoints'):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {num_epochs} epochs\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step(val_metrics['loss'])
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Diversity: {train_metrics['diversity']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}")
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch, 'val_loss': best_val_loss}, save_dir / 'best_model.pt')
                print(f"✓ Best model saved")
                
        return self.history


def generate_condition_vectors(model, user_embeddings, domain_to_idx, device, output_path):
    model.eval()
    print(f"\nGenerating condition vectors... Shape: {user_embeddings.shape}")
    
    num_users = len(user_embeddings)
    emb_dim = user_embeddings.shape[1]
    num_domains = len(domain_to_idx)
    idx_to_domain = {v: k for k, v in domain_to_idx.items()}
    
    # NOTE: We keep ALL users (even zero-vectors) to maintain ID alignment
    all_conditions = torch.zeros(num_users, num_domains, emb_dim)
    
    with torch.no_grad():
        for domain_name, domain_id in domain_to_idx.items():
            print(f"Processing domain: '{domain_name}'")
            batch_size = 512
            for i in tqdm(range(0, num_users, batch_size)):
                end_idx = min(i + batch_size, num_users)
                batch_emb = user_embeddings[i:end_idx].to(device)
                
                # Check for zero vectors in batch to save compute (optional)
                # But running model on zeros is fine, it outputs close to zero.
                
                conditioned = model(batch_emb, domain_id)
                all_conditions[i:end_idx, domain_id] = conditioned.cpu()

    output_data = {
        'condition_vectors': all_conditions,
        'domain_to_idx': domain_to_idx,
        'num_users': num_users,
        'embedding_dim': emb_dim
    }
    torch.save(output_data, output_path)
    print(f"✓ Saved to {output_path}")
    return output_data


def load_gat_embeddings(path):
    print(f"Loading GAT embeddings from '{path}'...")
    gat_data = torch.load(path, map_location='cpu')
    embeddings_dict = gat_data['embeddings']
    user_embeddings = embeddings_dict['user']
    
    print(f"✓ Loaded USER embeddings: {user_embeddings.shape}")
    
    # DO NOT DROP ROWS! 
    # Just identify valid ones for training.
    norms = torch.norm(user_embeddings, dim=1)
    valid_mask = norms > 1e-5
    
    num_valid = valid_mask.sum().item()
    print(f"  Valid users (non-zero): {num_valid} / {user_embeddings.shape[0]}")
    
    available_domains = ['movie', 'book']
    domain_to_idx = {name: idx for idx, name in enumerate(available_domains)}
    
    # Return everything + the mask
    return user_embeddings, valid_mask, available_domains, domain_to_idx


def main():
    config = {
        'gat_embeddings_path': './embeddings/gat_embeddings.pt',
        'num_heads': 4, # Optimized for 32-dim
        'dropout': 0.1,
        'batch_size': 128,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'train_split': 0.8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 1. Load Data (Without Dropping)
    user_embeddings, valid_mask, available_domains, domain_to_idx = load_gat_embeddings(
        config['gat_embeddings_path']
    )
    
    # 2. Create Train/Val Split (Using ONLY Valid Users)
    valid_indices = torch.nonzero(valid_mask).squeeze() # Indices of valid users
    num_valid_users = len(valid_indices)
    
    perm = torch.randperm(num_valid_users)
    train_count = int(config['train_split'] * num_valid_users)
    
    train_user_idxs = valid_indices[perm[:train_count]]
    val_user_idxs = valid_indices[perm[train_count:]]
    
    # Helper to create domain-duplicated dataset
    def create_dataset_inputs(user_indices_subset):
        # user_indices_subset: Real indices in the big tensor
        embs_list = []
        labels_list = []
        
        subset_embeddings = user_embeddings[user_indices_subset]
        
        for domain in available_domains:
            embs_list.append(subset_embeddings)
            labels_list.extend([domain] * len(subset_embeddings))
            
        return torch.cat(embs_list, dim=0), labels_list

    print("Creating datasets...")
    train_embs, train_lbls = create_dataset_inputs(train_user_idxs)
    val_embs, val_lbls = create_dataset_inputs(val_user_idxs)
    
    print(f"Train samples: {len(train_embs)} | Val samples: {len(val_embs)}")
    
    train_loader = DataLoader(
        UserEmbeddingDataset(train_embs, train_lbls, domain_to_idx),
        batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        UserEmbeddingDataset(val_embs, val_lbls, domain_to_idx),
        batch_size=config['batch_size'], shuffle=False
    )
    
    # 3. Initialize Model
    model = AttentionConditioner(
        emb_dim=user_embeddings.shape[1],
        num_domains=len(available_domains),
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    # 4. Train
    trainer = ConditionerTrainer(model, config['device'], config['learning_rate'], config['weight_decay'])
    trainer.train(train_loader, val_loader, config['num_epochs'])
    
    # 5. Generate (Using ALL users to keep alignment)
    checkpoint = torch.load('checkpoints/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    generate_condition_vectors(
        model, user_embeddings, domain_to_idx, config['device'], 
        './outputs/condition_vectors_for_diffusion.pt'
    )

if __name__ == "__main__":
    main()