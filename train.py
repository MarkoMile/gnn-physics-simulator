"""
Training loop for the PyTorch GNN Physics Simulator.

Implements DeepMind's optimization strategies:
- Adam Optimizer (lr=1e-4)
- Exponential Learning Rate Decay (0.1 over 5M steps)

- 20M Maximum Training Steps
"""

import os
import yaml
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.loader import DataLoader

from dataset import ParticleDataset, batch_graphs
from models import GNNSimulator

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_loss(predicted_acc: torch.Tensor, target_acc: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error (MSE) loss between predicted and target dimensionless accelerations."""
    return nn.functional.mse_loss(predicted_acc, target_acc)

def train_step(model, batch, optimizer, device):
    """Executes a single optimizer step against a batched simulation graph."""
    model.train()
    batch = batch.to(device)
    
    optimizer.zero_grad()
    
    # Forward Pass
    predictions = model(batch)
    
    # Calculate Loss (Predictions are dimensionless un-normalized tensors)
    loss = compute_loss(predictions, batch.y)
    
    # Backward Pass
    loss.backward()
    
    # Potentially clip gradients to prevent exploding forces (not used in original paper)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
    
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def validate(model, val_loader, device):
    """Evaluates the model across the Validation split efficiently without tracking gradients."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        batch = batch.to(device)
        predictions = model(batch)
        loss = compute_loss(predictions, batch.y)
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(num_batches, 1)

def train(config_path: str, use_wandb: bool = False):
    """
    Main training loop.
    Maps datasets, handles schedulers, checkpoints best weights, and limits step bounds.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    if use_wandb:
        import wandb
        wandb.init(project="gnn-physics-simulator", config=config)
        
    # 1. Prepare Datasets
    data_cfg = config['data']
    train_cfg = config['training']
    
    print("\nLoading Datasets...")
    # Initialize the raw splits utilizing the random-walk dataset implementations
    train_dataset = ParticleDataset(data_cfg['dataset_path'], split='train', dataset_format=data_cfg['dataset_format'], noise_std=train_cfg.get('noise_std', 3e-4))
    valid_dataset = ParticleDataset(data_cfg['dataset_path'], split='valid', dataset_format=data_cfg['dataset_format'], noise_std=0.0) # No noise in evaluation!
    
    # Wrap in PyTorch Geometric's fast DataLoader mapping `batch_graphs` collate logic internally
    # Drop last to ensure stable gradient tensor sizes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg.get('batch_size', 2), 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset, 
        batch_size=train_cfg.get('batch_size', 2), 
        shuffle=False
    )
    
    # 2. Initialize Model
    print("Building GNNSimulator Network...")
    model = GNNSimulator.from_config(config).to(device)
    
    # 3. Setup Optimizers & Schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get('learning_rate', 1e-4))
    
    # DeepMind specified an exponential decay of 0.1 over 5 million steps
    epochs = train_cfg.get('epochs', 100) # Fallback bounded epochs
    total_steps = len(train_loader) * epochs
    
    # Calculate exactly what the gamma decay multiplier should be per step to hit 0.1 exactly at 5M
    # gamma^(5,000,000) = 0.1  -->  gamma = 0.1^(1/5,000,000)
    decay_steps = train_cfg.get('lr_decay_steps', 5_000_000)
    gamma = 0.1 ** (1.0 / decay_steps)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    # DeepMind specifies training bounds of roughly 20 Million gradient updates
    max_training_steps = 20_000_000
    
    # 4. Training Loop Variables
    global_step = 0
    best_val_loss = float('inf')
    early_stopping_patience = train_cfg.get('early_stopping_patience', 15)
    patience_counter = 0
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nBeginning Training (Max Steps: {max_training_steps:,})")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        # Wrapping loader in TQDM for a beautiful progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch in progress_bar:
            if global_step >= max_training_steps:
                print(f"Reached absolute step limit of {max_training_steps}. Halting.")
                break
                
            loss = train_step(model, batch, optimizer, device)
            scheduler.step()
            
            epoch_loss += loss
            global_step += 1
            
            # Update progress bar trailing dictionary
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({'loss': f"{loss:.4f}", 'lr': f"{current_lr:.2e}"})
            
            if use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss,
                    "train/learning_rate": current_lr,
                    "global_step": global_step
                })
        
        # Epoch complete - Compute exact validation bounds
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, device)
        
        print(f"\nEpoch {epoch} Results | Train Loss: {avg_train_loss:.5f} | Valid Loss: {val_loss:.5f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": avg_train_loss,
                "val/loss": val_loss
            })
            
        # 5. Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the optimal state dictionary! (This captures weights, biases, AND the dynamic normalizer statistics!)
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, save_path)
            print(f"--> Saved improved checkpoint to {save_path}")
        else:
            patience_counter += 1
            print(f"--> Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement!")
                break
                
        if global_step >= max_training_steps:
            break

    print(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.5f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GNN Physics Simulator")
    parser.add_config = parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    train(args.config, args.wandb)
