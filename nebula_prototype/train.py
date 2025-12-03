import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from tqdm import tqdm
from model.nebula import NebulaModel
from model.diffusion import DiffusionHead
from data.fineweb_loader import get_dataloader

@dataclass
class Config:
    d_model: int = 512       # Scaled up
    num_layers: int = 16      # Scaled up
    num_heads: int = 16        # Adjusted for d_model
    vocab_size: int = 50257   # GPT-2
    lr: float = 3e-4          # Standard for this size
    batch_size: int = 32      # H100 can handle this (or more)
    seq_len: int = 1024       # Standard context
    num_steps: int = 1000     # Longer run

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    model = NebulaModel(config).to(device)
    
    num_params = count_parameters(model)
    print(f"Model initialized with {num_params:,} parameters.")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Diffusion helper
    diffusion_helper = DiffusionHead(config.vocab_size)
    mask_token_id = 50256 # GPT-2 EOS as mask for prototype
    
    dataloader = get_dataloader(split="train", seq_len=config.seq_len, batch_size=config.batch_size)
    
    model.train()
    
    print("Starting training...")
    for step, (input_ids, _) in enumerate(tqdm(dataloader, total=config.num_steps)):
        if step >= config.num_steps:
            break
            
        input_ids = input_ids.to(device)
        
        # 1. Forward Diffusion (Masking)
        masked_input, labels, mask_mask = diffusion_helper.forward_diffusion(input_ids, mask_token_id)
        
        # 2. Model Prediction
        optimizer.zero_grad()
        logits = model(masked_input) # [B, L, V]
        
        # 3. Loss Calculation (Only on masked tokens)
        # Flatten for loss
        logits_flat = logits.view(-1, config.vocab_size)
        labels_flat = labels.view(-1)
        
        loss = criterion(logits_flat, labels_flat)
        
        # 4. Backward
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    train()
