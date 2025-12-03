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
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    vocab_size: int = 50257 # GPT-2
    lr: float = 1e-4
    batch_size: int = 4
    seq_len: int = 128 # Small for prototype
    num_steps: int = 100

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    model = NebulaModel(config).to(device)
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
