import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.nebula import NebulaModel
from model.diffusion import DiffusionHead
from model.config import NEBULA_CONFIGS
from data.fineweb_loader import get_dataloader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):
    # 1. Config & Device
    if args.config not in NEBULA_CONFIGS:
        raise ValueError(f"Unknown config: {args.config}. Available: {list(NEBULA_CONFIGS.keys())}")
    
    config = NEBULA_CONFIGS[args.config]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading config: {args.config}")

    # 2. Logging Setup
    writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
    if args.use_wandb:
        wandb.init(project="nebula-mamba", name=args.run_name, config=config.__dict__)

    # 3. Model Setup
    model = NebulaModel(config).to(device)
    num_params = count_parameters(model)
    print(f"Model initialized with {num_params:,} parameters.")
    
    # 4. Compilation (Torch 2.0+)
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Diffusion helper
    diffusion_helper = DiffusionHead(config.vocab_size)
    mask_token_id = 50256 # GPT-2 EOS as mask
    
    # Data
    dataloader = get_dataloader(split="train", seq_len=config.seq_len, batch_size=config.batch_size)
    
    model.train()
    
    print("Starting training...")
    step = 0
    pbar = tqdm(total=config.num_steps)
    
    for input_ids, _ in dataloader:
        if step >= config.num_steps:
            break
            
        input_ids = input_ids.to(device)
        
        # Forward Diffusion
        masked_input, labels, mask_mask = diffusion_helper.forward_diffusion(input_ids, mask_token_id)
        
        # Prediction
        optimizer.zero_grad()
        logits = model(masked_input) 
        
        # Loss
        logits_flat = logits.view(-1, config.vocab_size)
        labels_flat = labels.view(-1)
        loss = criterion(logits_flat, labels_flat)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Logging
        loss_val = loss.item()
        writer.add_scalar("Loss/train", loss_val, step)
        if args.use_wandb:
            wandb.log({"loss": loss_val, "step": step})
            
        pbar.set_description(f"Loss: {loss_val:.4f}")
        pbar.update(1)
        step += 1

    writer.close()
    if args.use_wandb:
        wandb.finish()
        
    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/{args.run_name}.pt")
    print(f"Saved checkpoint to checkpoints/{args.run_name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="small", help="Model config name (small, base, 500M)")
    parser.add_argument("--run_name", type=str, default="nebula_run", help="Name for logs and checkpoint")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()
    
    # Create dirs
    import os
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    train(args)
