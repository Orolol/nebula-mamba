import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from model.nebula import NebulaModel
from model.diffusion import DiffusionHead
from model.config import NEBULA_CONFIGS
from data.data_loader_packed import PackedFinewebDataset

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
    
    # Check Mamba
    try:
        import mamba_ssm
        print("SUCCESS: Mamba-SSM is installed and will be used.")
    except ImportError:
        print("WARNING: Mamba-SSM not found. Using slow/mock implementation!")

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
    
    # Data (Packed & Optimized)
    print("Initializing PackedFinewebDataset...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = PackedFinewebDataset(
        split="train",
        max_length=config.seq_len,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        buffer_docs=10000, # Large buffer for better packing
        prefetch_batches=32
    )
    
    # Packed dataset handles batching internally, so batch_size=None
    # It also has its own producer thread, so num_workers=0 is usually best to avoid overhead/duplication
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)
    
    model.train()
    
    print("Starting training with BF16 Mixed Precision & Packed Loader...")
    step = 0
    pbar = tqdm(total=config.num_steps)
    
    import time
    t0 = time.time()
    total_tokens_trained = 0
    
    # BF16 Context
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using Mixed Precision: {dtype}")
    
    for batch in dataloader:
        if step >= config.num_steps:
            break
            
        # Packed loader returns dict
        input_ids = batch["input_ids"].to(device)
        
        # Calculate tokens per second (excluding padding)
        pad_token_id = tokenizer.pad_token_id
        num_tokens = (input_ids != pad_token_id).sum().item()
        total_tokens_trained += num_tokens
        
        # Forward Diffusion
        # Note: We ignore batch['labels'] (AR labels) and generate diffusion masks
        masked_input, labels, mask_mask = diffusion_helper.forward_diffusion(input_ids, mask_token_id)
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            logits = model(masked_input) 
            
            # Loss
            logits_flat = logits.view(-1, config.vocab_size)
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_sec = num_tokens / dt
        
        # Logging
        loss_val = loss.item()
        writer.add_scalar("Loss/train", loss_val, step)
        writer.add_scalar("Perf/tokens_per_sec", tokens_per_sec, step)
        writer.add_scalar("Perf/total_tokens", total_tokens_trained, step)
        
        if args.use_wandb:
            wandb.log({
                "loss": loss_val, 
                "tokens_per_sec": tokens_per_sec,
                "total_tokens": total_tokens_trained,
                "step": step
            })
            
        pbar.set_description(f"Loss: {loss_val:.4f} | TPS: {tokens_per_sec:.0f}")
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
