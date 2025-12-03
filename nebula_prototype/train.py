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
from data.data_loader_fast import FastFinewebDataset

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
        print("Compiling Experts and Diffusion Head with torch.compile...")
        # Compile experts in each layer to avoid Mamba graph breaks
        for layer in model.layers:
            layer['experts'] = torch.compile(layer['experts'])
        
        # Compile diffusion head
        model.diffusion_head = torch.compile(model.diffusion_head)
        
        # Note: We do NOT compile the full model or the Mamba blocks
        # because Mamba's CUDA kernels currently cause graph breaks/issues with Dynamo.

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Diffusion helper
    diffusion_helper = DiffusionHead(config.vocab_size)
    mask_token_id = 50256 # GPT-2 EOS as mask
    
    # Data (Fast & Unpacked)
    print("Initializing FastFinewebDataset (Unpacked)...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = FastFinewebDataset(
        split="train",
        max_length=config.seq_len,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        buffer_docs=10000,
        prefetch_batches=32
    )
    
    # Fast dataset handles batching internally, so batch_size=None
    # It also has its own producer thread, so num_workers=0 is usually best to avoid overhead/duplication
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)
    
    model.train()
    
    print("Starting training with BF16 Mixed Precision & Fast Loader...")
    print(f"Token-based Gradient Accumulation: Target = {args.target_tokens} tokens/step")
    
    step = 0
    pbar = tqdm(total=config.num_steps)
    
    import time
    t0 = time.time()
    total_tokens_trained = 0
    
    # BF16 Context
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using Mixed Precision: {dtype}")
    
    # Timing stats
    timers = {
        "data": 0.0,
        "diffusion": 0.0,
        "forward": 0.0,
        "loss": 0.0,
        "backward": 0.0,
        "optim": 0.0,
        "total": 0.0
    }
    
    t_start_step = time.time()
    accumulated_tokens = 0
    optimizer.zero_grad()
    
    for batch in dataloader:
        t_data_end = time.time()
        timers["data"] += (t_data_end - t_start_step)
        
        if step >= config.num_steps:
            break
            
        # Packed loader returns dict
        input_ids = batch["input_ids"].to(device)
        
        # Calculate tokens per second (excluding padding)
        pad_token_id = tokenizer.pad_token_id
        num_tokens = (input_ids != pad_token_id).sum().item()
        total_tokens_trained += num_tokens
        
        # Forward Diffusion
        t_diff_start = time.time()
        masked_input, labels, mask_mask = diffusion_helper.forward_diffusion(input_ids, mask_token_id)
        torch.cuda.synchronize()
        timers["diffusion"] += (time.time() - t_diff_start)
        
        # Mixed Precision Forward
        t_fwd_start = time.time()
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            logits = model(masked_input) 
            
            # Loss
            logits_flat = logits.view(-1, config.vocab_size)
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            
            # Scale loss for token-based accumulation
            # We want the gradient to be weighted by the number of tokens in this batch
            # relative to the target number of tokens per step.
            scaled_loss = loss * (num_tokens / args.target_tokens)
            
        torch.cuda.synchronize()
        timers["forward"] += (time.time() - t_fwd_start)
        
        # Backward
        t_bwd_start = time.time()
        scaled_loss.backward()
        torch.cuda.synchronize()
        timers["backward"] += (time.time() - t_bwd_start)
        
        accumulated_tokens += num_tokens
        
        # Optimizer Step (only if target tokens reached)
        if accumulated_tokens >= args.target_tokens:
            t_opt_start = time.time()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            timers["optim"] += (time.time() - t_opt_start)
            
            # Metrics (recorded only on step)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # TPS is based on total tokens processed since last step
            tokens_per_sec = accumulated_tokens / dt
            
            # Logging
            loss_val = loss.item() # Note: this is the loss of the LAST micro-batch, not average. 
                                   # For smoother logs, we could accumulate loss value too, but this is acceptable.
            
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
            
            # Print timing breakdown every 10 steps
            if step % 10 == 0:
                total_time = sum(timers.values())
                print(f"\n[Step {step}] Timing Breakdown (avg over 10 steps):")
                print(f"  Data Load: {timers['data']*1000/10:.1f}ms")
                print(f"  Diffusion: {timers['diffusion']*1000/10:.1f}ms")
                print(f"  Forward:   {timers['forward']*1000/10:.1f}ms")
                print(f"  Backward:  {timers['backward']*1000/10:.1f}ms")
                print(f"  Optimizer: {timers['optim']*1000/10:.1f}ms")
                # Reset timers
                for k in timers: timers[k] = 0.0
            
            accumulated_tokens = 0
            
        t_start_step = time.time()

    writer.close()
    if args.use_wandb:
        wandb.finish()
        
    # Save checkpoint
    # Sanitize state_dict keys (remove _orig_mod prefix from torch.compile)
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        elif "._orig_mod." in k:
            new_state_dict[k.replace("._orig_mod.", ".")] = v
        else:
            new_state_dict[k] = v
            
    torch.save(new_state_dict, f"checkpoints/{args.run_name}.pt")
    print(f"Saved checkpoint to checkpoints/{args.run_name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="small", help="Model config name (small, base, 500M)")
    parser.add_argument("--run_name", type=str, default="nebula_run", help="Name for logs and checkpoint")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--target_tokens", type=int, default=65536, help="Target number of tokens per optimizer step (Gradient Accumulation)")
    args = parser.parse_args()
    
    # Create dirs
    import os
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    train(args)
