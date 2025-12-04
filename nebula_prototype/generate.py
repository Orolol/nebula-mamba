import torch
import torch.nn.functional as F
import argparse
from transformers import GPT2TokenizerFast
from model.nebula import NebulaModel
from model.config import NEBULA_CONFIGS

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Config & Model
    config = NEBULA_CONFIGS[args.config]
    model = NebulaModel(config).to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    # Note: If model was compiled, state_dict keys might have '_orig_mod' prefix. 
    # For prototype simplicity, we assume standard loading or user handles prefix.
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # Fix for torch.compile: strip '_orig_mod.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        elif "._orig_mod." in k:
            new_state_dict[k.replace("._orig_mod.", ".")] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    mask_token_id = 50256
    
    # Initial State: Fully Masked Sequence
    seq_len = 64 # Generate short sequence
    input_ids = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    
    # Reverse Diffusion (Simplified Iterative Decoding)
    # In a real diffusion model, we'd use a noise scheduler. 
    # Here we do "Mask-Predict-Iterative" (like MaskGIT)
    
    num_steps = 20 # Number of unmasking steps
    print("Generating...")
    
    for step in range(num_steps):
        with torch.no_grad():
            logits = model(input_ids)
            
            # Apply temperature
            logits = logits / args.temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample tokens (Stochastic) instead of Greedy (Argmax)
            # This helps break repetitive loops and adds variety
            pred_ids = torch.multinomial(probs.view(-1, config.vocab_size), 1).view(input_ids.shape)
            
            # Get confidence of the sampled tokens
            max_probs = torch.gather(probs, -1, pred_ids.unsqueeze(-1)).squeeze(-1)
            
            # 2. Determine how many tokens to unmask this step
            # Linear schedule: unmask N/steps tokens per step
            tokens_to_keep = int(seq_len * (step + 1) / num_steps)
            
            # 3. Keep the most confident predictions
            # We want to keep tokens that are ALREADY unmasked + new confident ones
            threshold_idx = torch.topk(max_probs, k=tokens_to_keep, dim=1).indices
            
            # Create new input
            new_input = torch.full_like(input_ids, mask_token_id)
            
            # Scatter predicted tokens at confident positions
            # We need to be careful to preserve batch dim
            for b in range(input_ids.shape[0]):
                indices = threshold_idx[b]
                new_input[b, indices] = pred_ids[b, indices]
            
            input_ids = new_input
            
            # Decode current state
            text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print(f"Step {step+1}/{num_steps}: {text}")

    print("\nFinal Generation:")
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="small", help="Model config used for training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()
    
    generate(args)
