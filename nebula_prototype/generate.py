import torch
import torch.nn.functional as F
import argparse
from transformers import GPT2TokenizerFast
from model.nebula import NebulaModel
from model.config import NEBULA_CONFIGS

# Prompts de départ prédéfinis
DEFAULT_PROMPTS = [
    "The quick brown fox",
    "In the beginning",
    "Once upon a time",
    "The meaning of life is",
    "Scientists have discovered that",
    "Breaking news:",
    "The future of artificial intelligence",
    "In a world where",
    "The secret to happiness is",
    "According to recent studies",
]

def load_model(args):
    """Load model and tokenizer once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = NEBULA_CONFIGS[args.config]
    model = NebulaModel(config).to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
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

    return model, tokenizer, config, device


def generate_one(model, tokenizer, config, device, args, prompt=None):
    """Generate a single sequence."""
    mask_token_id = 50256
    seq_len = args.seq_len

    # Initialiser la séquence
    if prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        if prompt_len >= seq_len:
            prompt_ids = prompt_ids[:seq_len - 1]
            prompt_len = len(prompt_ids)

        input_ids = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
        input_ids[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

        fixed_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        fixed_mask[:prompt_len] = True
    else:
        input_ids = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
        fixed_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    num_steps = args.num_steps

    # Nombre de tokens à générer (exclut les tokens fixes du prompt)
    num_fixed = fixed_mask.sum().item()
    num_to_generate = seq_len - num_fixed

    for step in range(num_steps):
        with torch.no_grad():
            logits = model(input_ids)

            # Apply temperature
            logits = logits / args.temperature
            probs = F.softmax(logits, dim=-1)

            # Sample tokens (Stochastic)
            pred_ids = torch.multinomial(probs.view(-1, config.vocab_size), 1).view(input_ids.shape)

            # Get confidence of the sampled tokens
            max_probs = torch.gather(probs, -1, pred_ids.unsqueeze(-1)).squeeze(-1)

            # Pour les positions fixes (prompt), on force une probabilité très haute
            # pour qu'elles soient toujours gardées
            max_probs[0, fixed_mask] = 1.0

            # Determine how many tokens to keep this step
            # On doit garder au minimum les tokens fixes + proportion des tokens générés
            generated_to_keep = int(num_to_generate * (step + 1) / num_steps)
            tokens_to_keep = min(num_fixed + generated_to_keep, seq_len)

            # Keep the most confident predictions
            threshold_idx = torch.topk(max_probs, k=tokens_to_keep, dim=1).indices

            # Create new input
            new_input = torch.full_like(input_ids, mask_token_id)

            # Scatter predicted tokens at confident positions
            for b in range(input_ids.shape[0]):
                indices = threshold_idx[b]
                new_input[b, indices] = pred_ids[b, indices]

            # Toujours préserver les tokens du prompt
            new_input[0, fixed_mask] = input_ids[0, fixed_mask]

            input_ids = new_input

            # Decode current state
            if args.show_steps:
                text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                print(f"Step {step+1}/{num_steps}: {text}")

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with Nebula model")
    parser.add_argument("--config", type=str, default="small", help="Model config used for training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--prompt", type=str, default=None, help="Starting prompt for generation")
    parser.add_argument("--random-prompt", action="store_true", help="Use a random predefined prompt")
    parser.add_argument("--seq-len", type=int, default=64, help="Total sequence length to generate (default: 64)")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of denoising steps (default: 20)")
    parser.add_argument("--show-steps", action="store_true", help="Show intermediate generation steps")
    parser.add_argument("--num-generations", type=int, default=1, help="Number of generations to produce (default: 1)")
    parser.add_argument("--list-prompts", action="store_true", help="List available default prompts and exit")
    args = parser.parse_args()

    if args.list_prompts:
        print("Available default prompts:")
        for i, p in enumerate(DEFAULT_PROMPTS):
            print(f"  {i+1}. {p}")
        exit(0)

    # Load model once
    model, tokenizer, config, device = load_model(args)

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    elif args.random_prompt:
        import random
        prompt = random.choice(DEFAULT_PROMPTS)
    else:
        prompt = None

    if prompt:
        print(f"Prompt: \"{prompt}\"")

    # Generate multiple sequences
    for i in range(args.num_generations):
        if args.num_generations > 1:
            print(f"\n{'='*50}")
            print(f"Generation {i+1}/{args.num_generations}")
            print('='*50)

        result = generate_one(model, tokenizer, config, device, args, prompt)
        print(f"\nResult: {result}")
