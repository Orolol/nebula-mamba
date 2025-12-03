import torch
import torch.nn as nn
import numpy as np

class DiffusionHead(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward_diffusion(self, input_ids, mask_token_id):
        """
        Applies random masking to input_ids based on a random timestep t.
        Returns: masked_input_ids, labels (original ids), mask_mask (1 where masked)
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Sample t ~ Uniform(0, 1) for each sample in batch
        t = torch.rand(B, device=device)
        
        # Masking probability schedule (e.g., cosine or linear)
        # Simple linear: p_mask = t
        p_mask = t.unsqueeze(-1).expand(B, L)
        
        # Generate mask
        rand_mask = torch.rand(B, L, device=device)
        mask_mask = (rand_mask < p_mask)
        
        # Apply mask
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_mask] = mask_token_id
        
        # Labels: we only compute loss on masked tokens
        labels = input_ids.clone()
        labels[~mask_mask] = -100 # Ignore unmasked tokens
        
        # Also ignore padding tokens in the original input
        # Assuming pad_token_id is 50256 (same as mask/eos for GPT-2)
        pad_token_id = 50256
        labels[input_ids == pad_token_id] = -100
        
        return masked_input_ids, labels, mask_mask

class NebulaDiffusion(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.head(x)
