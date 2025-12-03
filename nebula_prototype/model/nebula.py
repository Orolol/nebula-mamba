import torch
import torch.nn as nn
from .mamba_block import BiMambaBlock
from .experts import SpatialExpertsLayer
from .diffusion import NebulaDiffusion

class NebulaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([])
        for _ in range(config.num_layers):
            # Interleave Mamba and Experts? Or Mamba then Experts?
            # Standard Transformer: Attention -> AddNorm -> FFN -> AddNorm
            # Nebula: BiMamba -> AddNorm -> SpatialExperts -> AddNorm
            
            layer = nn.ModuleDict({
                'mixer': BiMambaBlock(config.d_model),
                'norm1': nn.LayerNorm(config.d_model),
                'experts': SpatialExpertsLayer(config.d_model, num_heads=config.num_heads),
                'norm2': nn.LayerNorm(config.d_model)
            })
            self.layers.append(layer)
            
        self.final_norm = nn.LayerNorm(config.d_model)
        self.diffusion_head = NebulaDiffusion(config.d_model, config.vocab_size)
        
    def forward(self, input_ids):
        # input_ids: [B, L]
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            # Mixer (BiMamba)
            residual = x
            x = layer['norm1'](x)
            x = layer['mixer'](x)
            x = x + residual
            
            # Experts (MH-SPoE)
            residual = x
            x = layer['norm2'](x)
            x = layer['experts'](x)
            x = x + residual
            
        x = self.final_norm(x)
        logits = self.diffusion_head(x)
        return logits
