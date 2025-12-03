from dataclasses import dataclass

@dataclass
class NebulaConfig:
    d_model: int
    num_layers: int
    num_heads: int
    vocab_size: int = 50257
    seq_len: int = 1024
    dropout: float = 0.1
    
    # Training params
    lr: float = 3e-4
    batch_size: int = 16
    num_steps: int = 1000

# Presets
NEBULA_CONFIGS = {
    "small": NebulaConfig(
        d_model=256,
        num_layers=4,
        num_heads=4,
        batch_size=32
    ),
    "base": NebulaConfig(
        d_model=512,
        num_layers=12,
        num_heads=8,
        batch_size=16
    ),
    "500M": NebulaConfig(
        d_model=1024,
        num_layers=24, # Adjusted to hit ~500M with d_model=1024
        num_heads=16,
        batch_size=12 # Adjust based on VRAM
    )
}
