import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba-SSM not installed. Using mock for testing.")
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.linear(x)

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fusion = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        # x: [B, L, D]
        
        # Forward pass
        out_fwd = self.forward_mamba(x)
        
        # Backward pass (flip, process, flip back)
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.backward_mamba(x_rev)
        out_rev = torch.flip(out_rev, dims=[1])
        
        # Fusion
        combined = torch.cat([out_fwd, out_rev], dim=-1)
        return self.fusion(combined)
