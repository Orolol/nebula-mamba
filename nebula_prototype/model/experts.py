import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorizedSpatialExpertsLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, num_experts_per_pool=4, expert_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_experts = num_experts_per_pool
        
        if expert_dim is None:
            expert_dim = self.head_dim * 4 
        self.expert_dim = expert_dim
            
        # Router Weights: [NumHeads, HeadDim, NumExperts]
        self.router_w = nn.Parameter(torch.randn(num_heads, self.head_dim, num_experts_per_pool) * 0.02)
        self.router_b = nn.Parameter(torch.zeros(num_heads, num_experts_per_pool))
        
        # Expert Weights 1: [NumHeads, NumExperts, HeadDim, ExpertDim]
        self.expert_w1 = nn.Parameter(torch.randn(num_heads, num_experts_per_pool, self.head_dim, expert_dim) * 0.02)
        self.expert_b1 = nn.Parameter(torch.zeros(num_heads, num_experts_per_pool, expert_dim))
        
        # Expert Weights 2: [NumHeads, NumExperts, ExpertDim, HeadDim]
        self.expert_w2 = nn.Parameter(torch.randn(num_heads, num_experts_per_pool, expert_dim, self.head_dim) * 0.02)
        self.expert_b2 = nn.Parameter(torch.zeros(num_heads, num_experts_per_pool, self.head_dim))
        
        # Fusion
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask_info=None):
        # x: [B, L, D]
        B, L, D = x.shape
        H = self.num_heads
        E = self.num_experts
        D_h = self.head_dim
        
        # 1. Split into heads: [B, L, H, D_h]
        x_heads = x.view(B, L, H, D_h)
        
        # 2. Routing
        # x_heads: [B, L, H, D_h]
        # router_w: [H, D_h, E]
        # logits: [B, L, H, E]
        router_logits = torch.einsum('blhd,hde->blhe', x_heads, self.router_w) + self.router_b.view(1, 1, H, E)
        router_probs = F.softmax(router_logits, dim=-1) # [B, L, H, E]
        
        # Hard MoE (Top-K=2)
        topk_probs, topk_indices = torch.topk(router_probs, k=2, dim=-1)
        
        # Sparse weights: [B, L, H, E]
        sparse_weights = torch.zeros_like(router_probs)
        sparse_weights.scatter_(-1, topk_indices, topk_probs)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 3. Expert Computation (Vectorized)
        
        # Project to Expert Dim: [B, L, H, E, Expert_Dim]
        # x_heads: [B, L, H, D_h] -> [B, L, H, 1, D_h]
        # expert_w1: [H, E, D_h, Expert_Dim]
        # out: [B, L, H, E, Expert_Dim]
        h = torch.einsum('blhd,hedf->blhef', x_heads, self.expert_w1) + self.expert_b1.view(1, 1, H, E, self.expert_dim)
        
        h = F.gelu(h)
        
        # Project back: [B, L, H, E, D_h]
        # expert_w2: [H, E, Expert_Dim, D_h]
        out = torch.einsum('blhef,hefd->blhed', h, self.expert_w2) + self.expert_b2.view(1, 1, H, E, D_h)
        
        # 4. Weighted Sum (Routing)
        # sparse_weights: [B, L, H, E] -> [B, L, H, E, 1]
        # out: [B, L, H, E, D_h]
        # sum over E -> [B, L, H, D_h]
        head_out = (out * sparse_weights.unsqueeze(-1)).sum(dim=3)
        
        # 5. Fuse Heads
        x_fused = head_out.view(B, L, D)
        
        return self.output_proj(x_fused) + x

# Alias for compatibility
SpatialExpertsLayer = VectorizedSpatialExpertsLayer
