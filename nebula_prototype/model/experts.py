import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchedExpertPool(nn.Module):
    def __init__(self, d_model, num_experts, expert_dim):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.expert_dim = expert_dim
        
        # Weights: [NumExperts, D_model, Expert_Dim]
        self.w1 = nn.Parameter(torch.randn(num_experts, d_model, expert_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_experts, expert_dim))
        
        # Weights: [NumExperts, Expert_Dim, D_model]
        self.w2 = nn.Parameter(torch.randn(num_experts, expert_dim, d_model) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        
    def forward(self, x, weights):
        """
        x: [B, L, D] (Input to all experts)
        weights: [B, L, NumExperts] (Routing weights)
        """
        # Dense MoE computation: Compute all experts, then weight sum.
        # Efficient for small num_experts (e.g. 4-8).
        
        # 1. Project to Expert Dim: [B, L, E, Expert_Dim]
        # x: [B, L, D] -> [B, L, 1, D]
        # w1: [E, D, H]
        # einsum: bld, edh -> bleh
        h = torch.einsum('bld,edh->bleh', x, self.w1) + self.b1.view(1, 1, self.num_experts, self.expert_dim)
        
        # 2. Activation
        h = F.gelu(h)
        
        # 3. Project back: [B, L, E, D]
        # w2: [E, H, D]
        # einsum: bleh, ehd -> bled
        out = torch.einsum('bleh,ehd->bled', h, self.w2) + self.b2.view(1, 1, self.num_experts, self.d_model)
        
        # 4. Weighted Sum
        # weights: [B, L, E] -> [B, L, E, 1]
        # out: [B, L, E, D]
        # sum over E
        final_out = (out * weights.unsqueeze(-1)).sum(dim=2)
        
        return final_out

class SpatialExpertsLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, num_experts_per_pool=4, expert_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if expert_dim is None:
            expert_dim = self.head_dim * 4 
            
        # Batched Expert Pools (one per head)
        self.expert_pools = nn.ModuleList([
            BatchedExpertPool(self.head_dim, num_experts_per_pool, expert_dim)
            for _ in range(num_heads)
        ])
        
        # Routers (one per head)
        self.routers = nn.ModuleList([
            nn.Linear(self.head_dim, num_experts_per_pool) for _ in range(num_heads)
        ])
        
        # Fusion
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask_info=None):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # Split into heads: [B, L, H, D_h]
        x_heads = x.view(B, L, self.num_heads, self.head_dim)
        
        head_outputs = []
        
        for h in range(self.num_heads):
            # Input for this head: [B, L, D_h]
            h_input = x_heads[:, :, h, :]
            
            # Routing logits: [B, L, NumExperts]
            router_logits = self.routers[h](h_input)
            router_probs = F.softmax(router_logits, dim=-1)
            
            # For Dense Batched MoE, we can just use the probabilities directly (Soft MoE)
            # Or we can mask out non-top-k to keep the sparsity property (Hard MoE)
            
            # Hard MoE (Top-K=2)
            topk_probs, topk_indices = torch.topk(router_probs, k=2, dim=-1)
            
            # Create a sparse weight matrix [B, L, E]
            # Initialize with zeros
            sparse_weights = torch.zeros_like(router_probs)
            # Scatter topk probs
            sparse_weights.scatter_(-1, topk_indices, topk_probs)
            
            # Normalize weights so they sum to 1 (optional, but good for stability)
            sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-6)
            
            # Compute using Batched Pool
            h_out = self.expert_pools[h](h_input, sparse_weights)
            
            head_outputs.append(h_out)
            
        # Concatenate heads: [B, L, D]
        x_fused = torch.cat(head_outputs, dim=-1)
        
        # Final projection
        return self.output_proj(x_fused) + x # Residual connection
