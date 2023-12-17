import pdb
import torch
import torch.nn as nn


def l21_norm(weight):
    return torch.sum(torch.sqrt(torch.sum(weight**2, dim=1) + 1e-8))  # 1e-8 for numerical stability

class SelectiveLinear(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.lin_proj = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.lin_reweight = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        u = self.activation(self.lin_proj(x))
        v = self.sigmoid(self.lin_reweight(x))
        return u * v