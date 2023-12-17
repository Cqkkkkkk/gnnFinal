import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from typing import Optional, List
from config import cfg

# hidden_dim -> hidden_dim -> hidden_dim
class BaseTFMBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 batched: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.h = num_heads
        self.d_k = hidden_dim // num_heads
        self.batched = batched
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Currently not used
        # self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.activation = F.gelu

    def reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.o_proj.reset_parameters()

    def forward(self, x_q: Tensor, x_kv: Tensor, bias: Optional[Tensor]) -> Tensor:
        if self.batched:
            q = rearrange(self.q_proj(x_q), "b q (h d) -> b h q d", h=self.h)
            k = rearrange(self.k_proj(x_kv), "b k (h d) -> b h k d", h=self.h)
            v = rearrange(self.v_proj(x_kv), "b k (h d) -> b h k d", h=self.h)
            scores = torch.einsum("b h q d, b h k d -> b h q k", q, k)
        else:
            q = rearrange(self.q_proj(x_q), "n (h d) -> h n d", h=self.h)
            k = rearrange(self.k_proj(x_kv), "n (h d) -> h n d", h=self.h)
            v = rearrange(self.v_proj(x_kv), "n (h d) -> h n d", h=self.h)
            scores = torch.einsum("h q d, h k d -> h q k", q, k)

        scores = scores / math.sqrt(self.d_k)

        if bias is not None:
            scores = scores + bias  # torch.Size([N, N])

        scores = F.softmax(scores, dim=-1)

        scores = F.dropout(scores, p=self.dropout, training=self.training)

        if self.batched:
            x = torch.einsum("b h q k, b h k d -> b h q d", scores, v)
            x = rearrange(x, "b h q d -> b q (h d)").contiguous()
        else:
            x = torch.einsum("h q k, h k d -> h q d", scores, v)
            x = rearrange(x, "h q d -> q (h d)").contiguous()

        x = self.o_proj(x)

        # residual connection and dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x_q + x

        # currently not used
        # x2 = self.linear2(self.dropout[2](self.activation(
        #     self.linear1(self.norm1(x)))))
        # x = x + self.dropout[3](x2)
        # x = self.norm2(x)
        return x

