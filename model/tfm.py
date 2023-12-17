
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NewTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, att_drop=0, activation='none') -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.in_dim, self.in_dim)
        # self.o_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif activation == 'relu':
            self.act = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif activation == 'none':
            self.act = self.plain_activation
        else:
            raise NotImplementedError

        self.reset_parameters()

    def reset_parameters(self):
        for _, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x):

        q = rearrange(self.q_proj(x), "b q (h d) -> b h q d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b k (h d) -> b h k d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b k (h d) -> b h k d", h=self.num_heads)
        scores = torch.einsum("b h q d, b h k d -> b h q k", q, k)
        scores = scores / math.sqrt(self.hidden_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.att_drop(scores)

        out = torch.einsum("b h q k, b h k d -> b h q d", scores, v)
        out = rearrange(out, "b h q d -> b q (h d)").contiguous()

        # out = self.o_proj(out) + x
        out = self.gamma * out + x
        return out

    def plain_activation(self, x):
        return x


class Transformer(nn.Module):
    '''
        The transformer-based semantic fusion.
    '''

    def __init__(self, n_channels, num_heads=1, att_drop=0, act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
  
        self.hidden_dim = n_channels // 4
        assert self.n_channels % (self.num_heads * 4) == 0

        self.q_proj = nn.Linear(self.n_channels, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.n_channels, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.n_channels, self.n_channels, bias=False)

        
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            # self.act = lambda x: x
            self.act = self.plain_activation
        else:
            raise NotImplementedError

        self.reset_parameters()

    def reset_parameters(self):
        for _, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x):
        B, M, C = x.size()  # batchsize, num_metapaths, channels
        H = self.num_heads

        # [B, H, M, -1]
        query = self.q_proj(x).view(B, M, H, -1).permute(0, 2, 1, 3)
        # [B, H, -1, M]
        key = self.k_proj(x).view(B, M, H, -1).permute(0, 2, 3, 1)
        # [B, H, M, -1]
        value = self.v_proj(x).view(B, M, H, -1).permute(0, 2, 1, 3)
        # [B, H, M, M(normalized)]
        atten = torch.matmul(query, key) / math.sqrt(query.size(-1))
        scores = F.softmax(self.act(atten), dim=-1)
        scores = self.att_drop(scores)

        # [B, H, M, -1] -> [B, M, H, -1]  
        o = self.gamma * (scores @ value).permute(0, 2, 1, 3)     
       
        return o.reshape(B, M, C) + x

    def plain_activation(self, x):
        return x
