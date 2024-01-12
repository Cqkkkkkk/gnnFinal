import torch
import torch.nn as nn
from utils import xavier_uniform_

import pdb
import math
import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor
from torch_geometric.nn import Linear
from torch_geometric.typing import NodeType
from config import cfg


class NormedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, dropout) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim, affine=False)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.lin(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out

    def reset_parameters(self):
        self.lin.reset_parameters()


class SelectiveLinear(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.reweight = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        u = self.activation(self.proj(x))
        v = self.sigmoid(self.reweight(x))
        return u * v

    def reset_parameters(self):
        self.proj.weight.data.uniform_(-0.5, 0.5)
        self.reweight.weight.data.uniform_(-0.5, 0.5)


class HeteroMLP(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, input_dropout=0, selective=False):
        super().__init__()
        self.lins = torch.nn.ModuleDict()
        self.dropout = nn.Dropout(input_dropout)
        self.selective = selective
        for key in in_channels_dict.keys():
            if self.selective:
                self.lins[key] = SelectiveLinear(in_channels_dict[key], hidden_channels)
            else:
                self.lins[key] = Linear(in_channels_dict[key], hidden_channels, bias=False)

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        out_dict = {}
        for key in x_dict:
            out_dict[key] = self.dropout(self.lins[key](x_dict[key]))
        return out_dict

    def reset_parameters(self):
        for lin in self.lins.values():
            if self.selective:
                lin.reset_parameters()
            else:
                lin.weight.data.uniform_(-0.5, 0.5)


class SelectiveHeteroMLP(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels):
        super().__init__()
        self.lins = torch.nn.ModuleDict()
        for key in in_channels_dict.keys():
            self.lins[key] = SelectiveLinear(
                in_channels_dict[key], hidden_channels)

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        for key in x_dict:
            x_dict[key] = self.lins[key](x_dict[key])
        return x_dict


class LinearPerMetapath(nn.Module):
    '''
        Linear projection per metapath for feature projection.
    '''

    def __init__(self, in_dim, out_dim, num_metapaths, dropout=0.5):
        super(LinearPerMetapath, self).__init__()
        self.cin = in_dim
        self.cout = out_dim
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(
            self.num_metapaths, self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.cout))

        self.norm = nn.LayerNorm([self.num_metapaths, out_dim])
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        out = torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class DownstreamProjection(nn.Module):
    def __init__(self, hidden_dim, out_dim, dropout, n_task_layers) -> None:
        super().__init__()

        self.task_mlp = nn.Sequential()
        self.task_mlp.append(nn.PReLU())
        self.task_mlp.append(nn.Dropout(dropout))
        self.n_task_layers = n_task_layers

        for _ in range(n_task_layers - 1):
            self.task_mlp.append(NormedLinear(hidden_dim, hidden_dim, dropout))
        
        self.task_mlp.append(nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False, track_running_stats=False)
        ))


    def forward(self, x):
        x = self.task_mlp(x)
        return x

    def reset_parameters(self):
        for lin in self.task_mlp:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        if cfg.model.task.init_last_layer:
            self.task_mlp[-1][0].reset_parameters()
