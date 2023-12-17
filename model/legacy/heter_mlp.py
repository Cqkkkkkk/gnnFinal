import pdb
import math
import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor
from torch_geometric.nn import Linear
from torch_geometric.typing import NodeType
from config import cfg

from model.selective_linear import SelectiveLinear

def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)



class HeteroMLP(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, input_dropout=0):
        super().__init__()
        self.lins = torch.nn.ModuleDict()
        self.dropout = nn.Dropout(input_dropout)
        for key in in_channels_dict.keys():
            self.lins[key] = Linear(in_channels_dict[key], hidden_channels, bias=False)

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        out_dict = {}
        for key in x_dict:
            out_dict[key] = self.dropout( self.lins[key](x_dict[key]))
        return out_dict

    def reset_parameters(self):
        for lin in self.lins.values():
            lin.weight.data.uniform_(-0.5, 0.5)

class SelectiveHeteroMLP(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels):
        super().__init__()
        self.lins = torch.nn.ModuleDict()
        for key in in_channels_dict.keys():
            self.lins[key] = SelectiveLinear(in_channels_dict[key], hidden_channels)

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        for key in x_dict:
            x_dict[key] = self.lins[key](x_dict[key])
        return x_dict