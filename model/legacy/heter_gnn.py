import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv

from config import cfg
from model.heter_mlp import HeteroMLP
from model.message_passing import HeteroOneHopMessagePassing, HeteroMessagePassingAlongMP
from model.label_prop import HeteroLabelPropagateAlongMP


class HeteroGNN(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metapaths):
        super().__init__()

        self.input_proj = HeteroMLP(in_channels_dict, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.label_props = torch.nn.ModuleList()
        self.metapaths = metapaths

        for metapath in metapaths:
            self.convs.append(HeteroMessagePassingAlongMP(metapath))
            self.label_props.append(HeteroLabelPropagateAlongMP(metapath))
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, y_dict):
        x_dict = self.input_proj(x_dict)
        outs = []
        props = []
        for conv in self.convs:
            out = F.relu(conv(x_dict, edge_index_dict))
            outs.append(out)
        
        for label_prop in self.label_props:
            prop = label_prop(y_dict, edge_index_dict)
            props.append(prop)

        # pdb.set_trace()
        outs = torch.stack(outs, dim=0)
        out = torch.mean(outs, dim=0)
        props = torch.stack(props, dim=0)
        prop = torch.mean(props, dim=0)

        # return self.lin(out)
        return prop + self.lin(out)