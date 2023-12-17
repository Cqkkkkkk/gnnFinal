import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear

from config import cfg
from model.adaptivehconv import SimpleAdaptiveHConv
from model.semantic_fusion import Transformer


class AdaptiveHGNN(nn.Module):
    def __init__(self, in_channels_dict, 
                 hidden_channels, 
                 out_channels,
                 metapaths,
                 ):
        super().__init__()

        # self.convs = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleDict()
        self.lps = []
        self.metapaths = metapaths
        for metapath in metapaths:
            # self.convs.append(HeteroMessagePassingAlongMP(metapath))
            str_metapath = '-'.join(metapath)
            self.convs[str_metapath] = SimpleAdaptiveHConv(
                                                   in_channels_dict, 
                                                   hidden_channels,
                                                  len(metapath))
            self.lps.append(metapath)
            

        self.lin = Linear(hidden_channels, out_channels)
        self.lin_lp = Linear(out_channels, out_channels)
        self.tfm = Transformer(
            input_feature_dim=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=out_channels,
            num_metapaths=len(metapaths),
            num_heads=4,
            num_layers=0,
            dropout=0.5,
            readout='mean'
        )


    def forward(self,  pre_calculated_mps, pre_calculated_lps):

        outs = []
        props = []
        # Adaptive Heterogeneous message-passing. Only calculated once during training.
        for key, conv in self.convs.items():
            outs.append(F.relu(conv(pre_calculated_mps[key], key.split('-'))))
            props.append(pre_calculated_lps[key])
        
        # Transformer-based (or attention based) intra-metapath aggregation.

        outs = torch.stack(outs, dim=0) # (num_metapaths, num_tgt_nodes, hidden_channels)
        out = torch.mean(outs, dim=0)
        # out = self.tfm(outs)

        out = self.lin(out)

        props = torch.stack(props, dim=0)
        prop = torch.mean(props, dim=0)
        prop = self.lin_lp(prop)
        return F.log_softmax(out + prop, dim=-1)
        # return self.lin(x_dict['author'])
        # return prop + out
        return out