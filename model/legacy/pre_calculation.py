import pdb
import torch
import torch.nn as nn

from config import cfg
from model.label_prop import HeteroLabelPropagateAlongMP
from model.message_passing import HeteroMessagePassingAlongMP


class PreCalculator(nn.Module):
    def __init__(self, metapaths) -> None:
        super().__init__()
    

        self.convs = torch.nn.ModuleDict()
        self.label_props = torch.nn.ModuleDict()

        for metapath in metapaths:
            str_metapath = '-'.join(metapath)
            self.convs[str_metapath] = HeteroMessagePassingAlongMP(metapath, collect_mode=True)
            self.label_props[str_metapath] = HeteroLabelPropagateAlongMP(metapath)
    
    def forward(self, x, edge_index_dict, y_dict):
        mps = {}
        lps = {}
        for key, conv in self.convs.items():
            mps[key] = conv(x, edge_index_dict)
        for key, label_prop in self.label_props.items():
            lps[key] = label_prop(y_dict, edge_index_dict)
        return mps, lps
