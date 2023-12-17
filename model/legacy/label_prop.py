import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv

from config import cfg
from model.message_passing import HeteroOneHopMessagePassing


class HeteroLabelPropagateAlongMP(nn.Module):
    def __init__(self, metapath: list) -> None:
        super().__init__()
        self.metapath = metapath
        self.convs = torch.nn.ModuleList()

        for i, src in enumerate(self.metapath[:-1]):
            dst = self.metapath[i + 1]
            # Symmtric message passing.
            conv = HeteroConv({
                (src, 'to', dst): HeteroOneHopMessagePassing(src, dst),
                (dst, 'to', src): HeteroOneHopMessagePassing(dst, src),
            }, aggr='sum')
            self.convs.append(conv)


    def forward(self, y_dict, edge_index_dict):
        for conv in self.convs:
            # Only return transformed target features.
            out_dict = conv(y_dict, edge_index_dict)
            y_dict.update(out_dict)
            # Update the target features, while preserve the untrasformed features.
        
        return y_dict[self.metapath[0]]

