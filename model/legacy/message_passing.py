import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.utils import degree

from torch import Tensor
from torch_geometric.typing import OptPairTensor
from typing import Union
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# Homogenegous message passing.
class OneHopMessagePassing(MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        # Generate the degree based normalization like GCN
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        return out

    def message(self, x_j, norm):
        # Normalize the node features x_j via norm
        return norm.view(-1, 1) * x_j


# Homogeneous message passing of generalized page-rank.
class GPRProp(MessagePassing):
    def __init__(self, K, alpha):
        super(GPRProp, self).__init__(aggr='add')
        self.K = K
        self.alpha = alpha

        gamma = alpha * (1 - alpha)**torch.arange(K + 1)
        gamma[-1] = (1 - alpha)**K

        # self.gamma = nn.parameter.Parameter(gamma)
        self.gamma = gamma

    def reset_parameters(self):
        torch.nn.init.zeros_(self.gamma)
        for k in range(self.K + 1):
            self.gamma.data[k] = self.alpha * (1 - self.alpha)**k
        self.gamma.data[-1] = (1 - self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index,
                                    edge_weight,
                                    num_nodes=x.size(0),
                                    dtype=x.dtype)
        hidden = x * self.gamma[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.gamma[k + 1]
            hidden = hidden + gamma * x

        return hidden

    def message(self, x_j, norm=None):
        return x_j if norm is None else norm.view(-1, 1) * x_j


class HeteroOneHopMessagePassing(MessagePassing):
    def __init__(self, src, dst):
        # super().__init__(flow='target_to_source', node_dim=0)
        super().__init__()
        self.src = src
        self.dst = dst
        # self.lin_l = nn.Linear(64, 64)
        # self.lin_r = nn.Linear(64, 64)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index, norm=False):
       
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if norm:
            edge_index_tmp = edge_index.clone()
            edge_index_tmp[1] += x[0].size(0)
            row, col = edge_index_tmp
            deg = degree(col)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            out = self.propagate(edge_index, x=x, norm=norm)
        else:
            out = self.propagate(edge_index, x=x)

        # out = self.lin_l(out)

        # x_r = x[1]
        # out += self.lin_r(x_r)

        return out

    def message(self, x_j, norm):
        # Normalize the node features x_j via norm
        return norm.view(-1, 1) * x_j

    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(src={self.src}, dst={self.dst})'

# 整个系统需要重新根据relation严格重修。现在MP框架有问题。
class HeteroMessagePassingAlongMP(nn.Module):
    def __init__(self, metapath: list, collect_mode: bool=False) -> None:
        super().__init__()
        self.metapath = metapath
        self.collect_mode = collect_mode 
        self.convs = torch.nn.ModuleList()

        for i, src in enumerate(self.metapath[:-1]):
            dst = self.metapath[i+1]
            # Symmtric message passing.
            conv = HeteroConv({
                (src, 'to', dst): HeteroOneHopMessagePassing(src, dst),
                (dst, 'rev_to', src): HeteroOneHopMessagePassing(dst, src),
            }, aggr='sum')
            self.convs.append(conv)

    # ERROR ON FB-American: no message is processed
    # If in collect_mode, returns all the intermediate target features.
    def forward(self, x_dict, edge_index_dict):
        if self.collect_mode:
            state_lists = [x_dict[self.metapath[0]]]

        for conv in self.convs:
            # Only return transformed target features.
            out_dict = conv(x_dict, edge_index_dict)
            out_dict = {key: x.relu() for key, x in out_dict.items()}
            # Update the target features, while preserve the untrasformed features.
            x_dict.update(out_dict) 
            if self.collect_mode:
                state_lists.append(x_dict[self.metapath[0]])

        if self.collect_mode:
            # Return the final target features.
            return state_lists
        else:
            return x_dict[self.metapath[0]]

    def __repr__(self) -> str:
        return '-'.join(self.metapath)