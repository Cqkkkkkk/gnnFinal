import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch_geometric.nn import MessagePassing


from torch_geometric.nn.conv.gcn_conv import gcn_norm


# Homogeneous message passing of generalized page-rank.
class GPRProp(MessagePassing):
    """
    Generalized PageRank propagation layer.

    Args:
        K (int): Number of power iterations.
        alpha (float): Damping factor.

    Attributes:
        K (int): Number of power iterations.
        alpha (float): Damping factor.
        gamma (torch.Tensor): Damping coefficients.
    """

    def __init__(self, K, alpha):
        super(GPRProp, self).__init__(aggr='add')
        self.K = K
        self.alpha = alpha

        gamma = alpha * (1 - alpha)**torch.arange(K + 1)
        gamma[-1] = (1 - alpha)**K

        self.gamma = gamma

    def reset_parameters(self):
        """
        Reset the parameters of the GPRProp message passing class.
        """
        torch.nn.init.zeros_(self.gamma)
        for k in range(self.K + 1):
            self.gamma.data[k] = self.alpha * (1 - self.alpha)**k
        self.gamma.data[-1] = (1 - self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        """
        Perform forward propagation using GPRProp.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Graph edge indices.
            edge_weight (torch.Tensor, optional): Edge weights.

        Returns:
            torch.Tensor: Hidden node features after propagation.
        """
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
        """
        Define the message function for GPRProp.

        Args:
            x_j (torch.Tensor): Source node features.
            norm (torch.Tensor, optional): Normalization coefficients.

        Returns:
            torch.Tensor: Message passed from source to target nodes.
        """
        return x_j if norm is None else norm.view(-1, 1) * x_j
