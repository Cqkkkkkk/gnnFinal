import pdb
import torch
import torch.nn as nn
from typing import List, Dict
from torch.nn.parameter import Parameter

from config import cfg
from model.message_passing import HeteroMessagePassingAlongMP


class SimpleAdaptiveHConv(nn.Module):
    """
    A simple implementation of the Adaptive Heterogeneous Convolutional Neural Network.

    Args:
        metapath (List[str]): A list of strings representing the metapath.
        num_embeddings (int): The number of embeddings.
        alpha (float): A float value for alpha.

    Returns:
        final_embedding (torch.Tensor): A tensor representing the final embedding.
    """
    def __init__(self, 
                 in_dims_dict,
                 hidden_dim,
                 num_embeddings, 
                 alpha=0.85) -> None:
        super().__init__()

        self.input_proj = nn.ModuleDict()
        for key, value in in_dims_dict.items():
            self.input_proj[key] = nn.Linear(value, hidden_dim)

        gamma = alpha * (1 - alpha)**torch.arange(num_embeddings + 1)
        gamma[-1] = (1 - alpha)**num_embeddings
        self.gamma = Parameter(gamma)
        # self.gamma = gamma

    
    def forward(self, pre_calculated_mp_list: List[torch.Tensor], mp_list_type: List[str]):
        transformed_mp_list = []
        for mp, mp_type in zip(pre_calculated_mp_list, mp_list_type):
            try:
                transformed_mp_list.append(self.input_proj[mp_type](mp))
            except:
                pdb.set_trace()
        final_embeddings = [self.gamma[i] * embedding for i, embedding in enumerate(transformed_mp_list)]
        final_embedding = torch.sum(torch.stack(final_embeddings), dim=0)
        return final_embedding
    

class FullAdaptiveHConv(nn.Module):
    """
    A PyTorch module implementing the Full Adaptive Heterogeneous Convolutional layer.

    Args:
        metapath (List[str]): A list of node types in the metapath. Example: ['Author', 'Paper', 'Author']
        num_nodes (int): The number of target nodes in the graph.
        in_dim (int): The input feature dimension.
        hidden_dim (int): The hidden feature dimension.
        num_embeddings (int): The number of embeddings to use.
        alpha (float): The alpha value for the adaptive weighting scheme.

    Attributes:
        conv (HeteroMessagePassingAlongMP): The Heterogeneous Message Passing layer.
        stage_list (None): A list of stages in the layer.
        gamma (Parameter): The adaptive weighting scheme parameter.
        distribution_project_f (nn.Linear): The linear layer for projecting input features for feature distribution.
        distribution_project_g (nn.Linear): The linear layer for projecting input features for topology distribution.
    """
    def __init__(self, metapath: List[str],
                num_nodes: int,
                num_embeddings: int, 
                alpha: float) -> None:
        super().__init__()
        self.conv = HeteroMessagePassingAlongMP(metapath)
        self.stage_list = None

        gamma = alpha * (1 - alpha)**torch.arange(num_embeddings + 1)
        gamma[-1] = (1 - alpha)**num_embeddings
        gamma = gamma.unsqueeze(1).repeat(1, num_nodes * num_nodes).reshape(-1, num_nodes, num_nodes)

        self.gamma = Parameter(gamma)
        self.adaptive_weight_f = Parameter(torch.tensor([0.5]))

    def forward(self, x_dict, edge_index_dict, sim_f, sim_t):
        # Only compute the message-passing once.
        if not self.state_lists:
            self.state_lists = self.conv(x_dict, edge_index_dict)
        distributions = self.adaptive_weight_f * sim_f + (1 - self.adaptive_weight_f) * sim_t

        final_embeddings = [torch.matmul(self.gamma[i], embedding) + distributions for i, embedding in enumerate(self.state_lists)]
        # final_embeddings = [torch.matmul(self.gamma[i], embedding) * distributions for i, embedding in enumerate(self.state_lists)]

        final_embedding = torch.sum(torch.stack(final_embeddings), dim=0)
        return final_embedding