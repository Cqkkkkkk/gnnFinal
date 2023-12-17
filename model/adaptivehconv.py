import pdb
import torch
import torch.nn as nn
from typing import List, Dict
from torch.nn.parameter import Parameter

from config import cfg


class SimpleAdaptiveHConv(nn.Module):
    """
    A simple implementation of the Adaptive Heterogeneous Convolutional Neural Network.

    Args:
        num_embeddings (int): The number of embeddings.
        alpha (float): A float value for alpha.

    Returns:
        final_embedding (torch.Tensor): A tensor representing the final embedding.
    """

    def __init__(self,
                 num_embeddings,
                 alpha=0.85) -> None:
        super().__init__()

        self.alpha = alpha
        self.num_embeddings = num_embeddings
        if num_embeddings >= 2:
            gamma = alpha * torch.pow((1 - alpha),
                                      torch.arange(num_embeddings))
            gamma[-1] = torch.pow((1 - alpha),
                                  torch.tensor(num_embeddings - 1))
        else:
            gamma = torch.tensor([1.0])
        self.gamma = Parameter(gamma)

    def forward(self, remapped_mp_dict: Dict[str, torch.Tensor]):
        final_embeddings = []
        matmul_mappings = {key: i for i,
                           key in enumerate(remapped_mp_dict.keys())}

        for key, tensor in remapped_mp_dict.items():
            final_embeddings.append(self.gamma[matmul_mappings[key]] * tensor)

        final_embeddings = torch.stack(final_embeddings, dim=0)
        return torch.sum(final_embeddings, dim=0)

    def reset_parameters(self):
        if self.num_embeddings >= 2:
            gamma = self.alpha * \
                torch.pow((1 - self.alpha), torch.arange(self.num_embeddings))
            gamma[-1] = torch.pow((1 - self.alpha),
                                  torch.tensor(self.num_embeddings - 1))
        else:
            gamma = torch.tensor([1.0])
        self.gamma = Parameter(gamma)


# # NOT AVAILABLE YET
# class FullAdaptiveHConv(nn.Module):
#     """
#     A PyTorch module implementing the Full Adaptive Heterogeneous Convolutional layer.

#     Args:
#         metapath (List[str]): A list of node types in the metapath. Example: ['Author', 'Paper', 'Author']
#         num_nodes (int): The number of target nodes in the graph.
#         in_dim (int): The input feature dimension.
#         hidden_dim (int): The hidden feature dimension.
#         num_embeddings (int): The number of embeddings to use.
#         alpha (float): The alpha value for the adaptive weighting scheme.

#     Attributes:
#         conv (HeteroMessagePassingAlongMP): The Heterogeneous Message Passing layer.
#         stage_list (None): A list of stages in the layer.
#         gamma (Parameter): The adaptive weighting scheme parameter.
#         distribution_project_f (nn.Linear): The linear layer for projecting input features for feature distribution.
#         distribution_project_g (nn.Linear): The linear layer for projecting input features for topology distribution.
#     """
#     def __init__(self, metapath: List[str],
#                 num_nodes: int,
#                 num_embeddings: int,
#                 alpha: float) -> None:
#         super().__init__()
#         self.conv = HeteroMessagePassingAlongMP(metapath)
#         self.stage_list = None

#         gamma = alpha * (1 - alpha)**torch.arange(num_embeddings + 1)
#         gamma[-1] = (1 - alpha)**num_embeddings
#         gamma = gamma.unsqueeze(1).repeat(1, num_nodes * num_nodes).reshape(-1, num_nodes, num_nodes)

#         self.gamma = Parameter(gamma)
#         self.adaptive_weight_f = Parameter(torch.tensor([0.5]))

#     def forward(self, x_dict, edge_index_dict, sim_f, sim_t):
#         # Only compute the message-passing once.
#         if not self.state_lists:
#             self.state_lists = self.conv(x_dict, edge_index_dict)
#         distributions = self.adaptive_weight_f * sim_f + (1 - self.adaptive_weight_f) * sim_t

#         final_embeddings = [torch.matmul(self.gamma[i], embedding) + distributions for i, embedding in enumerate(self.state_lists)]
#         # final_embeddings = [torch.matmul(self.gamma[i], embedding) * distributions for i, embedding in enumerate(self.state_lists)]

#         final_embedding = torch.sum(torch.stack(final_embeddings), dim=0)
#         return final_embedding
