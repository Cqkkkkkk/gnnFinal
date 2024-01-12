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

