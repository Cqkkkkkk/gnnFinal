import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F


class WeightedSum(nn.Module):
    def __init__(self, num_nodes) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        init_add = torch.zeros(num_nodes)
        init_add[-1] = 1
        self.weighted_add = nn.parameter.Parameter(init_add)


    def forward(self, x):
        # x: [batch_num x hop_seq_length x feature_dim]
        x = x.transpose(1, 2)
        x = x * self.weighted_add
        x = torch.sum(x, dim=-1, keepdim=False)
        return x


class AttSum(nn.Module):
    def __init__(self, num_nodes, in_dim) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.attn_layer = nn.Linear(2 * self.in_dim, 1)

    def forward(self, x):
        # x: [batch_num x hop_seq_length x feature_dim]
        x_center, x_neighbor = torch.split(x, [1, self.num_nodes - 1], dim=1)
        target = x_center.repeat(1, self.num_nodes - 1, 1)
        
        attn_in = torch.cat([target, x_neighbor], dim=-1)
        attn_score = self.attn_layer(attn_in)
        attn_score = F.softmax(attn_score, dim=1)

        x_neighbor = x_neighbor * attn_score
        x_neighbor = torch.sum(x_neighbor, dim=1)
        
        output = x_center.squeeze(dim=1) + x_neighbor
        return output