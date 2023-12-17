import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F

from model.readouts import WeightedSum, AttSum
from config import cfg
from model.tfm_block import BaseTFMBlock


class Transformer(nn.Module):

    def __init__(
        self,
        input_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_metapaths: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.5,
        readout: str = 'mean'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_layers
        self.dropout = dropout
        self.readout = readout
        self.input_proj = nn.Linear(input_feature_dim, hidden_dim, bias=False)                                

        if self.readout == 'weighted-sum':
            self.weighted_sum = WeightedSum(num_nodes=num_metapaths)
        elif self.readout == 'att-sum':
            self.att_sum = AttSum(num_nodes=num_metapaths, in_dim=hidden_dim)


        self.transformer = nn.ModuleList([
            BaseTFMBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=self.dropout,
                batched=True
            ) for _ in range(num_layers)
        ])

        self.final_fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        
        bias = None
        x = self.input_proj(x)

        for i in range(self.num_blocks):
            x = self.transformer[i](x_q=x,
                                    x_kv=x, 
                                    bias=bias)


        # Readout 
        if self.readout == 'mean':
            x = torch.mean(x, dim=0, keepdim=False)
        elif self.readout == 'weighted-sum':
            x = self.weighted_sum(x)
        elif self.readout == 'att-sum':
            x = self.att_sum(x)

        # x += x_res[:, -1, :]

        # x += x_res
        x = self.final_fc(x)
        # x = F.log_softmax(x, dim=1)

        return x


    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.hop2token.reset_parameters()
        if cfg.adj.enable:
            self.adj_encoder.reset_parameters()
        for tfm_block in self.transformer:
            tfm_block.reset_parameters()
        self.final_fc.reset_parameters()
