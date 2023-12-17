import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import pdb
from model.linears import HeteroMLP, LinearPerMetapath, DownstreamProjection
from model.tfm import Transformer, NewTransformer
from model.adaptivehconv import SimpleAdaptiveHConv
from config import cfg
from utils import parse_metapaths


class AdaptiveHGNN(nn.Module):

    def __init__(self,
                 hidden_dim,
                 out_dim,
                 tgt_type,
                 dropout=0.5,
                 input_dropout=0.5,
                 alpha=0.85,
                 attention_dropout=0,
                 num_attention_heads=1,
                 n_fp_layers=2,
                 n_task_layers=3,
                 residual=True,
                 label_prop=False,
                 selective=False,
                 l2_norm=False,
                 in_mps_dim_dict=None,
                 in_lps_dim_dict=None
                 ):
        super(AdaptiveHGNN, self).__init__()
        self.tgt_type = tgt_type

        self.residual = residual
        self.feat_keys = sorted(in_mps_dim_dict.keys())
        self.num_channels = len(self.feat_keys)
        self.label_prop = label_prop
        self.l2_norm = l2_norm

        self.input_proj_mps = HeteroMLP(
            in_channels_dict=in_mps_dim_dict,
            hidden_channels=hidden_dim, 
            input_dropout=input_dropout,
            selective=selective
        )

        if self.label_prop:
            self.label_feat_keys = sorted(in_lps_dim_dict.keys())
            self.num_channels += len(self.label_feat_keys)
            self.input_proj_lps = HeteroMLP(
                in_lps_dim_dict, hidden_dim, input_dropout=input_dropout)

        self.adaptivehconvs = nn.ModuleDict()
        for key in self.feat_keys:
            self.adaptivehconvs[key] = SimpleAdaptiveHConv(
                num_embeddings=len(key.split('-')),
                alpha=alpha
            )

        self.feature_projection = nn.Sequential(LinearPerMetapath(
            hidden_dim, hidden_dim, self.num_channels))
        for _ in range(n_fp_layers - 1):
            self.feature_projection.append(LinearPerMetapath(
                hidden_dim, hidden_dim, self.num_channels))

        self.semantic_fusion = Transformer(hidden_dim,
                                           num_heads=num_attention_heads,
                                           att_drop=attention_dropout,
                                           act='none')
        # self.semantic_fusion = NewTransformer(hidden_dim,
        #                                       hidden_dim // 4,
        #                                       num_heads=num_attention_heads,
        #                                       att_drop=attention_dropout,
        #                                       activation='none')
        self.fc_after_concat = nn.Linear(
            self.num_channels * hidden_dim, hidden_dim)

        if self.residual:
            self.res_fc = nn.Linear(hidden_dim, hidden_dim)

        self.task_mlp = DownstreamProjection(
            hidden_dim, out_dim, dropout, n_task_layers)

        self.reset_parameters()

        optimizer_split = {
            'normal': [name for name in self._modules.keys() if name != 'semantic_fusion'],
            'tfm': ['semantic_fusion']
        }
        self.parameters_split = {
            'normal': [p for name in optimizer_split['normal'] for p in self._modules[name].parameters()],
            'tfm': [p for name in optimizer_split['tfm'] for p in self._modules[name].parameters()]
        }

    def reset_parameters(self):
        self.input_proj_mps.reset_parameters()
        if self.label_prop:
            self.input_proj_lps.reset_parameters()
        for conv in self.adaptivehconvs.values():
            conv.reset_parameters()
        for lin in self.feature_projection:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        self.semantic_fusion.reset_parameters()
        self.fc_after_concat.reset_parameters()
        if self.res_fc:
            self.res_fc.reset_parameters()
        self.task_mlp.reset_parameters()

    def adaptive_reweighting(self, feature_dict):
        metapaths = list(feature_dict.keys())
        parsed_metapaths = parse_metapaths(metapaths)
        remapped_features = {}
        for key, val in parsed_metapaths.items():
            remapped_features[key] = {k: feature_dict[k] for k in val if k in feature_dict.keys()}
        return remapped_features

    def forward(self, feature_dict, label_dict):
        if isinstance(feature_dict[self.tgt_type], torch.Tensor):
            features = self.input_proj_mps(feature_dict)
        else:
            raise NotImplementedError

        num_tgt_nodes = features[self.tgt_type].shape[0]

        # Adaptive reweighting
        if not cfg.model.disable_simple_adaptive:
            features = self.adaptive_reweighting(features)
            for key, conv in self.adaptivehconvs.items():
                features[key] = conv(features[key])
        else:
            print('SimpleAdaptiveConv disabled')
            
        x = [features[k] for k in self.feat_keys]

        if self.label_prop:
            labels = self.input_proj_lps(label_dict)
            x += [labels[k] for k in self.label_feat_keys]

        x = torch.stack(x, dim=1)  # [B, C, D]
        # project features per metapath [num_nodes, num_metapaths, in_dim] -> [num_nodes, num_metapaths, out_dim]
        x = self.feature_projection(x)

        x = self.semantic_fusion(x).transpose(1, 2)

        x = self.fc_after_concat(x.reshape(num_tgt_nodes, -1))
        if self.residual:
            x = x + self.res_fc(features[self.tgt_type])

        x = self.task_mlp(x)
        if self.l2_norm:
            norm = F.normalize(x, p=2, dim=1)
            x = x / norm
        return x
        