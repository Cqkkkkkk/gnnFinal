import math
import torch
import torch.nn as nn
from model.tfm import Transformer
from utils import parse_metapaths
from model.adaptivehconv import SimpleAdaptiveHConv


class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1: # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False


class AdaptiveHGNN_mag(nn.Module):
    def __init__(self, data_size, nfeat, hidden, nclass,
                 num_feats, num_label_feats, tgt_key,
                 dropout, input_drop, att_drop, label_drop,
                 n_layers_1, n_layers_2, n_layers_3,
                 act, residual=True, bns=True,
                 label_residual=True):
        super().__init__()
        self.residual = residual
        self.tgt_key = tgt_key
        self.label_residual = label_residual

        if any([v != nfeat for k, v in data_size.items()]):
            self.embedings = nn.ParameterDict({})
            for k, v in data_size.items():
                if v != nfeat:
                    self.embedings[k] = nn.Parameter(
                        torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))
        else:
            self.embedings = None
        # pdb.set_trace()
        self.adaptivehconvs = nn.ModuleDict()
        for key in data_size.keys():
            self.adaptivehconvs[key] = SimpleAdaptiveHConv(len(key.split('-')), alpha=0.5)

        self.feat_project_layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_feats, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_feats, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            Conv1d1x1(hidden, hidden, num_feats, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_feats, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        if num_label_feats > 0:
            self.label_feat_project_layers = nn.Sequential(
                Conv1d1x1(nclass, hidden, num_label_feats, bias=True, cformat='channel-first'),
                nn.LayerNorm([num_label_feats, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout),
                Conv1d1x1(hidden, hidden, num_label_feats, bias=True, cformat='channel-first'),
                nn.LayerNorm([num_label_feats, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.label_feat_project_layers = None

        self.semantic_aggr_layers = Transformer(n_channels=hidden, 
                                                num_heads=1,
                                                att_drop=att_drop,
                                                act=act)
        self.concat_project_layer = nn.Linear((num_feats + num_label_feats) * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            if bns:
                return [
                    nn.BatchNorm1d(hidden),
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ]
            else:
                return [
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ]

        lr_output_layers = [
            [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
            for _ in range(n_layers_2-1)]
        self.lr_output = nn.Sequential(*(
            [ele for li in lr_output_layers for ele in li] + [
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass)]))

        if self.label_residual:
            label_fc_layers = [
                [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
                for _ in range(n_layers_3-2)]
            self.label_fc = nn.Sequential(*(
                [nn.Linear(nclass, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns) \
                + [ele for li in label_fc_layers for ele in li] + [nn.Linear(hidden, nclass, bias=True)]))
            self.label_drop = nn.Dropout(label_drop)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.feat_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        if self.label_feat_project_layers is not None:
            for layer in self.label_feat_project_layers:
                if isinstance(layer, Conv1d1x1):
                    layer.reset_parameters()

        
        nn.init.xavier_uniform_(self.concat_project_layer.weight, gain=gain)
        nn.init.zeros_(self.concat_project_layer.bias)

        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        if self.label_residual:
            for layer in self.label_fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def adaptive_reweighting(self, feature_dict):
        metapaths = list(feature_dict.keys())
        parsed_metapaths = parse_metapaths(metapaths)
        remapped_features = {}
        for key, val in parsed_metapaths.items():
            remapped_features[key] = {k: feature_dict[k] for k in val if k in feature_dict.keys()}
        return remapped_features

    def forward(self, feats_dict, layer_feats_dict, label_emb):
        if self.embedings is not None:
            for k, v in feats_dict.items():
                if k in self.embedings:
                    feats_dict[k] = v @ self.embedings[k]
        # pdb.set_trace()

        

        tgt_feat = self.input_drop(feats_dict[self.tgt_key])
        B = num_node = tgt_feat.size(0)

        feats_dict = self.adaptive_reweighting(feats_dict)
        for key, conv in self.adaptivehconvs.items():
            feats_dict[key] = conv(feats_dict[key])
        # pdb.set_trace()
    
        x = self.input_drop(torch.stack(list(feats_dict.values()), dim=1))
        x = self.feat_project_layers(x)

        if self.label_feat_project_layers is not None:
            label_feats = self.input_drop(torch.stack(list(layer_feats_dict.values()), dim=1))
            label_feats = self.label_feat_project_layers(label_feats)
            x = torch.cat((x, label_feats), dim=1)

        x = self.semantic_aggr_layers(x)
        
        x = self.concat_project_layer(x.reshape(B, -1))

        if self.residual:
            x = x + self.res_fc(tgt_feat)
        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        if self.label_residual:
            x = x + self.label_fc(self.label_drop(label_emb))
        return x
