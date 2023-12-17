import os
import gc
import pdb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any
from torch_sparse import remove_diag

from config import cfg
from utils import edge_index_to_sparse_tensor, hg_propagate, to_dgl, clear_hg, fullname2short


class HeteroLabelPropagation:
	def __init__(self) -> None:
		pass

	def __call__(self, data, tgt_type, num_hops, verbose=False) -> Any:
		if verbose:
			print(f'Generating {self.num_hops}-hop label propagation')

		assert len(data.node_types) == len(data.x_dict.keys()), \
			'ERROR: Missing feature for some node types'

		adjs = self.cal_adjs(data)

		train_nid = torch.where(data[tgt_type].train_mask)
		init_labels = data[tgt_type].y
		num_classes = torch.unique(init_labels).numel()
		num_tgt_node = init_labels.shape[0]

		propagated_labels = self.propagate_label(
			num_label_hops=num_hops,
			train_nid=train_nid,
			init_labels=init_labels,
			num_nodes=num_tgt_node,
			num_classes=num_classes,
			tgt_type=tgt_type,
			adjs=adjs,
			prop_device='cpu'
		)

		propagated_labels = {k: v.to(cfg.device).detach(
		).clone() for k, v in propagated_labels.items()}

		return propagated_labels

	def cal_adjs(self, data):
		num_nodes = {k: v.size(0) for k, v in data.x_dict.items()}
		adjs = {}
		for (src, _, dst), edge_index in data.edge_index_dict.items():
			adj = edge_index_to_sparse_tensor(
				edge_index, num_nodes[src], num_nodes[dst])
			adjs[f'{src}-{dst}'] = adj

		# Row normalization
		for k in adjs.keys():
			adjs[k].storage._value = None
			adjs[k].storage._value = torch.ones(
				adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]
		return adjs

	def propagate_label(self, num_label_hops, train_nid, init_labels, num_nodes, num_classes, tgt_type, adjs, prop_device, verbose=False):
		label_feats = {}

		label_onehot = torch.zeros((num_nodes, num_classes))
		label_onehot[train_nid] = F.one_hot(
			init_labels[train_nid], num_classes).float()

		max_length = num_label_hops + 1

		meta_adjs = self.hg_propagate_sparse_pyg(
			adjs, tgt_type, num_label_hops, max_length, verbose=verbose, prop_device=prop_device)

		if verbose:
			print(f'For label propagation, meta_adjs: (in SparseTensor mode)')
			for k, v in meta_adjs.items():
				print(k, v.sizes())

		for k, v in meta_adjs.items():
			label_feats[k] = remove_diag(v) @ label_onehot
			gc.collect()

		print(f'[LP] Involved {len(label_feats.keys())} label prooagation:')
		if verbose:
			for key, val in label_feats.items():
				print(f'[LP] Metapath {key} with embedding of shape {val.shape}')

		return label_feats

	def hg_propagate_sparse_pyg(self, adjs, tgt_types, num_hops, max_length, verbose=False, prop_device='cpu'):

		store_device = 'cpu'
		if not isinstance(tgt_types, list):
			tgt_types = [tgt_types]

		# metapath should start with target type in label propagation
		label_feats = {k: v.clone() for k, v in adjs.items()
					   if k.split('-')[-1] in tgt_types}
		adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

		for hop in range(2, max_length):
			new_adjs = {}
			for rtype_r, adj_r in label_feats.items():
				metapath_types = rtype_r.split('-')
				if len(metapath_types) != hop:
					continue
				dtype_r, _ = metapath_types[0], metapath_types[-1]
				for rtype_l, adj_l in adjs_g.items():
					dtype_l, stype_l = rtype_l.split('-')
					if stype_l == dtype_r:
						name = f'{dtype_l}-{rtype_r}'
						if (hop == num_hops and dtype_l not in tgt_types):
							continue

						if name not in new_adjs:
							if verbose:
								print('[LP] Generating ...', name)
							if prop_device == 'cpu':
								new_adjs[name] = adj_l.matmul(adj_r)
							else:
								with torch.no_grad():
									new_adjs[name] = adj_l.matmul(
										adj_r.to(prop_device)).to(store_device)
						else:
							if verbose:
								print(f'Warning: {name} already exists')
			label_feats.update(new_adjs)

			removes = []
			for k in label_feats.keys():
				metapath_types = k.split('-')
				if metapath_types[0] in tgt_types:
					continue  # metapath should end with target type in label propagation
				if len(metapath_types) <= hop:
					removes.append(k)
			for k in removes:
				label_feats.pop(k)
			if verbose and len(removes):
				print('[LP] Remove', removes)
			del new_adjs
			gc.collect()

		if prop_device != 'cpu':
			del adjs_g
			torch.cuda.empty_cache()

		return label_feats


class HeteroLabelPropagationDense:
	def __init__(self) -> None:
		pass
	
	def __call__(self, data, label_onehot, num_nodes, num_classes, num_hops, tgt_type, verbose=False) -> Any:
		g = to_dgl(data)
		g = clear_hg(g)
		g.nodes[tgt_type].data[tgt_type] = label_onehot
		g = hg_propagate(g, tgt_type, num_hops, num_hops + 1, echo=verbose)
		
		lps = {}
		keys = list(g.nodes[cfg.dataset.tgt_type].data.keys())
		print(f'Involved label keys {keys}')
		for k in keys:
			if k == cfg.dataset.tgt_type: 
				continue
			lps[k] = g.nodes[cfg.dataset.tgt_type].data.pop(k)

		del g

		label_emb = torch.zeros((num_nodes, num_classes))
		for key in lps.keys():
			short_key = fullname2short(key)
			if short_key in ['PAP', 'PFP', 'PPP']:
				diag = torch.load(f'datasets/ogb-mag/mag/processed/{short_key}_diag.pt')
				lps[key] = lps[key] - diag.unsqueeze(-1) * label_onehot
				label_emb += lps[key]

		label_emb += lps['paper-paper']
		label_emb = label_emb / 4
		
		return lps, label_emb
