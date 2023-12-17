import pdb
import math
import torch
import numpy as np
import torch.nn.functional as F
from config import cfg

import torch
import torch_geometric
import pdb
from typing import Any, Union
from typing import Union, Any
from config import cfg
from torch_sparse import SparseTensor
from ogb.nodeproppred import Evaluator
import dgl.function as fn
import gc
import matplotlib.pyplot as plt


def construct_onehot_label(batch):
    y_dict = {}
    for key, _ in batch.x_dict.items():
        if key == cfg.dataset.tgt_type:
            y_dict[key] = F.one_hot(
                batch.y, num_classes=cfg.dataset.num_classes).float().to(cfg.device)
            # Set non-training nodes' labels to 0.
            # y_dict[key][batch[cfg.dataset.tgt_type].batch_size: ] = 0
            y_dict[key][~batch[cfg.dataset.tgt_type]['train_mask']] = 0
        else:
            y_dict[key] = torch.zeros((batch[key]['x'].size(
                0), cfg.dataset.num_classes)).to(cfg.device)
    return y_dict


def set_seed_global(seed: int, force_deter=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if force_deter:
        torch.use_deterministic_algorithms(True)
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def cal_acc(pred, label):
    return (pred == label).sum() / len(label)


def generate_random_splits(num_nodes, train_ratio, val_ratio):
    set_seed_global(cfg.seed)
    print('[Warning] Using non-public split')
    test_ratio = 1 - train_ratio - val_ratio
    train_mask = torch.full((num_nodes, ), False, dtype=torch.bool)
    val_mask = torch.full((num_nodes, ), False, dtype=torch.bool)
    test_mask = torch.full((num_nodes, ), False, dtype=torch.bool)

    permute = torch.randperm(num_nodes)
    train_idx = permute[: int(train_ratio * num_nodes)]
    val_idx = permute[int(train_ratio * num_nodes)
                          : int((train_ratio + val_ratio) * num_nodes)]
    test_idx = permute[int(1 - test_ratio * num_nodes):]
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def cal_edge_homophily(edge_index, labels):
    cnt = 0
    for u, v in edge_index.T:
        # u = u.item()
        # v = v.item()
        if labels[u] == labels[v]:
            cnt += 1
    return cnt / len(edge_index.T)


def cal_local_edge_homophily(edge_index, labels):
    cnt = np.zeros_like(labels)
    total = np.zeros_like(labels, dtype=np.float32)
    total.fill(1e-8)
    for u, v in edge_index.T:
        # u = u.item()
        # v = v.item()
        if labels[u] == labels[v]:
            cnt[u] += 1
            cnt[v] += 1
        total[u] += 1
        total[v] += 1
    result = cnt / total
    result = result[total >= 1]
    return result

def draw_local_edge_homophily_distribution(result, name, bins=10):
    bins = np.linspace(0, 1, 6)

    plt.figure(figsize=(8, 6))
    counts, _, bars = plt.hist(result, bins=bins, edgecolor='black')

    # Adding the frequency on top of each bar
    for bar, freq in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(freq), ha='center', va='bottom')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.xticks(bins)
    plt.savefig(f'plots/actor/local_homo-{name}.png', dpi=300)


def cal_hetero_connection(edge_index1, edge_index2):
    edge2_dict = {dst.item(): src.item() for dst, src in zip(*edge_index2)}
    mp_list = []

    for src1, dst in zip(*edge_index1):
        if dst in edge2_dict:
            src2 = edge2_dict[dst]
            if src1 != src2:
                mp_list.append((src1, src2))

    mp_list = np.array(mp_list)
    mp_list = np.unique(mp_list, axis=0).T
    return mp_list


# Improved edge-homophily count
def cal_improved_edge_homophily(edge_index, labels):

    num_classes = np.unique(labels).shape[0]

    node_cnt_per_class = np.zeros(num_classes)
    homo_node_cnt_per_class = np.zeros(num_classes)

    for u, v in edge_index.T:
        if labels[u] == labels[v]:
            homo_node_cnt_per_class[labels[u]] += 1
        node_cnt_per_class[labels[u]] += 1

    # pdb.set_trace()
    # h_ks = cnt_homo_classes / cnt_classes
    total_nodes = np.sum(node_cnt_per_class)
    h_hat = 0

    for cnt, ck in zip(homo_node_cnt_per_class, node_cnt_per_class):
        hk = cnt / ck
        h_hat = h_hat + np.max(hk - (ck / total_nodes), 0)

    h_hat /= (num_classes - 1)
    return h_hat


# Example: metapath = ['author', 'paper', 'author', 'paper', 'author']
def cal_hetero_edge_homophily(edge_index_dict, labels, metapath, sep='to', local=False):
    assert metapath[0] == metapath[-1]
    assert len(metapath) >= 2
    mp_name = '-'.join(metapath)

    if len(metapath) == 2:
        if local:
            result = cal_local_edge_homophily(edge_index_dict[(metapath[0], sep, metapath[1])], labels)
            draw_local_edge_homophily_distribution(result, '-'.join(metapath), bins=5)
        else:
            result = cal_edge_homophily(
                edge_index=edge_index_dict[(metapath[0], sep, metapath[1])], 
                labels=labels
            )
            print(f'Metapath( {mp_name} ) indexed edge homophily: {result:.2f}')
        return result

    src_type_start = metapath[0]
    dst_type1 = metapath[1]
    dst_type2 = metapath[2]
    edge_index1 = edge_index_dict[(src_type_start, sep, dst_type1)].numpy()
    edge_index2 = edge_index_dict[(dst_type1, sep, dst_type2)].numpy()
    merged_index = cal_hetero_connection(edge_index1, edge_index2)
    last_dst_type = dst_type2

    for cur_type in metapath[3:]:
        new_index = edge_index_dict[(last_dst_type, sep, cur_type)].numpy()
        merged_index = cal_hetero_connection(merged_index, new_index)
        last_dst_type = cur_type
        # edge_index_dict[(src_type_start, 'to', merged_type)] = torch.from_numpy(merged_index)

    mp_list = merged_index
    if local:
       result = cal_local_edge_homophily(mp_list, labels)
       draw_local_edge_homophily_distribution(result, mp_name, bins=5)
    else:
        result = cal_edge_homophily(mp_list, labels)
        print(f'Metapath( {mp_name} ) indexed edge homophily: {result:.2f}')
    return result


# Example: metapath = ['author', 'paper', 'author', 'paper', 'author']
def cal_improved_hetero_edge_homophily(edge_index_dict, labels, metapath, sep='to'):
    assert metapath[0] == metapath[-1]
    assert len(metapath) >= 2

    if len(metapath) == 2:
        result = cal_improved_edge_homophily(
            edge_index_dict[(metapath[0], sep, metapath[1])], labels)
        print('Metapath(', '-'.join(metapath),
              f') indexed improved edge homophily: {result:.2f}')
        return result

    src_type_start = metapath[0]
    dst_type1 = metapath[1]
    dst_type2 = metapath[2]
    edge_index1 = edge_index_dict[(src_type_start, sep, dst_type1)].numpy()
    edge_index2 = edge_index_dict[(dst_type1, sep, dst_type2)].numpy()
    merged_index = cal_hetero_connection(edge_index1, edge_index2)
    last_dst_type = dst_type2

    for cur_type in metapath[3:]:
        new_index = edge_index_dict[(last_dst_type, sep, cur_type)].numpy()
        merged_index = cal_hetero_connection(merged_index, new_index)
        last_dst_type = cur_type
        # edge_index_dict[(src_type_start, 'to', merged_type)] = torch.from_numpy(merged_index)

    mp_list = merged_index
    result = cal_improved_edge_homophily(mp_list, labels)
    print('Metapath(', '-'.join(metapath),
          f') indexed improved edge homophily: {result:.2f}')
    return result


def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # Calculate uniform bounds from standard deviation
    a = math.sqrt(3.0) * std
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


def l21_norm(weight):
    # 1e-8 for numerical stability
    return torch.sum(torch.sqrt(torch.sum(weight**2, dim=1) + 1e-8))


def to_dgl(
    data: Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData']
) -> Any:
    r"""Converts a :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
    object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The data object.

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
        >>> x = torch.randn(5, 3)
        >>> edge_attr = torch.randn(6, 2)
        >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes=5, num_edges=6,
            ndata_schemes={'x': Scheme(shape=(3,))}
            edata_schemes={'edge_attr': Scheme(shape=(2, ))})

        >>> data = HeteroData()
        >>> data['paper'].x = torch.randn(5, 3)
        >>> data['author'].x = torch.ones(5, 3)
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        >>> data['author', 'cites', 'paper'].edge_index = edge_index
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes={'author': 5, 'paper': 5},
            num_edges={('author', 'cites', 'paper'): 5},
            metagraph=[('author', 'paper', 'cites')])
    """
    import dgl

    from torch_geometric.data import Data, HeteroData

    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        else:
            row, col, _ = data.adj_t.t().coo()

        g = dgl.graph((row, col))

        for attr in data.node_attrs():
            g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, store in data.edge_items():
            if store.get('edge_index') is not None:
                row, col = store.edge_index
            else:
                row, col, _ = store['adj_t'].t().coo()
            new_edge_type = (
                edge_type[0], f'{edge_type[0]}-{edge_type[2]}', edge_type[2])
            data_dict[new_edge_type] = (row, col)

        g = dgl.heterograph(data_dict, num_nodes_dict=data.num_nodes_dict)
        for node_type, store in data.node_items():
            for attr, value in store.items():
                if attr != 'x':
                    continue
                attr = node_type
                g.nodes[node_type].data[attr] = value

        for edge_type, store in data.edge_items():
            for attr, value in store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")


def edge_index_to_sparse_tensor(edge_index, num_nodes_src, num_nodes_dst):
    """
    Convert edge index to a sparse tensor.

    Args:
        edge_index (torch.Tensor): The edge index tensor of shape (2, num_edges).
        num_nodes_src (int): The number of source nodes.
        num_nodes_dst (int): The number of destination nodes.

    Returns:
        torch_sparse.SparseTensor: The sparse tensor representation of the edge index.
    """
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    sparse_size = (num_nodes_src, num_nodes_dst)
    adj = SparseTensor(row=torch.LongTensor(
        row), col=torch.LongTensor(col), sparse_sizes=sparse_size)
    return adj


def list_contain(fatherlists, sublist):
    for fatherlist in fatherlists:
        if ''.join(sublist) in ''.join(fatherlist):
            return True
    return False


def recursive_one_step_shorter(parsed_metapath, parsed_metapaths):
    one_step_shorter_mp = parsed_metapath[:-1]
    results = []
    if len(one_step_shorter_mp) == 0:
        return results
    else:
        if list_contain(parsed_metapaths, one_step_shorter_mp):
            # print(f'For mp {parsed_metapath} there exists step shorter mp {one_step_shorter_mp}')
            results += ['-'.join(one_step_shorter_mp)]
        return results + recursive_one_step_shorter(one_step_shorter_mp, parsed_metapaths)


# Group the metapaths by their origin
def parse_metapaths(metapaths):
    # Sort the metapaths in reverse length order
    parsed_metapaths = sorted(
        [mp.split('-') for mp in metapaths], key=lambda x: len(x), reverse=True)
    results = {key: [key] for key in metapaths}

    for parsed_metapath in parsed_metapaths:

        recur_parse_result = recursive_one_step_shorter(
            parsed_metapath, parsed_metapaths)
        results['-'.join(parsed_metapath)] += (recur_parse_result)

    return results


def dict_to_device(data, device):
	for key, val in data.items():
		data[key] = val.to(device)
	return data


def cal_adj(edge_index, num_nodes_src, num_nodes_dst):
	adj = edge_index_to_sparse_tensor(edge_index, num_nodes_src, num_nodes_dst)
	adj.storage._value = None
	adj.storage._value = torch.ones(adj.nnz()) / adj.sum(dim=-1)[adj.storage.row()]
	return adj


def fullname2short(fullname):
    return ''.join([name[0].upper() for name in fullname.split('-')])


def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]



def clear_hg(new_g, echo=False):
	if echo: print('Remove keys left after propagation')
	for ntype in new_g.ntypes:
		keys = list(new_g.nodes[ntype].data.keys())
		if len(keys):
			if echo: print(ntype, keys)
			for k in keys:
				new_g.nodes[ntype].data.pop(k)
	return new_g


def hg_propagate(new_g, tgt_type, num_hops, max_hops, echo=False):
	for hop in range(1, max_hops):
		for etype in new_g.etypes:
			stype, _, dtype = new_g.to_canonical_etype(etype)

			for k in list(new_g.nodes[stype].data.keys()):
				if len(k.split('-')) == hop:
					current_dst_name = f'{dtype}-{k}'
					if hop == num_hops and dtype != tgt_type or hop > num_hops :
						continue
					if echo: print(k, etype, current_dst_name)
					new_g[etype].update_all(
						fn.copy_u(k, 'm'),
						fn.mean('m', current_dst_name), etype=etype)

		# remove no-use items
		for ntype in new_g.ntypes:
			if ntype == tgt_type: continue
			removes = []
			for k in new_g.nodes[ntype].data.keys():
				if len(k.split('-')) <= hop:
					removes.append(k)
			for k in removes:
				new_g.nodes[ntype].data.pop(k)
			if echo and len(removes): print('remove', removes)
		gc.collect()

		if echo: print(f'-- hop={hop} ---')
		for ntype in new_g.ntypes:
			for k, v in new_g.nodes[ntype].data.items():
				if echo: print(f'{ntype} {k} {v.shape}')
		if echo: print(f'------\n')

	return new_g