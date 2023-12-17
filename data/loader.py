import pdb
import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from config import cfg

from utils import generate_random_splits


def construct_loader(data):
    if cfg.dataset.split == 'random':
        num_target_nodes = data[cfg.dataset.tgt_type]['x'].size(0)
        train_ratio, val_ratio = cfg.dataset.random_split[0], cfg.dataset.random_split[1]
        train_mask, val_mask, test_mask = generate_random_splits(
            num_target_nodes, train_ratio, val_ratio)
        data[cfg.dataset.tgt_type]['train_mask'] = torch.tensor(train_mask)
        data[cfg.dataset.tgt_type]['val_mask'] = torch.tensor(val_mask)
        data[cfg.dataset.tgt_type]['test_mask'] = torch.tensor(test_mask)
    elif cfg.dataset.split == 'sehgnn':
        # Use the data splits adopted by SeHGNN
        print('Use the data splits adopted by SeHGNN')
        val_ratio = 0.2
        train_val_nid = np.load('datasets/DBLP/dblp_train_val.npy')
        np.random.shuffle(train_val_nid)
        num_nodes_train_val = train_val_nid.shape[0]
        num_nodes_total = data[cfg.dataset.tgt_type]['x'].size(0)
        split = int(num_nodes_train_val * val_ratio)
        train_nid = train_val_nid[split:]
        val_nid = train_val_nid[:split]
        train_mask = [
            True if i in train_nid else False for i in range(num_nodes_total)]
        val_mask = [
            True if i in val_nid else False for i in range(num_nodes_total)]
        test_mask = [True if i not in val_nid and i not in val_nid else False for i in range(
            num_nodes_total)]
        data[cfg.dataset.tgt_type]['train_mask'] = torch.tensor(train_mask)
        data[cfg.dataset.tgt_type]['val_mask'] = torch.tensor(val_mask)
        data[cfg.dataset.tgt_type]['test_mask'] = torch.tensor(test_mask)
        # pdb.set_trace()

    train_loader = NeighborLoader(
        data,
        # Sample 15 neighbors for each node and each edge type for 2 iterations:
        num_neighbors=cfg.optim.num_neighbors,
        # Use a batch size of 128 for sampling training nodes of type "author":
        batch_size=cfg.optim.batch_size,
        input_nodes=(cfg.dataset.tgt_type,
                     data[cfg.dataset.tgt_type].train_mask),
    )

    eval_loader = NeighborLoader(
        data,
        # Sample 15 neighbors for each node and each edge type for 2 iterations:
        num_neighbors=cfg.optim.num_neighbors,
        batch_size=cfg.optim.batch_size,
        input_nodes=(cfg.dataset.tgt_type,
                     data[cfg.dataset.tgt_type].val_mask),
    )

    test_loader = NeighborLoader(
        data,
        # Sample 15 neighbors for each node and each edge type for 2 iterations:
        num_neighbors=cfg.optim.num_neighbors,
        # Use a batch size of 128 for sampling training nodes of type "author":
        batch_size=cfg.optim.batch_size,
        input_nodes=(cfg.dataset.tgt_type,
                     data[cfg.dataset.tgt_type].test_mask),
    )

    return train_loader, eval_loader, test_loader
