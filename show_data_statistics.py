import os
import sys
import pdb
import torch
import argparse
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP, OGB_MAG, IMDB, HGBDataset

# sys.path.append(os.path.dirname(os.getcwd()))


from data.actor import Actor
from data.fb_american import FBDataset
from data.mag_year import MAG_YEAR
from utils import cal_hetero_edge_homophily, cal_improved_hetero_edge_homophily


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Main entry")
    parser.add_argument('--dataset', default='dblp', type=str)
    parser.add_argument('--improved', action='store_true')
    return parser.parse_args()


def main(args, dataset, tgt_node, metapaths):
    data = dataset[0]
    # print(data)
    data.y = data[tgt_node].y
    num_classes = torch.unique(data.y).numel()
    new_dict = {}
    for key, item in data.edge_index_dict.items():
        new_dict[(key[0], 'to', key[-1])] = item
    print('-' * 50)
    print(f'Dataset: ', args.dataset)
    print(f'Target node {tgt_node} : {data.y.size(0)}')
    print(f'Total nodes: {np.sum([x.size(0) for x in data.x_dict.values()])}')
    print(f'Feature dim: {data.x_dict[tgt_node].size(-1)}')
    print(
        f'Total edges: {np.sum([edge_index.size(-1) for edge_index in new_dict.values()])}')
    print(f'Total class: {num_classes}')
    # cal_hetero_edge_homophily(new_dict, data.y, metapath, sep='to')
    for metapath in metapaths:
        if args.improved:
            cal_improved_hetero_edge_homophily(
                new_dict, data.y, metapath, sep='to')
        else:
            cal_hetero_edge_homophily(new_dict, data.y, metapath, sep='to', local=True)


if __name__ == '__main__':
    args = parse_args()
    transforms = T.Compose([
        T.ToUndirected(),  # Add reverse edge types.
        # T.AddSelfLoops(),
    ])

    if args.dataset == 'dblp':
        dataset = DBLP(root='./datasets/DBLP', transform=transforms)
        tgt_node = 'author'
        metapaths = []
        metapaths.append(['author', 'paper', 'author'])
        metapaths.append(['author', 'paper', 'author', 'paper', 'author'])
    elif args.dataset == 'imdb':
        dataset = IMDB(root='./datasets/IMDB', transform=transforms)
        tgt_node = 'movie'
        metapaths = []
        metapaths.append(['movie', 'actor', 'movie'])
        metapaths.append(['movie', 'director', 'movie'])
    elif args.dataset == 'acm':
        dataset = HGBDataset(root='./datasets', name='ACM',
                             transform=transforms)
        tgt_node = 'paper'
        metapaths = []
        metapaths.append(['paper', 'author', 'paper'])
        metapaths.append(['paper', 'paper'])
        metapaths.append(['paper', 'subject', 'paper'])
        metapaths.append(['paper', 'term', 'paper'])
    elif args.dataset == 'fb-american':
        dataset = FBDataset(root="./datasets/facebook", name="American75")
        tgt_node = 'person'
        metapaths = []
        metapaths.append(['person', 'status', 'person'])
        metapaths.append(['person', 'major', 'person'])
        metapaths.append(['person', 'second major', 'person'])
        metapaths.append(['person', 'house', 'person'])
        metapaths.append(['person', 'year', 'person'])
        metapaths.append(['person', 'high school', 'person'])
    elif args.dataset == 'fb-mit':
        dataset = FBDataset(root="./datasets/facebook", name="MIT8")
        data = dataset[0]
        tgt_node = 'person'
        metapaths = []
        metapaths.append(['person', 'status', 'person'])
        metapaths.append(['person', 'major', 'person'])
        metapaths.append(['person', 'second major', 'person'])
        metapaths.append(['person', 'house', 'person'])
        metapaths.append(['person', 'year', 'person'])
        metapaths.append(['person', 'high school', 'person'])
    elif args.dataset == 'actor':
        dataset = Actor(root="./datasets/actor")
        tgt_node = 'starring'
        metapaths = []
        metapaths.append(['starring', 'starring'])
        metapaths.append(['starring', 'director', 'starring'])
        metapaths.append(['starring', 'writer', 'starring'])

    elif args.dataset == 'freebase':
        dataset = HGBDataset(
            root='./datasets', name='Freebase', transform=transforms)
        tgt_node = 'book'
        metapaths = []
        metapaths.append(['book', 'book'])
    elif args.dataset == 'ogb-mag':
        dataset = OGB_MAG(root='./datasets/ogb-mag', transform=transforms)
        tgt_node = 'paper'
        metapaths = []
        metapaths.append(['paper', 'paper'])
        metapaths.append(['paper', 'author', 'paper'])
    elif args.dataset == 'mag-year':
        dataset = MAG_YEAR(root='./datasets/mag-year', transform=transforms)
        tgt_node = 'paper'
        metapaths = []
        metapaths.append(['paper', 'paper'])
        metapaths.append(['paper', 'author', 'paper'])

    main(args, dataset, tgt_node, metapaths)
