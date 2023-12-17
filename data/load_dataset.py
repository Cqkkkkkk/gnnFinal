import pdb
import os
import torch
import sparse_tools
from torch_geometric.datasets import DBLP
from config import cfg
from torch_geometric.datasets import DBLP, OGB_MAG, IMDB, HGBDataset
from data.actor import Actor
from data.fb_american import FBDataset
from data.mag_year import MAG_YEAR
from utils import generate_random_splits, cal_adj, set_seed_global


def load_dataset(transforms):
    metapaths = []
    if cfg.dataset.name == 'dblp':
        dataset = DBLP(root='./datasets/DBLP', transform=transforms)
        data = dataset[0]
        data['conference']['x'] = torch.eye(20)
        del data['conference']['num_nodes']
    elif cfg.dataset.name == 'imdb':
        dataset = IMDB(root='./datasets/IMDB', transform=transforms)
        data = dataset[0]
    elif cfg.dataset.name == 'acm':
        dataset = HGBDataset(root='./datasets', name='ACM',
                             transform=transforms)
        data = dataset[0]
        cfg.dataset.tgt_type = 'paper'
        data[cfg.dataset.tgt_type].y = torch.tensor(torch.load('datasets/acm/processed/acm_y.pt'))

        data['term']['x'] = torch.eye(1902)
        del data['term']['num_nodes']     

        # del data['term']
        # for src, rel, dst in data.metadata()[1]:
        #     if src == 'term' or dst == 'term':
        #         del data[(src, rel, dst)]
        # pdb.set_trace()
  
    elif cfg.dataset.name == 'fb-american':
        dataset = FBDataset(root="./datasets/facebook", name="American75")
        data = dataset[0]
        cfg.dataset.tgt_type = 'person'
        data[cfg.dataset.tgt_type].y = data[cfg.dataset.tgt_type].y + 1

        for key in data.metadata()[0]:
            data[key].x = data[key].x.to_dense()

        del data['pos']
        for key in data.x_dict.keys():
            del data[key]['num_nodes']
    elif cfg.dataset.name == 'fb-mit':
        dataset = FBDataset(root="./datasets/facebook", name="MIT8")
        data = dataset[0]
        cfg.dataset.tgt_type = 'person'
        data[cfg.dataset.tgt_type].y = data[cfg.dataset.tgt_type].y + 1

        for key in data.metadata()[0]:
            data[key].x = data[key].x.to_dense()

        del data['pos']
        for key in data.x_dict.keys():
            del data[key]['num_nodes']

    elif cfg.dataset.name == 'actor':
        dataset = Actor(root="./datasets/actor")
        data = dataset[0]
        cfg.dataset.tgt_type = 'starring'
        for key in data.metadata()[0]:
            data[key].x = data[key].x.to_dense()

    elif cfg.dataset.name == 'freebase':
        dataset = HGBDataset(
            root='./datasets', name='Freebase', transform=transforms)
        data = dataset[0]
        cfg.dataset.tgt_type = 'book'
        metapaths.append(['book', 'book'])
    elif cfg.dataset.name == 'ogb-mag':
        dataset = OGB_MAG(root='./datasets/ogb-mag', transform=transforms)
        data = dataset[0]
        cfg.dataset.tgt_type = 'paper'
        embed_size = 256
        data['author'].x = torch.Tensor(data['author'].num_nodes, embed_size).uniform_(-0.5, 0.5)
        data['field_of_study'].x = torch.Tensor(data['field_of_study'].num_nodes, embed_size).uniform_(-0.5, 0.5)
        data['institution'].x = torch.Tensor(data['institution'].num_nodes, embed_size).uniform_(-0.5, 0.5)

        paper_author_adj = cal_adj(data[('paper', 'rev_writes', 'author')].edge_index, data['paper'].num_nodes, data['author'].num_nodes)
        paper_paper_adj = cal_adj(data[('paper', 'cites', 'paper')].edge_index, data['paper'].num_nodes, data['paper'].num_nodes)
        paper_field_adj = cal_adj(data[('paper', 'has_topic', 'field_of_study')].edge_index, data['paper'].num_nodes, data['field_of_study'].num_nodes)

        diag_name = 'datasets/ogb-mag/mag/processed/PFP_diag.pt'
        if not os.path.exists(diag_name):
            paper_field_paper_diag = sparse_tools.spspmm_diag_sym_ABA(paper_field_adj)
            torch.save(paper_field_paper_diag, diag_name)
        
        diag_name = 'datasets/ogb-mag/mag/processed/PPP_diag.pt'
        if not os.path.exists(diag_name):
            paper_paper_paper_diag = sparse_tools.spspmm_diag_sym_AAA(paper_paper_adj)
            torch.save(paper_paper_paper_diag, diag_name)

        diag_name = f'datasets/ogb-mag/mag/processed/PAP_diag.pt'
        if not os.path.exists(diag_name):
            paper_author_paper_diag = sparse_tools.spspmm_diag_sym_ABA(paper_author_adj)
            torch.save(paper_author_paper_diag, diag_name)


    elif cfg.dataset.name == 'mag-year':
        dataset = MAG_YEAR(root='./datasets/mag-year', transform=transforms)
        data = dataset[0]
        cfg.dataset.tgt_type = 'paper'

        # Override splits mannually
        # The default splits is in-balanced (to be precise, there are categrories that have no training data)
        # which makes nearly all the methods fail to train. So we generate a new split mannually.
        set_seed_global(cfg.seed)
        num_nodes = data[cfg.dataset.tgt_type].num_nodes

        train_mask = torch.full((num_nodes, ), False, dtype=torch.bool)
        val_mask = torch.full((num_nodes, ), False, dtype=torch.bool)
        test_mask = torch.full((num_nodes, ), False, dtype=torch.bool)

        num_train_nodes = int(num_nodes * cfg.dataset.random_split[0])
        num_val_nodes = int(num_nodes * cfg.dataset.random_split[1])

        permute = torch.randperm(num_nodes)
        train_idx = permute[: num_train_nodes]
        val_idx = permute[num_train_nodes: num_train_nodes + num_val_nodes]
        test_idx = permute[num_train_nodes + num_val_nodes:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        data[cfg.dataset.tgt_type].train_mask = train_mask
        data[cfg.dataset.tgt_type].val_mask = val_mask
        data[cfg.dataset.tgt_type].test_mask = test_mask
        

        embed_size = 256
        data['author'].x = torch.Tensor(data['author'].num_nodes, embed_size).uniform_(-0.5, 0.5)
        data['field_of_study'].x = torch.Tensor(data['field_of_study'].num_nodes, embed_size).uniform_(-0.5, 0.5)
        data['institution'].x = torch.Tensor(data['institution'].num_nodes, embed_size).uniform_(-0.5, 0.5)

        paper_author_adj = cal_adj(data[('paper', 'rev_writes', 'author')].edge_index, data['paper'].num_nodes, data['author'].num_nodes)
        paper_paper_adj = cal_adj(data[('paper', 'cites', 'paper')].edge_index, data['paper'].num_nodes, data['paper'].num_nodes)
        paper_field_adj = cal_adj(data[('paper', 'has_topic', 'field_of_study')].edge_index, data['paper'].num_nodes, data['field_of_study'].num_nodes)

        diag_name = 'datasets/mag-year/mag/processed/PFP_diag.pt'
        if not os.path.exists(diag_name):
            paper_field_paper_diag = sparse_tools.spspmm_diag_sym_ABA(paper_field_adj)
            torch.save(paper_field_paper_diag, diag_name)
        
        diag_name = 'datasets/mag-year/mag/processed/PPP_diag.pt'
        if not os.path.exists(diag_name):
            paper_paper_paper_diag = sparse_tools.spspmm_diag_sym_AAA(paper_paper_adj)
            torch.save(paper_paper_paper_diag, diag_name)

        diag_name = f'datasets/mag-year/mag/processed/PAP_diag.pt'
        if not os.path.exists(diag_name):
            paper_author_paper_diag = sparse_tools.spspmm_diag_sym_ABA(paper_author_adj)
            torch.save(paper_author_paper_diag, diag_name)

    data.y = data[cfg.dataset.tgt_type].y
    cfg.dataset.num_classes = torch.unique(data.y).numel()

    return data, metapaths


def override_splits(data):
    # Splits overriding
    if cfg.dataset.split == 'random':
        train_mask, val_mask, test_mask = generate_random_splits(
            num_nodes=data[cfg.dataset.tgt_type].num_nodes,
            train_ratio=cfg.dataset.random_split[0],
            val_ratio=cfg.dataset.random_split[1],
        )
        data[cfg.dataset.tgt_type].train_mask = train_mask
        data[cfg.dataset.tgt_type].val_mask = val_mask
        data[cfg.dataset.tgt_type].test_mask = test_mask

    return data