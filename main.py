import os
import pdb
import sys
import torch
import argparse
import warnings
import numpy as np
import torch_geometric.transforms as T
from tqdm import tqdm
from torch.optim import Adam

from config import cfg
from utils import set_seed_global
from data.load_dataset import load_dataset, override_splits
from train_eval import train_epoch, eval_epoch, test
from model.adaptivehgnn import AdaptiveHGNN
from model.propagate_feature import HeteroMessagePassing
from model.propagate_label import HeteroLabelPropagation
from lr_scheduler import get_cosine_schedule_with_warmup


# Surpress the Sparse CSR warning
warnings.filterwarnings(
    'ignore', '.*Sparse CSR tensor support is in beta state.*')


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Main entry")
    parser.add_argument('--cfg', dest='cfg_file', default='configs/base.yaml',
                        help='Config file path', type=str)

    if len(sys.argv) == 1:
        print('Now you are using the default configs.')
        parser.print_help()

    return parser.parse_args()


def main():
    transforms = T.Compose([
        T.ToUndirected(),  # Add reverse edge types.
        T.NormalizeFeatures(),
        # T.AddSelfLoops(),
    ])
    data, metapaths = load_dataset(transforms)

    data = override_splits(data)

    message_passer = HeteroMessagePassing()
    mps = message_passer(data,
                         tgt_type=cfg.dataset.tgt_type,
                         num_hops=cfg.model.mp_hop,
                         verbose=True)
    if cfg.model.label_prop:
        label_propagater = HeteroLabelPropagation()
        lps = label_propagater(data,
                            tgt_type=cfg.dataset.tgt_type,
                            num_hops=cfg.model.lp_hop,
                            verbose=False)
    else:
        lps = {}
        
    in_lps_dim_dict = {}
    for key, val in lps.items():
        in_lps_dim_dict[key] = val.size(-1)

    in_mps_dim_dict = {}
    for key, val in mps.items():
        in_mps_dim_dict[key] = val.size(-1)

    model = AdaptiveHGNN(
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.dataset.num_classes,
        tgt_type=cfg.dataset.tgt_type,
        dropout=cfg.model.dropout,
        input_dropout=cfg.model.input_dropout,
        alpha=cfg.model.alpha,
        attention_dropout=cfg.model.attention_dropout,
        num_attention_heads=cfg.model.num_attention_heads,
        n_fp_layers=cfg.model.n_fp_layers,
        n_task_layers=cfg.model.task.layers,
        residual=cfg.model.residual,
        label_prop=cfg.model.label_prop,
        selective=cfg.model.selective,
        in_mps_dim_dict=in_mps_dim_dict,
        in_lps_dim_dict=in_lps_dim_dict
    ).to(cfg.device)

    optimizer_normal = Adam(
        model.parameters_split['normal'],
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.wd)
    optimizer_tfm = Adam(
        model.parameters_split['tfm'],
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.wd)
    cosine_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer_tfm,
                                                          num_warmup_steps=cfg.optim.warmup,
                                                          num_training_steps=cfg.optim.epochs)
    
    ckpt_path = f'./ckpts/{cfg.dataset.name}.pt'

    best_val_acc = 0
    best_val_loss = 1e8
    with tqdm(range(cfg.optim.epochs)) as tq:
        for epoch in tq:
            train_loss, train_acc = train_epoch(
                model, data, mps, lps,
                optimizer=[optimizer_normal, optimizer_tfm],
                lr_scheduler=cosine_lr_scheduler
            )
            val_acc, val_loss = eval_epoch(model, data, mps, lps)
            infos = {
                'Epoch': epoch,
                'TrainLoss': '{:.3}'.format(train_loss),
                'Train': '{:.3}'.format(train_acc.item()),
                'Val': '{:.3}'.format(val_acc.item()),
                # 'Test': '{:.3}'.format(test_acc.item())
            }
            tq.set_postfix(infos)
            if cfg.optim.eval_protocol == 'acc' and best_val_acc < val_acc:
                best_val_acc = val_acc
                torch.save(model, ckpt_path)
            elif cfg.optim.eval_protocol == 'loss' and best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save(model, ckpt_path)
    model = torch.load(ckpt_path)
    macro_f1, micro_f1 = test(model, data, mps, lps)
    return macro_f1.item(), micro_f1.item()


if __name__ == '__main__':

    args = parse_args()
    cfg.merge_from_file(args.cfg_file)

    macro_f1s = []
    micro_f1s = []
    # Repeat for different random seeds
    for i in range(cfg.repeat):
        set_seed_global(cfg.seed)
        cfg.seed += i * 146
        macro_f1, micro_f1 = main()
        macro_f1s.append(macro_f1 * 100)
        micro_f1s.append(micro_f1 * 100)
    macro_f1s = np.array(macro_f1s)
    micro_f1s = np.array(micro_f1s)
    print(f"Dataset {cfg.dataset.name}: ")
    # print(f'[ACC]: Mean {np.mean(accs):.2f} Std {np.std(accs):.2f}')
    print(f'[MacroF1]: Mean {macro_f1s.mean():.2f} Std {macro_f1s.std():.2f}')
    print(f'[MicroF1]: Mean {micro_f1s.mean():.2f} Std {micro_f1s.std():.2f}')
