import logging
import os
from yacs.config import CfgNode as CN


# Global config object
cfg = CN()


def set_cfg(cfg):
    r'''
     This function sets the default config value.
     1) Note that for an experiment, only part of the arguments will be used
     The remaining unused arguments won't affect anything.
     2) We support *at most* two levels of configs, e.g., cfg.dataset.name
     '''

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #

    # Select the device, cpu or cuda
    cfg.device = 'cuda:1'

    # Random seed
    cfg.seed = 42

    # Repeat experitment times
    cfg.repeat = 1

    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #

    cfg.dataset = CN()

    cfg.dataset.name = 'dblp'

    # Target node type in HG which will be automatically set, do not modify manually.
    cfg.dataset.tgt_type = 'author'

    # Modified automatically by code, no need to set
    cfg.dataset.num_nodes = -1

    # Modified automatically by code, no need to set
    cfg.dataset.num_classes = -1

    # Dir to load the dataset. If the dataset is downloaded, it is in root
    cfg.dataset.root = './datasets'

    # Assert split in ['public', 'random']
    cfg.dataset.split = 'public'

    # Only works if split='ramdom' is set, train-val-(test)
    cfg.dataset.random_split = [0.6, 0.2]

    # For ogbn-mag where some nodes have no features. Their features are set to norm vectors.
    cfg.dataset.embed_size = 256
    # ------------------------------------------------------------------------ #
    # Optimization options
    # ------------------------------------------------------------------------ #

    cfg.optim = CN()

    # Maximal number of epochs
    cfg.optim.epochs = 100

    cfg.optim.warmup = 20

    cfg.optim.patience = 30

    # Base learning rate
    cfg.optim.lr = 5e-3

    # L2 regularization
    cfg.optim.wd = 0.0

    # Batch size, only works in minibatch mode
    cfg.optim.batch_size = 10000

    # Sampled neighbors size, only works in minibatch mode
    cfg.optim.num_neighbors = [15, 2]

    cfg.optim.eval_protocol = 'acc'

    # Use amp to accelerate training with float16(half) calculation
    cfg.optim.amp = False
    # Stages for multi-round training
    cfg.optim.stages = [300, 300, 300, 300]
    cfg.optim.enhance_threshold = 0.75

    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #

    cfg.model = CN()

    # Model to use
    cfg.model.name = 'AdaptiveHGNN'

    # Graph Rewire options
    cfg.model.rewire = CN()
    # Assered in ['feature', 'structure-pagerank', 'structure-gpr']

    cfg.model.rewire.mode = 'none' 
    cfg.model.rewire.limit = 1
    cfg.model.rewire.threshold = 0.95
    
    # Maximum hop ranges for message passing and label propagation
    cfg.model.mp_hop = 2
    cfg.model.lp_hop = 4

    # Selective linear unit or normal linear unit?
    cfg.model.selective = False
    cfg.model.disable_simple_adaptive = False
    cfg.model.alpha = 0.85

    # Hidden layer dim
    cfg.model.hidden_dim = 64

    # Dropout rate
    cfg.model.dropout = 0.5
    cfg.model.input_dropout = 0.5
    cfg.model.attention_dropout = 0.0
    cfg.model.label_dropout = 0.0

    # Number of attetnion heads
    cfg.model.num_attention_heads = 1
    cfg.model.activation = 'plain'

    # Layer number
    cfg.model.n_fp_layers = 2
    cfg.model.n_residual_layers = 1

    cfg.model.residual = True
    cfg.model.label_prop = False

    cfg.model.task = CN()
    cfg.model.task.layers = 3
    cfg.model.task.init_last_layer = True

    cfg.model.l2_norm = False


set_cfg(cfg)
