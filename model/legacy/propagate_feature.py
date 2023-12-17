import gc
import pdb
import torch
import dgl.function as fn
from typing import Any

from config import cfg
from utils import to_dgl


class HeteroMessagePassing:
    """
    Class for performing heterogeneous message passing in a graph.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, data, tgt_type, num_hops, verbose=False) -> Any:
        """
        Perform heterogeneous message passing on the input graph.

        Args:
            data (Any): Input graph data.
            tgt_type (str): Target node type.
            num_hops (int): Number of hops for feature propagation.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        Returns:
            propagated_feats (dict): Dictionary containing the propagated features.
        """
        if verbose:
            print(f'Generating {num_hops}-hop feature propagation')

        g = to_dgl(data)
        propagated_feats = {}

        g = self.hg_propagate_feat_dgl(g,
                                       tgt_type,
                                       num_hops,
                                       verbose=verbose)
        for k in g.nodes[tgt_type].data.keys():
            propagated_feats[k] = g.nodes[tgt_type].data[k].to(cfg.device).detach().clone()

        print(f'[MP] For target node type `{tgt_type}`, the {len(propagated_feats)} propagated messages are as:')
        if verbose:
            for k, v in propagated_feats.items():
                print(f'[MP] Metapath: {k} with embedding of shape {v.shape}')

        del g

        return propagated_feats

    def hg_propagate_feat_dgl(self, g, tgt_type, num_hops, verbose=False):
        """
        Perform feature propagation in the heterogeneous graph.

        Args:
            g (DGLHeteroGraph): Input heterogeneous graph.
            tgt_type (str): Target node type.
            num_hops (int): Number of hops for feature propagation.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        Returns:
            g (DGLHeteroGraph): Updated heterogeneous graph after feature propagation.
        """
        for hop in range(1, num_hops + 1):
            for etype in g.etypes:
                stype, _, dtype = g.to_canonical_etype(etype)
                for cur_src in list(g.nodes[stype].data.keys()):

                    if len(cur_src.split('-')) != hop:
                        continue
                    if hop == num_hops and dtype != tgt_type:
                        continue

                    cur_dst = f'{dtype}-{cur_src}'

                    g[etype].update_all(
                        fn.copy_u(cur_src, 'm'), 
                        fn.mean('m', cur_dst), 
                        etype=etype
                    )

            # remove no-use items
            for ntype in g.ntypes:
                if ntype == tgt_type:
                    continue
                removes = []
                for cur_src in g.nodes[ntype].data.keys():
                    if len(cur_src.split('-')) <= hop:
                        removes.append(cur_src)
                for cur_src in removes:
                    g.nodes[ntype].data.pop(cur_src)
                if verbose and removes:
                    print('[MP] Remove', removes)
            gc.collect()
        return g
   