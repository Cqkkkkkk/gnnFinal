import os
import gc
import time
import uuid
import pdb
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

# from model.tmpmodel import SeHGNN_mag, AdaptiveHGNN_mag
from model.adaptivehgnn_mag import AdaptiveHGNN_mag

from data.load_dataset import load_dataset

from utils import to_dgl, get_ogb_evaluator, set_seed_global

from config import cfg
from train_eval import train_epoch_batched, train_multi_stage_epoch_batched, generate_all_output
from model.propagate_feature import HeteroMessagePassing
from model.propagate_label import HeteroLabelPropagationDense


def main():

	transforms = T.Compose([
		T.ToUndirected(),  # Add reverse edge types.
		T.NormalizeFeatures(),
		# T.AddSelfLoops(),
	])
	data, metapaths = load_dataset(transforms)
	
	
	# pdb.set_trace()
	init_labels = data.y
	num_nodes = data['paper'].num_nodes
	n_classes = cfg.dataset.num_classes
	train_nid = torch.where(data['paper'].train_mask)[0]
	val_nid = torch.where(data['paper'].val_mask)[0]
	test_nid = torch.where(data['paper'].test_mask)[0]
	evaluator = get_ogb_evaluator('ogbn-mag')

	# =======
	# rearange node idx (for feats & labels)
	# =======
	train_node_nums = len(train_nid)
	valid_node_nums = len(val_nid)
	test_node_nums = len(test_nid)
	trainval_point = train_node_nums
	valtest_point = trainval_point + valid_node_nums
	total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)
	# pdb.set_trace()
	init2sort = torch.cat([train_nid, val_nid, test_nid])
	sort2init = torch.argsort(init2sort)

	assert torch.all(init_labels[init2sort][sort2init] == init_labels)
	labels = init_labels[init2sort]

	# =======
	# features propagate alongside the metapath
	# =======

	
	message_passer = HeteroMessagePassing()
	mps = message_passer(data, cfg.dataset.tgt_type,
						 cfg.model.mp_hop, verbose=False)
	mps = {k: v[init2sort] for k, v in mps.items()}

	gc.collect()

	all_loader = torch.utils.data.DataLoader(
		torch.arange(num_nodes), batch_size=cfg.optim.batch_size, shuffle=False, drop_last=False)

	if cfg.optim.amp:
		scalar = torch.cuda.amp.GradScaler()
	else:
		scalar = None

	labels_cuda = labels.long().to(cfg.device)

	for stage, epochs in enumerate(cfg.optim.stages):

		# =======
		# Expand training set & train loader
		# =======
		if stage > 0:

			preds = raw_preds.argmax(dim=-1)
			predict_prob = raw_preds.softmax(dim=1)

			confident_mask = predict_prob.max(
				1)[0] > cfg.optim.enhance_threshold
			val_enhance_offset = torch.where(
				confident_mask[trainval_point:valtest_point])[0]
			test_enhance_offset = torch.where(
				confident_mask[valtest_point:total_num_nodes])[0]
			val_enhance_nid = val_enhance_offset + trainval_point
			test_enhance_nid = test_enhance_offset + valtest_point
			enhance_nid = torch.cat((val_enhance_nid, test_enhance_nid))

			del train_loader
			train_batch_size = int(
				cfg.optim.batch_size * len(train_nid) / (len(enhance_nid) + len(train_nid)))
			train_loader = torch.utils.data.DataLoader(
				torch.arange(train_node_nums),
				batch_size=train_batch_size,
				shuffle=True,
				drop_last=False
			)
			enhance_batch_size = int(
				cfg.optim.batch_size * len(enhance_nid) / (len(enhance_nid) + len(train_nid)))
			enhance_loader = torch.utils.data.DataLoader(
				enhance_nid,
				batch_size=enhance_batch_size,
				shuffle=True,
				drop_last=False
			)
		else:
			train_loader = torch.utils.data.DataLoader(
				torch.arange(train_node_nums), batch_size=cfg.optim.batch_size, shuffle=True, drop_last=False)

		# =======
		# labels propagate alongside the metapath
		# =======
		lps = {}

		if stage > 0:
			label_onehot = predict_prob[sort2init].clone()
		else:
			label_onehot = torch.zeros((num_nodes, n_classes))

		label_onehot[train_nid] = F.one_hot(
			init_labels[train_nid], n_classes).float()

		
		label_propagator = HeteroLabelPropagationDense()
		lps, label_emb = label_propagator(data,
							   label_onehot,
							   num_nodes=num_nodes,
							   num_classes=n_classes,
							   num_hops=cfg.model.lp_hop,
							   tgt_type=cfg.dataset.tgt_type,
							   verbose=False)

		lps = {k: v[init2sort] for k, v in lps.items()}
		label_emb = label_emb[init2sort]

		if stage == 0:
			lps = {}

		# =======
		# Eval loader, both eval and test nodes
		# =======
		if stage > 0:
			del eval_loader
		eval_loader = []
		for batch_idx in range((num_nodes - trainval_point - 1) // cfg.optim.batch_size + 1):
			batch_start = batch_idx * cfg.optim.batch_size + trainval_point
			batch_end = min(num_nodes, (batch_idx+1) * cfg.optim.batch_size + trainval_point)

			batch_feats = {k: v[batch_start:batch_end] for k, v in mps.items()}
			batch_label_feats = {k: v[batch_start:batch_end]
								 for k, v in lps.items()}
			batch_labels_emb = label_emb[batch_start:batch_end]
			eval_loader.append(
				(batch_feats, batch_label_feats, batch_labels_emb))

		data_size = {k: v.size(-1) for k, v in mps.items()}

		# =======
		# Construct network
		# =======
		# pdb.set_trace()
		# model = SeHGNN_mag(
		model = AdaptiveHGNN_mag(
			data_size=data_size,
			nfeat=cfg.dataset.embed_size,
			hidden=cfg.model.hidden_dim,
			nclass=n_classes,
			num_feats=len(mps),
			num_label_feats=len(lps),
			tgt_key=cfg.dataset.tgt_type,
			dropout=cfg.model.dropout,
			input_drop=cfg.model.input_dropout,
			att_drop=cfg.model.attention_dropout,
			label_drop=cfg.model.label_dropout,
			n_layers_1=cfg.model.n_fp_layers,
			n_layers_2=cfg.model.task.layers,
			n_layers_3=cfg.model.n_residual_layers,
			act=cfg.model.activation,
			residual=cfg.model.residual,
			bns=True,
		)
		model = model.to(cfg.device)

		loss_fcn = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(
			model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd)

		best_epoch = 0
		best_val_acc = 0
		best_test_acc = 0
		count = 0
		eval_every = 1
		patience = 100

		for epoch in range(epochs):
			gc.collect()
			torch.cuda.empty_cache()
			start = time.time()
			if stage == 0:
				loss, acc = train_epoch_batched(model, train_loader, loss_fcn, optimizer,
												evaluator, cfg.device, mps, lps, labels_cuda, label_emb, scalar=scalar)
			else:
				loss, acc = train_multi_stage_epoch_batched(model, train_loader, enhance_loader, loss_fcn, optimizer,
															evaluator, cfg.device, mps, lps, labels_cuda, label_emb, predict_prob, gama=10, scalar=scalar)
			end = time.time()

			log = "Epoch {}, Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}\n".format(
				epoch, end-start, loss, acc*100)
			torch.cuda.empty_cache()

			if epoch % eval_every == 0:
				with torch.no_grad():
					model.eval()
					raw_preds = []

					start = time.time()
					for batch_feats, batch_label_feats, batch_labels_emb in eval_loader:
						batch_feats = {k: v.to(cfg.device)
									   for k, v in batch_feats.items()}
						batch_label_feats = {
							k: v.to(cfg.device) for k, v in batch_label_feats.items()}
						batch_labels_emb = batch_labels_emb.to(cfg.device)
						raw_preds.append(
							model(batch_feats, batch_label_feats, batch_labels_emb).cpu())
					raw_preds = torch.cat(raw_preds, dim=0)
					# pdb.set_trace()

					loss_val = loss_fcn(
						raw_preds[:valid_node_nums], labels[trainval_point:valtest_point]).item()
					loss_test = loss_fcn(
						raw_preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes]).item()

					# pdb.set_trace()
					preds = raw_preds.argmax(dim=-1)
					val_acc = evaluator(preds[:valid_node_nums], labels[trainval_point:valtest_point])
					test_acc = evaluator(
						preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes])

					end = time.time()
					log += f'Time: {end-start}, Val loss: {loss_val}, Test loss: {loss_test}\n'
					log += 'Val acc: {:.4f}, Test acc: {:.4f}\n'.format(
						val_acc*100, test_acc*100)

				if val_acc > best_val_acc:
					best_epoch = epoch
					best_val_acc = val_acc
					best_test_acc = test_acc

					torch.save(model.state_dict(),
							   f'ckpts/ogbn-mag-stage{stage}.pt')
					count = 0
				else:
					count = count + eval_every
					if count >= patience:
						break
				log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(
					best_epoch, best_val_acc*100, best_test_acc*100)
			print(log, flush=True)

		print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
			best_epoch, best_val_acc*100, best_test_acc*100))

		model.load_state_dict(torch.load(f'ckpts/ogbn-mag-stage{stage}.pt'))

		raw_preds = generate_all_output(
			model, mps, lps, label_emb, all_loader, cfg.device)
	return best_test_acc


def parse_args():
	parser = argparse.ArgumentParser(description='SeHGNN')
	parser.add_argument('--cfg_file', dest='cfg_file',
						default='configs/ogb-mag.yaml')

	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	cfg.merge_from_file(args.cfg_file)

	macro_f1s = []
	micro_f1s = []
	# Repeat for different random seeds
	for i in range(cfg.repeat):
		set_seed_global(cfg.seed)
		cfg.seed += i * 146
		micro_f1 = main()
		# macro_f1s.append(macro_f1 * 100)
		micro_f1s.append(micro_f1 * 100)

	macro_f1s = np.array(macro_f1s)
	micro_f1s = np.array(micro_f1s)
	print(f"Dataset {cfg.dataset.name}: ")
	# print(f'[ACC]: Mean {np.mean(accs):.2f} Std {np.std(accs):.2f}')
	print(f'[MacroF1]: Mean {macro_f1s.mean():.2f} Std {macro_f1s.std():.2f}')
	print(f'[MicroF1]: Mean {micro_f1s.mean():.2f} Std {micro_f1s.std():.2f}')
