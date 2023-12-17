import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from tqdm import tqdm
from config import cfg
from utils import construct_onehot_label
from utils import l21_norm, dict_to_device
from torcheval.metrics.functional import multiclass_f1_score


def train_epoch(model, data, mps, lps, optimizer, lr_scheduler=None):
    model.train()
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.zero_grad()
    elif isinstance(optimizer, List):
        for opt in optimizer:
            opt.zero_grad()
    data = data.to(cfg.device)
    out = model(mps, lps)
    train_mask = data[cfg.dataset.tgt_type]['train_mask']
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.step()
    elif isinstance(optimizer, List):
        for opt in optimizer:
            opt.step()
    if lr_scheduler:
        lr_scheduler.step()
    pred = out.argmax(dim=-1)
    corrects = (pred[train_mask] == data.y[train_mask]).sum()

    return loss.item(), corrects / train_mask.sum()


def eval_epoch(model, data, mps, lps):
    model.eval()
    val_mask = data[cfg.dataset.tgt_type]['val_mask']
    data = data.to(cfg.device)
    with torch.no_grad():
        out = model(mps, lps)
        pred = out.argmax(dim=-1)
        corrects = (pred[val_mask] == data.y[val_mask]).sum()
        loss = F.cross_entropy(out[val_mask], data.y[val_mask])

    return corrects / val_mask.sum(), loss


def test(model, data, mps, lps):
    model.eval()
    test_mask = data[cfg.dataset.tgt_type]['test_mask']
    data = data.to(cfg.device)

    with torch.no_grad():
        out = model(mps, lps)
        pred = out.argmax(dim=-1)
        macro_f1 = multiclass_f1_score(
            pred[test_mask], data.y[test_mask], num_classes=cfg.dataset.num_classes, average='macro')
        micro_f1 = multiclass_f1_score(
            pred[test_mask], data.y[test_mask], num_classes=cfg.dataset.num_classes, average='micro')
    return macro_f1, micro_f1





def train_epoch_batched(model, train_loader, loss_fcn, optimizer, evaluator, device,
		  feats, label_feats, labels_cuda, label_emb, mask=None, scalar=None):
	model.train()
	total_loss = 0
	iter_num = 0
	y_true, y_pred = [], []

	for batch in train_loader:
		batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
		batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
		batch_label_emb = label_emb[batch].to(device)
		batch_y = labels_cuda[batch]

		optimizer.zero_grad()
		if scalar is not None:
			with torch.cuda.amp.autocast():
				output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
				if isinstance(loss_fcn, nn.BCELoss):
					output_att = torch.sigmoid(output_att)
				loss_train = loss_fcn(output_att, batch_y)
			scalar.scale(loss_train).backward()
			scalar.step(optimizer)
			scalar.update()
		else:
			output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
			if isinstance(loss_fcn, nn.BCELoss):
				output_att = torch.sigmoid(output_att)
			L1 = loss_fcn(output_att, batch_y)
			loss_train = L1
			loss_train.backward()
			optimizer.step()

		y_true.append(batch_y.cpu().to(torch.long))
		if isinstance(loss_fcn, nn.BCELoss):
			y_pred.append((output_att.data.cpu() > 0).int())
		else:
			y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
		total_loss += loss_train.item()
		iter_num += 1
	loss = total_loss / iter_num
	acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
	return loss, acc


def train_multi_stage_epoch_batched(model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device,
					  feats, label_feats, labels, label_emb, predict_prob, gama, scalar=None):
	model.train()
	loss_fcn = nn.CrossEntropyLoss()
	y_true, y_pred = [], []
	total_loss = 0
	loss_l1, loss_l2 = 0., 0.
	iter_num = 0
	for idx_1, idx_2 in zip(train_loader, enhance_loader):
		# pdb.set_trace()
		idx = torch.cat((idx_1, idx_2), dim=0)
		L1_ratio = len(idx_1) * 1.0 / (len(idx_1) + len(idx_2))
		L2_ratio = len(idx_2) * 1.0 / (len(idx_1) + len(idx_2))

		batch_feats = {k: x[idx].to(device) for k, x in feats.items()}
		batch_labels_feats = {k: x[idx].to(device) for k, x in label_feats.items()}
		batch_label_emb = label_emb[idx].to(device)
		y = labels[idx_1].to(torch.long).to(device)
		extra_weight, extra_y = predict_prob[idx_2].max(dim=1)
		extra_weight = extra_weight.to(device)
		extra_y = extra_y.to(device)

		optimizer.zero_grad()
		if scalar is not None:
			with torch.cuda.amp.autocast():
				output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
				L1 = loss_fcn(output_att[:len(idx_1)],  y)
				L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
				L2 = (L2 * extra_weight).sum() / len(idx_2)
				loss_train = L1_ratio * L1 + gama * L2_ratio * L2
			scalar.scale(loss_train).backward()
			scalar.step(optimizer)
			scalar.update()
		else:
			output_att = model(batch_feats, label_emb[idx].to(device))
			L1 = loss_fcn(output_att[:len(idx_1)],  y)
			L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
			L2 = (L2 * extra_weight).sum() / len(idx_2)
			loss_train = L1_ratio * L1 + gama * L2_ratio * L2
			loss_train.backward()
			optimizer.step()

		y_true.append(labels[idx_1].to(torch.long))
		y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
		total_loss += loss_train.item()
		loss_l1 += L1.item()
		loss_l2 += L2.item()
		iter_num += 1

	print(loss_l1 / iter_num, loss_l2 / iter_num)
	loss = total_loss / iter_num
	approx_acc = evaluator(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
	return loss, approx_acc

@torch.no_grad()
def generate_all_output(model, feats, label_feats, label_emb, all_loader, device):
	model.eval()
	preds = []
	for batch in tqdm(all_loader):
		batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
		batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
		batch_label_emb = label_emb[batch].to(device)
		preds.append(model(batch_feats, batch_labels_feats, batch_label_emb).cpu())
	preds = torch.cat(preds, dim=0)
	return preds

