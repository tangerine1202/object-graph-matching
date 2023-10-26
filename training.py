# %%
import os
os.environ['VISIBLE_CUDA_DEVICES'] = '0'
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from pprint import pprint
import pickle as pkl

from tqdm.auto import tqdm
import numpy as np

import torch
import torch.optim as optim
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'torch device: {device}')

# GNN package
from torch_geometric.loader import DataLoader
import networkx as nx

# Visualization
import matplotlib.pyplot as plt

# %%
from solo2graph import QueryGraph, PairedGraph
from dataset import CustomDataset, transform_data
from models import CustomModel
from losses import CustomCriterion
from eva import compute_eval, compute_corr
from viz_utils import read_img, viz_corr

# %%

EPOCHS = 500
EVAL_EPOCHS = 20
EMB_DIM = 128
MATCH_THRESHOLD = 0.2

SOLO_NAME = 'poisson1_5r16'
SCENE = 'SimpleOffice'
# SOLO_NAME = 'D_pois3r8'
# SCENE = 'WP16'
DATA_DIR = f'data/{SCENE}/{SOLO_NAME}'
GRAPH_DIR = f'{DATA_DIR}/paired_graph'
CKPT_DIR = f'ckpt/{SCENE}/{SOLO_NAME}'
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# %% [markdown]
#

# %%
ds = CustomDataset(root=GRAPH_DIR, transform=transform_data, max_len=1)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
dc = next(iter(dl))

# %%
for attr_name, attr in {k: v for k, v in dc.items() if k.startswith('node_')}.items():
    print(f'{attr_name} dim: {attr.shape[2]}')

edge_attr_dim = dc['edge_attr'].shape[2]
print(f'edge_attr dim: {edge_attr_dim}')

# %%
ds = CustomDataset(root=GRAPH_DIR, transform=transform_data, max_len=5000, min_overlap=3)
# random select 1000 samples from ds
ds = torch.utils.data.Subset(ds, np.random.choice(len(ds), 5000, replace=False))
train_ds, eval_ds, test_ds = torch.utils.data.random_split(ds, [0.5, 0.2, 0.3])
# # # train_ds = torch.utils.data.Subset(train_ds, [0])  # for debugging

# # # Do not use batch_size>1 for now, it will crash the e1i, e1j, e2i, e2j indices
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

print('train size:', len(train_ds))
print('eval size:', len(eval_ds))
print('test size:', len(test_ds))

# %%


def data_dict_to_device(data_dict, device):
    for k in data_dict:
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].to(device)
    return data_dict


# %%


def train_step(model, data_dict, optimizer, criterion):
    data_dict = data_dict_to_device(data_dict, device)
    model.train()
    optimizer.zero_grad()
    pred_dict = model(data_dict)
    loss = criterion(pred_dict, data_dict)
    loss.backward()
    optimizer.step()
    return loss.item()

# %%


def eval_step(model, data_dict, criterion):
    data_dict = data_dict_to_device(data_dict, device)
    model.eval()
    with torch.no_grad():
        pred_dict = model(data_dict)
        loss = criterion(pred_dict, data_dict)
    metrics = compute_eval(pred_dict, data_dict)
    return loss.item(), metrics


# %%
model = CustomModel(edge_attr_dim, EMB_DIM, match_threshold=MATCH_THRESHOLD).to(device)
criterion = CustomCriterion(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, amsgrad=True)

e = 0
history = {'train_loss': [], 'eval_loss': [], 'eval_metrics': []}

# %%


# %%
# training loop
for _ in tqdm(range(EPOCHS)):
    e += 1
    train_loss = 0
    for data_dict in train_dl:
        train_loss += train_step(model, data_dict, optimizer, criterion)
    train_loss /= len(train_dl)

    if e % EVAL_EPOCHS == 0:
        mean_metrics = {}
        sum_eval_loss = 0

        with torch.no_grad():
            model.eval()
            for data_dict in eval_dl:
                eval_loss, metrics = eval_step(model, data_dict, criterion)
                sum_eval_loss += eval_loss
                for k, v in metrics.items():
                    if k in mean_metrics:
                        mean_metrics[k] = mean_metrics.get(k, 0) + v
                    else:
                        mean_metrics[k] = v

        for k, v in mean_metrics.items():
            mean_metrics[k] = v / len(eval_dl)
        eval_loss = sum_eval_loss / len(eval_dl)

        torch.save(model.state_dict(), f'{CKPT_DIR}/model_{e}.pth')

        history['train_loss'].append(train_loss)
        history['eval_loss'].append(eval_loss)
        history['eval_metrics'].append(mean_metrics)
        print(f'----- epoch: {e} -----')

        print(f'train loss: {train_loss:.6f}')
        print(f'eval  loss: {eval_loss:.6f}')
        print('eval metrics:')
        for k, v in mean_metrics.items():
            print(f'{k}: {v:.4f}')
        print()

# plt.plot(history['train_loss'], label='train loss')
# plt.plot(history['eval_loss'], label='eval loss')
# plt.legend()
# plt.figure()
# for k, v in history['eval_metrics'][0].items():
#     plt.plot([x[k] for x in history['eval_metrics']], label=f'eval {k}')
# plt.legend()
# plt.show()

# %%
mean_metrics = {}
sum_test_loss = 0
gid2pred = {}

cnt = 0
with torch.no_grad():
    model.eval()
    for data_dict in tqdm(test_dl):
        test_loss, metrics = eval_step(model, data_dict, criterion)

        cnt += 1
        pred_dict = model(data_dict)
        sum_test_loss += test_loss

        g1_step = data_dict['g1_step'][0].item()
        if g1_step not in gid2pred:
            gid2pred[g1_step] = {'pred': [pred_dict], 'data': [data_dict], 'eva': [metrics]}
        else:
            gid2pred[g1_step]['pred'].append(pred_dict)
            gid2pred[g1_step]['data'].append(data_dict)
            gid2pred[g1_step]['eva'].append(metrics)

        for k, v in metrics.items():
            if k in mean_metrics:
                mean_metrics[k] = mean_metrics[k] + v
            else:
                mean_metrics[k] = v


for k, v in mean_metrics.items():
    mean_metrics[k] = v / cnt  # len(test_dl)

mean_test_loss = sum_test_loss / cnt  # len(test_dl)
print(f'mean_test loss: {mean_test_loss:.4f}')
print('test metrics:')
for k, v in mean_metrics.items():
    print(f'{k}: {v:.4f}')

# %%
res = []

for gid, v in gid2pred.items():
    g1_pose = v['data'][0]['g1_camera_pose'][0]
    g1_step = v['data'][0]['g1_step'][0].item()
    g2_step = v['data'][0]['g2_step'][0].item()

    preds = []

    gt_pose = None
    gt_step = -1
    gt_pos_loss = np.inf
    gt_q_loss = np.inf

    for idx in range(len(v['pred'])):
        pred_dict = v['pred'][idx]
        data_dict = v['data'][idx]

        g1_pose = data_dict['g1_camera_pose'][0].cpu()
        g1_step = data_dict['g1_step'][0].item()
        g2_pose = data_dict['g2_camera_pose'][0].cpu()
        g2_step = data_dict['g2_step'][0].item()
        pos_loss = np.linalg.norm(g1_pose[:3] - g2_pose[:3]).item()
        q_loss = np.linalg.norm(g1_pose[3:] - g2_pose[3:]).item()

        # pred
        match_mask = (pred_dict['matches0'] != -1)
        if sum(match_mask) == 0:
            score = -np.inf
            continue
        else:
            match_score = pred_dict['matching_scores0'][match_mask].sort(descending=True)[0][:3]
            score = torch.mean(match_score).item()
        preds.append({
            'pred': pred_dict,
            'data': data_dict,
            'score': score,
            'g2_step': g2_step,
            'g2_pose': g2_pose,
            'pos_loss': pos_loss,
            'q_loss': q_loss,
        })

        # gt
        if pos_loss < gt_pos_loss:
            gt_pos_loss = pos_loss
            gt_q_loss = q_loss
            gt_pose = g2_pose.numpy()
            gt_step = g2_step
        if pos_loss == gt_pos_loss and q_loss < gt_q_loss:
            gt_pos_loss = pos_loss
            gt_q_loss = q_loss
            gt_pose = g2_pose.numpy()
            gt_step = g2_step

    preds = sorted(preds, key=lambda x: x['score'], reverse=True)

    # print(f'g1_step: {g1_step}')
    # print(f'gt_step: {gt_step}')
    # print(f'pred: {[p[0] for p in pred]}')
    # print()
    res.append({
        'g1_step': g1_step,
        'gt_step': gt_step,
        'gt_pose': gt_pose,
        'preds': preds,
    })


def compute_hits_at_k(gt, pred, k):
    for i in range(min(len(pred), k)):
        if np.allclose(pred[i]['pos_loss'], 0):
            return True
    return False


res = sorted(res, key=lambda x: x['g1_step'])

mean_len_of_pred = np.mean([len(d['preds']) for d in res])
std_len_of_pred = np.std([len(d['preds']) for d in res])
max_len_of_pred = np.max([len(d['preds']) for d in res])
print(f'mean_len_of_pred: {mean_len_of_pred:.4f}')
print(f'std_len_of_pred: {std_len_of_pred:.4f}')
print(f'max_len_of_pred: {max_len_of_pred:.4f}')
print()
