# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pprint import pprint
import pickle as pkl

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_R

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
import cv2
import matplotlib.pyplot as plt

# %%
from solo2graph import MapGraph, QueryGraph, PairedGraph
from dataset import PairListDataset, transform_3D_qm_data, transform_2D_qq_data
from models import CustomModel, compute_pose
from losses import CustomCriterion
from eva import compute_eval, compute_corr, compute_confusion_matrix
from viz_utils import read_img, viz_corr

# %%

EMB_DIM = 128
MATCH_THRESHOLD = 0.2
TOTAL_EPOCHS = 75
EVAL_EPOCHS = 3

SOLO_NAME = 'poisson1r36'
SCENE = 'SimpleOffice'
DATA_DIR = f'data/{SCENE}/{SOLO_NAME}'
GRAPH_DIR = f'{DATA_DIR}/graph'
PAIR_FNAME = 'qm_paired_list.csv'
TRANSFORM = transform_3D_qm_data
EVAL_TYPE = '3d'

MIN_OVERLAP = 3
TRAIN_RATIO, EVAL_RATIO, TEST_RATIO = 0.5, 0.2, 0.3

CKPT_DIR = f'ckpt/{SCENE}/{SOLO_NAME}'
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# %%
ds = PairListDataset(root=DATA_DIR, pair_fname=PAIR_FNAME, transform=TRANSFORM)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
dc = next(iter(dl))

# %%
print('--- attr ---')
for attr_name, attr in {k: v for k, v in dc.items() if k.startswith('node_')}.items():
    print(f'{attr_name} dim: {attr.shape[2]}')
edge_attr_dim = dc['edge_attr'].shape[2]
print(f'edge attr dim: {edge_attr_dim}')
print()

# %%
print('--- dataset ---')
ds = PairListDataset(root=DATA_DIR, pair_fname=PAIR_FNAME, transform=TRANSFORM, min_overlap=MIN_OVERLAP)
print('dataset size:', len(ds))

train_ds, eval_ds, test_ds = torch.utils.data.random_split(ds, [TRAIN_RATIO, EVAL_RATIO, TEST_RATIO])
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

print('train size:', len(train_ds))
print('eval size:', len(eval_ds))
print('test size:', len(test_ds))
print()

# %%


def data_dict_to_device(data_dict, device):
    return {k: v.to(device) for k, v in data_dict.items()}
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


def eval_step(model, data_dict, criterion, eval_type):
    data_dict = data_dict_to_device(data_dict, device)
    model.eval()
    with torch.no_grad():
        pred_dict = model(data_dict)
        loss = criterion(pred_dict, data_dict)
    metrics = compute_eval(pred_dict, data_dict, eval_type=eval_type)
    return loss.item(), metrics, pred_dict


# %%
model = CustomModel(edge_attr_dim, EMB_DIM, match_threshold=MATCH_THRESHOLD).to(device)
criterion = CustomCriterion(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, amsgrad=True)

e = 0
history = {'train_loss': [], 'eval_loss': [], 'eval_metrics': []}

# %%
print('--- model ---')
print(model)
print()

# %%
# training loop
for _ in tqdm(range(TOTAL_EPOCHS)):
    e += 1
    train_loss = 0
    for train_i, data_dict in enumerate(train_dl):
        train_loss += train_step(model, data_dict, optimizer, criterion)
    train_loss /= len(train_dl)

    if e % EVAL_EPOCHS == 0:
        metrics_seq = {}
        sum_eval_loss = 0

        with torch.no_grad():
            model.eval()
            for data_dict in eval_dl:
                eval_loss, metrics, pred_dict = eval_step(model, data_dict, criterion, eval_type=EVAL_TYPE)
                sum_eval_loss += eval_loss

            for k, v in metrics.items():
                if k not in metrics_seq:
                    metrics_seq[k] = []
                metrics_seq[k].append(v)

        eval_loss = sum_eval_loss / len(eval_dl)

        torch.save(model.state_dict(), f'{CKPT_DIR}/model_{e}.pth')

        history['train_loss'].append(train_loss)
        history['eval_loss'].append(eval_loss)
        history['eval_metrics'].append({k: np.mean(v) for k, v in metrics_seq.items()})
        print(f'----- epoch: {e} -----')

        print(f'train loss: {train_loss:.6f}')
        print(f'eval  loss: {eval_loss:.6f}')
        print('eval metrics:')
        for k, v in metrics_seq.items():
            print(f'{k:>15}: {np.mean(v):8.4f} ± {np.std(v):8.4f}, median {np.median(v):.4f}')
        print()


plt.figure()
plt.plot(history['train_loss'], label='train loss')
plt.plot(history['eval_loss'], label='eval loss')
plt.legend()
plt.savefig(f'{CKPT_DIR}/loss.png')

plt.figure()
for k, v in history['eval_metrics'][0].items():
    plt.plot([x[k] for x in history['eval_metrics']], label=f'eval {k}')
plt.legend()
plt.savefig(f'{CKPT_DIR}/eval_metric.png')
# plt.show()

# plt.plot(history['train_loss'], label='train loss')
# plt.plot(history['eval_loss'], label='eval loss')
# plt.legend()
# plt.figure()
# for k, v in history['eval_metrics'][0].items():
#     plt.plot([x[k] for x in history['eval_metrics']], label=f'eval {k}')
# plt.legend()
# plt.show()

# %%
metrics_seq = {}
sum_test_loss = 0
gid2pred = {}

cnt = 0
model.eval()
for data_dict in tqdm(test_dl):
    test_loss, metrics, pred_dict = eval_step(model, data_dict, criterion, eval_type=EVAL_TYPE)
    sum_test_loss += test_loss

    # if data_dict['e1i'].shape[1] < 5:
    #     continue

    g1_step = data_dict['qry_step'][0].item()
    if g1_step not in gid2pred:
        gid2pred[g1_step] = {'pred': [pred_dict], 'data': [data_dict], 'eva': [metrics]}
    else:
        gid2pred[g1_step]['pred'].append(pred_dict)
        gid2pred[g1_step]['data'].append(data_dict)
        gid2pred[g1_step]['eva'].append(metrics)

    for k, v in metrics.items():
        if k not in metrics_seq:
            metrics_seq[k] = []
        metrics_seq[k].append(v)

    # FIXME: prevent CUDA OOM error
    data_dict_to_device(data_dict, 'cpu')
    cnt += 1

mean_test_loss = sum_test_loss / cnt  # len(test_dl)
print(f'mean_test loss: {mean_test_loss:.4f}')
print('test metrics:')
for k, v in metrics_seq.items():
    print(f'{k:>15}: {np.mean(v):8.4f} ± {np.std(v):8.4f}, median {np.median(v):.4f}')


mask_5cm = np.array(metrics_seq['t_rmse']) < 0.05
mask_10cm = np.array(metrics_seq['t_rmse']) < 0.1
mask_5deg = np.array(metrics_seq['r_err']) < 5
mask_10deg = np.array(metrics_seq['r_err']) < 10

mask_5cm5deg = np.logical_and(mask_5cm, mask_5deg)
mask_10cm5deg = np.logical_and(mask_10cm, mask_5deg)
print(f'5cm/5deg: {np.sum(mask_5cm5deg) / len(test_ds):.4f}')
print(f'10cm/5deg: {np.sum(mask_10cm5deg) / len(test_ds):.4f}')
