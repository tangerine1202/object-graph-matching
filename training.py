# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
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
from torch_geometric.loader import DataLoader, PrefetchLoader

# Visualization
import cv2
import matplotlib.pyplot as plt

# %%
from dataset import (
    PairListDataset,
    transform_2Dto3D_qm_data,
    transform_3Dto3D_qm_data,
)
from models import (
    Model_2Dto3D,
    Model_3Dto3D,
)
from losses import CustomCriterion
from eva import compute_eval

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--data_src', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--text_src', type=str)
parser.add_argument('--qry_edge_type', type=str)
parser.add_argument('--map_bbox3d_type', type=str)
parser.add_argument('--map_edge_type', type=str)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--eval_epochs', type=int, default=5)
args = parser.parse_args()


DATA_SRC = args.data_src
TASK = args.task

TEXT_SRC = args.text_src
QRY_EDGE_TYPE = args.qry_edge_type
MAP_BBOX3D_TYPE = args.map_bbox3d_type
MAP_EDGE_TYPE = args.map_edge_type
if TASK == '2Dto3D':
    TASK = f'2D_{QRY_EDGE_TYPE}_to_3D_{MAP_BBOX3D_TYPE}_{MAP_EDGE_TYPE}'
elif TASK == '3Dto3D':
    TASK = f'3D_{QRY_EDGE_TYPE}_to_3D_{MAP_BBOX3D_TYPE}_{MAP_EDGE_TYPE}'
else:
    raise NotImplementedError

TOTAL_EPOCHS = args.total_epochs
EVAL_EPOCHS = args.eval_epochs

EMB_DIM = 128
MIN_OVERLAP = 3
MATCH_THRESHOLD = 0.2

MIN_OVERLAP = 3
TRAIN_RATIO, EVAL_RATIO, TEST_RATIO = 0.5, 0.2, 0.3

if DATA_SRC == 'SimpleOffice':
    SOLO_NAME = f'poisson{input("solo name: poisson")}'
    SCENE = 'SimpleOffice'
    DATA_DIR = f'data/{SCENE}/{SOLO_NAME}'
    GRAPH_DIR = 'graph'
    CKPT_DIR = f'ckpt/{SCENE}/{SOLO_NAME}/{TASK}'
elif DATA_SRC == 'ScanNet':
    N_SCENES = f'{input("n_scenes: ")}_scenes'
    DATA_DIR = f'data/scannet_graph/{N_SCENES}'
    GRAPH_DIR = ''
    CKPT_DIR = f'ckpt/scannet_graph/{N_SCENES}/{TASK}'
else:
    raise NotImplementedError

if TEXT_SRC == 'onehot':
    if DATA_SRC == 'SimpleOffice':
        TEXT_DIM = 27
    elif DATA_SRC == 'ScanNet':
        TEXT_DIM = 40
elif TEXT_SRC == 'glove':
    TEXT_DIM = 100
elif TEXT_SRC == 'blip':
    TEXT_DIM = 768
elif TEXT_SRC == 'no_text':
    TEXT_DIM = 0
else:
    raise NotImplementedError

PAIR_FNAME = 'qm_paired_list.csv'
if TASK.startswith('2D'):
    MODEL = Model_2Dto3D
    TRANSFORM = transform_2Dto3D_qm_data
elif TASK.startswith('3D'):
    MODEL = Model_3Dto3D
    TRANSFORM = transform_3Dto3D_qm_data
else:
    raise NotImplementedError


if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

print('task:', TASK)


# %%
print('--- dataset ---')
ds = PairListDataset(root=DATA_DIR, pair_fname=PAIR_FNAME, transform=TRANSFORM,
                     text_src=TEXT_SRC,
                     qry_edge_type=QRY_EDGE_TYPE,
                     map_bbox3d_type=MAP_BBOX3D_TYPE, map_edge_type=MAP_EDGE_TYPE,
                     graph_dir=GRAPH_DIR, min_overlap=MIN_OVERLAP)
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
    if torch.isnan(loss):
        print('loss is nan, skip this batch')
        return None
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
    if torch.isnan(loss):
        print('loss is nan, skip this batch')
        return None, None, None
    metrics = compute_eval(pred_dict, data_dict)
    return loss.item(), metrics, pred_dict


# %%
model = MODEL(emb_dim=EMB_DIM,
              text_dim=TEXT_DIM,
              qry_edge_type=QRY_EDGE_TYPE,
              map_bbox3d_type=MAP_BBOX3D_TYPE, map_edge_type=MAP_EDGE_TYPE,
              match_threshold=MATCH_THRESHOLD).to(device)
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
        train_step_loss = train_step(model, data_dict, optimizer, criterion)
        if train_step_loss is None:
            continue
        train_loss += train_step_loss
    train_loss /= len(train_dl)

    if e % EVAL_EPOCHS == 0:
        metrics_seq = {}
        sum_eval_loss = 0

        with torch.no_grad():
            model.eval()
            for data_dict in eval_dl:
                eval_loss, metrics, pred_dict = eval_step(model, data_dict, criterion)
                if eval_loss is None:
                    continue
                sum_eval_loss += eval_loss

            for k, v in metrics.items():
                if k not in metrics_seq:
                    metrics_seq[k] = []
                metrics_seq[k].append(v)

        eval_loss = sum_eval_loss / len(eval_dl)

        ckpt_fpath = f'{CKPT_DIR}/model_{TEXT_SRC}_{e}.pth'
        print(f'save checkpoint: {ckpt_fpath}')
        torch.save(model.state_dict(), ckpt_fpath)

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
plt.savefig(f'{CKPT_DIR}/loss_{TEXT_SRC}.png')

plt.figure()
for k, v in history['eval_metrics'][0].items():
    if k in ['pre', 'rec', 'f1']:
        plt.plot([x[k] for x in history['eval_metrics']], label=f'eval {k}')
plt.legend()
plt.savefig(f'{CKPT_DIR}/eval_metric_{TEXT_SRC}.png')
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
    test_loss, metrics, pred_dict = eval_step(model, data_dict, criterion)
    if test_loss is None:
        continue
    sum_test_loss += test_loss

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


pose_src = 'pose' if TASK == '2Dto3D' else 'pose_from_3D'
mask_5cm = np.array(metrics_seq[f'{pose_src}_t_rmse']) < 0.05
mask_10cm = np.array(metrics_seq[f'{pose_src}_t_rmse']) < 0.1
mask_5deg = np.array(metrics_seq[f'{pose_src}_r_err']) < 5
mask_10deg = np.array(metrics_seq[f'{pose_src}_r_err']) < 10

mask_5cm5deg = np.logical_and(mask_5cm, mask_5deg)
mask_10cm5deg = np.logical_and(mask_10cm, mask_5deg)
print(f'{pose_src}  5cm/5deg: {np.sum(mask_5cm5deg) / len(mask_10cm5deg):.4f}')
print(f'{pose_src} 10cm/5deg: {np.sum(mask_10cm5deg) / len(mask_10cm5deg):.4f}')
