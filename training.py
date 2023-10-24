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
from solo2graph import Graph, PairedGraph
from dataset import CustomDataset, transform_data
from models import CustomModel
from losses import CustomCriterion
from eva import compute_eval

# %%

EPOCHS = 500
EVAL_EPOCHS = 20

SOLO_NAME = 'poisson3r8'
SCENE = 'SimpleOffice'
# SOLO_NAME = 'D_pois3r8'
# SCENE = 'WP16'
DATA_DIR = f'data/{SCENE}/{SOLO_NAME}/paired_graph'
CKPT_DIR = f'ckpt/{SCENE}/{SOLO_NAME}'
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# %% [markdown]
#

# %%
ds = CustomDataset(root=DATA_DIR, transform=transform_data)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
dc = next(iter(dl))

# %%
# dc.keys()

# %%
# ds = CustomDataset(root=DATA_DIR, scene=SCENE)
# ds[0]['node_df'].columns

# %%
# # e1is = []
# len_e2 = []
# for dc in iter(dl):
#   e1is.append(len(dc['e1i'][0]))
#   len_e2.append(len(dc['e2i'][0]) + len(dc['e2j'][0]))

# len_e2 = np.array(len_e2)
# for k in [1, 3, 5]:
#   p_rand = np.clip(1 - (len_e2 - k) / len_e2, 0, 1)
#   print(f'rand hits@{k}: {p_rand.mean():.4f}, {p_rand.std():.4f}')

# %%
node_attr_dim = 1  # dc['node_attr'].shape[2]
# node_visual_dim = dc['bbox_embs'].shape[2]
edge_attr_dim = dc['edge_attr'].shape[2]
emb_dim = 128

print(f'node attr dim: {node_attr_dim}')
# print(f'node visual dim: {node_visual_dim}')
print(f'edge attr dim: {edge_attr_dim}')
print(f'emb dim: {emb_dim}')

# %%
ds = CustomDataset(root=DATA_DIR, transform=transform_data)
# random select 1000 samples from ds
# ds = torch.utils.data.Subset(ds, np.random.choice(len(ds), 1000, replace=False))
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
model = CustomModel(node_attr_dim, edge_attr_dim, emb_dim, match_threshold=0.2).to(device)
criterion = CustomCriterion(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

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

plt.plot(history['train_loss'], label='train loss')
plt.plot(history['eval_loss'], label='eval loss')
plt.legend()
plt.figure()
for k, v in history['eval_metrics'][0].items():
    plt.plot([x[k] for x in history['eval_metrics']], label=f'eval {k}')
plt.legend()
plt.show()

# %%
mean_metrics = {}
sum_test_loss = 0
gid2pred = {}

with torch.no_grad():
    model.eval()
    for data_dict in tqdm(test_dl):
        test_loss, metrics = eval_step(model, data_dict, criterion)
        sum_test_loss += test_loss

        pred_dict = model(data_dict)

        g1_step = data_dict['g1_step'][0].item()
        if g1_step not in gid2pred:
            gid2pred[g1_step] = {'pred': [pred_dict], 'data': [data_dict]}
        else:
            gid2pred[g1_step]['pred'].append(pred_dict)
            gid2pred[g1_step]['data'].append(data_dict)

        for k, v in metrics.items():
            if k in mean_metrics:
                mean_metrics[k] = mean_metrics[k] + v
            else:
                mean_metrics[k] = v


for k, v in mean_metrics.items():
    mean_metrics[k] = v / len(test_dl)

mean_test_loss = sum_test_loss / len(test_dl)
print(f'mean_test loss: {mean_test_loss:.4f}')
print('test metrics:')
for k, v in mean_metrics.items():
    print(f'{k}: {v:.4f}')

# %%
res = []

for gid, v in gid2pred.items():
    g1_pose = v['data'][0]['g1_camera_pose'][0]
    g1_step = v['data'][0]['g1_step'][0].item()

    pred = []

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

        # pred
        match_mask = (pred_dict['matches0'] != -1)
        if sum(match_mask) == 0:
            score = -np.inf
            continue
        else:
            match_score = pred_dict['matching_scores0'][match_mask].sort(descending=True)[0]
            score = torch.mean(match_score).item()
        pred.append((g2_step, score, g2_pose.numpy()))

        # gt
        pos_loss = np.linalg.norm(g1_pose[:3] - g2_pose[:3]).item()
        q_loss = np.linalg.norm(g1_pose[3:] - g2_pose[3:]).item()
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

    pred = sorted(pred, key=lambda x: x[1], reverse=True)

    # print(f'g1_step: {g1_step}')
    # print(f'gt_step: {gt_step}')
    # print(f'pred: {[p[0] for p in pred]}')
    # print()
    res.append({
        'g1_step': g1_step,
        'gt_step': gt_step,
        'gt_pose': gt_pose,
        'pred': pred,
    })


def compute_hits_at_k(gt, pred, k):
    gt_pose = gt
    for i in range(min(len(pred), k)):
        if np.allclose(gt_pose, pred[i][2]) and i < k:
            return True
    return False


res = sorted(res, key=lambda x: x['g1_step'])

mean_len_of_pred = np.mean([len(d['pred']) for d in res])
std_len_of_pred = np.std([len(d['pred']) for d in res])
max_len_of_pred = np.max([len(d['pred']) for d in res])
print(f'mean_len_of_pred: {mean_len_of_pred:.4f}')
print(f'std_len_of_pred: {std_len_of_pred:.4f}')
print(f'max_len_of_pred: {max_len_of_pred:.4f}')
print()

n_data = len(res)
n_no_match = sum([len(d['pred']) == 0 for d in res])
n_has_match = n_data - n_no_match
n_hits_at_1 = sum([compute_hits_at_k(d['gt_pose'], d['pred'], 1) for d in res])
n_hits_at_3 = sum([compute_hits_at_k(d['gt_pose'], d['pred'], 3) for d in res])
n_hits_at_5 = sum([compute_hits_at_k(d['gt_pose'], d['pred'], 5) for d in res])
n_hits_at_10 = sum([compute_hits_at_k(d['gt_pose'], d['pred'], 10) for d in res])
print(f'data len: {n_data}')
print(f'num of has , no matches: {n_has_match} , {n_no_match}')
print(f'hits@1: {n_hits_at_1 / (n_has_match if n_has_match > 0 else 1) * 100:.4f}%')
print(f'hits@3: {n_hits_at_3 / (n_has_match if n_has_match > 0 else 1) * 100:.4f}%')
print(f'hits@5: {n_hits_at_5 / (n_has_match if n_has_match > 0 else 1) * 100:.4f}%')
print(f'hits@10: {n_hits_at_10 / (n_has_match if n_has_match > 0 else 1) * 100:.4f}%')


# %%
# data_dict = list(iter(test_dl))[10]
# pred_dict = model(data_dict)
# cam_pose = data_dict['g1_camera_pose']

# with torch.no_grad():
#     metrics = compute_eval(pred_dict, data_dict)
#     for k, v in metrics.items():
#         if type(v) == torch.Tensor:
#             metrics[k] = v.item()
# pprint(metrics)

# # %%
# # turn e1i and e1j into numpy array
# e1i = data_dict['e1i'].squeeze(0).numpy()
# e1j = data_dict['e1j'].squeeze(0).numpy()
# e2i = data_dict['e2i'].squeeze(0).numpy() - data_dict['g1_node_count'].item()
# e2j = data_dict['e2j'].squeeze(0).numpy() - data_dict['g2_node_count'].item()
# e1 = np.concatenate([e1i, e1j])
# e2 = np.concatenate([e2i, e2j])
# print('g1 node count', data_dict['g1_node_count'].item())
# print('g2 node count', data_dict['g2_node_count'].item())
# print(f'e1i: {e1i.shape}, e2i: {e2i.shape}, e1j: {e1j.shape}, e2j: {e2j.shape}')
# # print(f'e1: {e1}')
# # print(f'e2: {e2}')

# # %%
# gt_match = [[i, j] for i, j in zip(e1i, e2i)]
# pred_match = [[i, j.item()] for i, j in enumerate(pred_dict['matches0']) if j != -1]

# print(f'gt match:')
# pprint(gt_match)
# print(f'pred match:')
# pprint(pred_match)

# # %%
# # gt_match = [[i, j] for i, j in zip(e1i, e2i)]
# # print(f'gt match:')
# # pprint(gt_match)

# # g1_node_count = data_dict['g1_node_count'][0].detach().cpu().numpy()
# # emb = pred_dict['joint_embs'].squeeze(0).detach().cpu().numpy()
# # emb = emb / np.linalg.norm(emb, axis=1)[:, None]
# # emb1 = emb[:data_dict['g1_node_count']]
# # emb2 = emb[data_dict['g1_node_count']:]
# # W = emb1 @ emb2.T
# # corr_ids = np.argmax(W, axis=1) + g1_node_count

# # G = nx.Graph()
# # for i in range(W.shape[0]):
# #     for j in range(W.shape[1]):
# #         G.add_edge(i, j + g1_node_count, weight=W[i, j])

# # match = nx.bipartite.minimum_weight_full_matching(G, corr_ids)
# # match = [[i, j] for i, j in match.items() if i < data_dict['g1_node_count'].item()]
# # print(f'pred match:')
# # pprint(match)

# # match_G = nx.Graph()
# # for i in match.keys():
# #     j = match[i]
# #     match_G.add_edge(i, j, weight=G[i][j]['weight'])
# # pos = nx.bipartite_layout(match_G, match.keys())
# # nx.draw(match_G, pos, node_color='lightblue', with_labels=True, node_size=500)

# # %%


# # %%
