import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import json
import glob
from pathlib import Path
from pprint import pprint
import pickle as pkl

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
if torch.backends.mps.is_available():
    device = torch.device('mps')
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device('cuda')
print(f'torch device: {device}')

# GNN package
import torch_geometric.nn as pyg
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv

labels = ['chair', 'window', 'door', 'desk', 'potted plant', 'street sign', 'tv',
          'dining table', 'couch', 'sink', 'monitor', 'floor', 'wall', 'ceiling', 'fire hydrant']
label2idx = {label: i for i, label in enumerate(labels)}

# ----- dataset ------


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.file_names = sorted([os.path.basename(name) for name in glob.glob(os.path.join(root, '*.pkl'))])
        self.transform = transform

    def len(self): pass
    def get(self, idx): pass

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.root, self.file_names[idx])
        data = pkl.load(open(data_path, 'rb'))

        if self.transform:
            data = self.transform(data)

        return data


def transform_data(data):
    data['node_df']['label_idx'] = data['node_df']['label_name'].apply(lambda x: label2idx[x])

    # make label_idx to one-hot even label_idx is not exist in table
    label_idx = data['node_df']['label_idx'].values
    label_idx = np.eye(len(labels))[label_idx]
    onehot_df = pd.DataFrame(label_idx, columns=[f'label_idx_{i}' for i in range(len(labels))])
    data['node_df'] = pd.concat([data['node_df'], onehot_df], axis=1)

    data['node_df'] = data['node_df'].drop(columns=['label_name', 'label_idx', 'obj_id', 'node_id', 'x', 'y', 'z'])
    data['edge_df'] = data['edge_df'].drop(columns=['obj_id_src', 'obj_id_dst'])

    graph = Data(
        x=torch.tensor(data['node_df'].values, dtype=torch.float),
        edge_index=torch.tensor(data['edge_df'][['node_id_src', 'node_id_dst']].values.T, dtype=torch.long),
        edge_attr=torch.tensor(data['edge_df'][[col for col in data['edge_df'].columns if col not in [
                               'node_id_src', 'node_id_dst']]].values, dtype=torch.float),
    )
    data['graph'] = graph

    data['e1i'] = torch.from_numpy(np.asarray(data['e1i']))
    data['e2i'] = torch.from_numpy(np.asarray(data['e2i']))
    data['e1j'] = torch.from_numpy(np.asarray(data['e1j']))
    data['e2j'] = torch.from_numpy(np.asarray(data['e2j']))

    del data['node_df']
    del data['edge_df']
    return data


# ----- model -----
class CustomModel(nn.Module):
    def __init__(self, in_channels, edge_attr_channels, hidden_channels, out_channels, num_layers=2):
        super(CustomModel, self).__init__()
        self.norm = nn.InstanceNorm1d(in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        # self.edge_lin1 = nn.Linear(edge_attr_channels, hidden_channels)
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_attr_channels)
        # self.gats = nn.ModuleList()
        # for i in range(num_layers - 2):
        #   self.gats.append(GATv2Conv(hidden_channels, hidden_channels))
        self.gat2 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_attr_channels)

    def forward(self, x, edge_index, edge_attr):
        # edge_attr = ((edge_attr - edge_attr.mean()) / edge_attr.std())
        # edge_attr = edge_attr.unsqueeze(1)
        # edge_attr = self.edge_lin1(edge_attr)
        # edge_attr = edge_attr.squeeze(1)

        x = self.norm(x)
        x = self.lin1(x)
        x = x.relu()
        x = self.gat1(x, edge_index, edge_attr)
        x = x.relu()
        # for gat in self.gats:
        #   x = gat(x, edge_index, edge_attr)
        #   x = x.relu()
        x = self.gat2(x, edge_index, edge_attr)
        return x


def train_step(model, data_dict, optimizer, icl_loss_fn):
    graph = data_dict['graph']
    model.train()
    optimizer.zero_grad()
    emb = model(graph.x, graph.edge_index, graph.edge_attr)
    loss = icl_loss_fn(emb, data_dict)
    loss.backward()
    optimizer.step()
    return loss.item()


# ----- loss -----

def calculate_prob_dist(e1i, e2i, e1j, e2j, temp):
    # FIXME: remove the squeeze(0) when batch_size>1
    e1i = e1i.squeeze(0)
    e2i = e2i.squeeze(0)
    e1j = e1j.squeeze(0)
    e2j = e2j.squeeze(0)

    deltaM_e1i_e2i = torch.exp(torch.matmul(e1i, torch.transpose(e2i, 0, 1)) / temp)
    deltaM_e1i_e1j = torch.exp(torch.matmul(e1i, torch.transpose(e1j, 0, 1)) / temp)
    deltaM_e1i_e2j = torch.exp(torch.matmul(e1i, torch.transpose(e2j, 0, 1)) / temp)

    deltaM_e1i_e2i_e1j = deltaM_e1i_e2i / (deltaM_e1i_e1j.sum() + 1e-9)
    deltaM_e1i_e2i_e2j = deltaM_e1i_e2i / (deltaM_e1i_e2j.sum() + 1e-9)
    q_e1i_e2i_inverse = 1.0 + 1.0 / (deltaM_e1i_e2i_e1j + 1e-9) + 1.0 / (deltaM_e1i_e2i_e2j + 1e-9)
    q_e1i_e2i = 1.0 / (q_e1i_e2i_inverse + 1e-9)

    return q_e1i_e2i


class ICLLoss(nn.Module):
    def __init__(self, device, temperature=0.1, alpha=0.5):
        super(ICLLoss, self).__init__()
        self.temp = 0.1  # temperature
        self.alpha = alpha
        self.device = device

    def forward(self, emb, data_dict):
        emb = F.normalize(emb, dim=1)
        e1i = emb[data_dict['e1i']]
        e2i = emb[data_dict['e2i']]
        e1j = emb[data_dict['e1j']]
        e2j = emb[data_dict['e2j']]

        qm_e1i_e2i = calculate_prob_dist(e1i, e2i, e1j, e2j, self.temp)
        qm_e2i_e1i = calculate_prob_dist(e2i, e1i, e2j, e1j, self.temp)

        lossA = qm_e1i_e2i
        lossB = qm_e2i_e1i

        loss = self.alpha * lossA + (1 - self.alpha) * lossB
        loss = -torch.log(loss).mean()
        return loss

# ----- evaluation -----


def compute_hits_k(rank_list, e1i_idxs, e2i_idxs, k=1):
    rank_list = rank_list.detach().cpu().numpy()
    correct, total = 0, 0
    for idx, e1i_idx in enumerate(e1i_idxs):
        e1_idx_rank_list = list(rank_list[e1i_idx])
        e1_idx_rank_list.remove(e1i_idx)
        e1_idx_rank_list_k = e1_idx_rank_list[:k]

        if e2i_idxs[idx] in e1_idx_rank_list_k:
            correct += 1

    total = e1i_idxs.shape[0]

    return correct, total


def compute_eval(emb, data_dict):
    e1i = data_dict['e1i']
    e2i = data_dict['e2i']
    # FIXME: remove the squeeze(0) when batch_size>1
    e1i = e1i.squeeze(0)
    e2i = e2i.squeeze(0)

    emb = emb / emb.norm(dim=1)[:, None]
    dist = 1 - torch.mm(emb, emb.transpose(0, 1))
    rank_list = torch.argsort(dist, dim=1)

    metrics = {}
    all_k = [1, 2, 3, 4, 5]
    for k in all_k:
        correct, total = compute_hits_k(rank_list, e1i, e2i, k)
        metrics[f'hits@{k}'] = correct / total
    return metrics


if __name__ == '__main__':
    SOLO_NAME = 'Rarea_poisson3'
    DATA_DIR = f'./data/{SOLO_NAME}/pair'

    num_node_features = 15
    num_edge_features = 3
    # print('num nodes:', graph.num_nodes)
    # print('num edges:', graph.num_edges)
    print('num node features:', num_node_features)
    print('num edge features:', num_edge_features)

    ds = CustomDataset(root=DATA_DIR, transform=transform_data)
    train_ds, eval_ds, test_ds = torch.utils.data.random_split(ds, [0.5, 0.2, 0.3])

    # Do not use batch_size>1 for now, it will crash the e1i, e1j, e2i, e2j indices
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print('train size:', len(train_ds))
    print('eval size:', len(eval_ds))
    print('test size:', len(test_ds))

    temp = 0.1
    lr = 5e-4
    hidden_channels = 64
    output_channels = 32

    icl_loss_fn = ICLLoss(device, temp)
    model = CustomModel(num_node_features, num_edge_features, hidden_channels, output_channels, num_layers=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ----- train -----
    num_epochs = 1000
    for epoch in tqdm(range(num_epochs)):
        train_losses = []
        for data_dict in train_dl:
            loss = train_step(model, data_dict, optimizer, icl_loss_fn)
            train_losses.append(loss)

        if epoch % 20 != 0:
            continue

        print(f'epoch: {epoch}')
        print(f'train loss: {np.mean(train_losses)}')

        eval_losses = []
        eval_metrics = {}
        for data_dict in eval_dl:
            model.eval()
            with torch.no_grad():
                graph = data_dict['graph']
                emb = model(graph.x, graph.edge_index, graph.edge_attr)
                loss = icl_loss_fn(emb, data_dict)
                eval_losses.append(loss.item())

                metrics = compute_eval(emb, data_dict)
                for k, v in metrics.items():
                    if k not in eval_metrics:
                        eval_metrics[k] = []
                    eval_metrics[k].append(v)
        print(f'eval loss: {np.mean(eval_losses)}')
        for k, v in eval_metrics.items():
            print(f'{k}: {np.mean(v):.4f}')
