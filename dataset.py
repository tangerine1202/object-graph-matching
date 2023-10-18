import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import glob
import pickle as pkl

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as VF
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset


class CustomDataset(Dataset):
    def __init__(self, root, scene, transform=None, ):
        self.root = root
        self.img_dir = 'sequence.0'
        self.file_names = sorted([os.path.basename(name)
                                 for name in glob.glob(os.path.join(root, '*.pkl'))])
        self.transform = transform

        self.scene = scene
        if self.scene == 'WP16':
            self.labels = ['chair', 'window', 'door', 'desk', 'potted plant', 'street sign', 'tv', 'table',
                           'hydrant', 'couch', 'sink', 'monitor', 'floor', 'wall', 'ceiling',]
        elif self.scene == 'SimpleOffice':
            self.labels = ['board', 'chair', 'door', 'louver', 'sink', 'storage', 'table', 'toilet', 'book',
                           'bottle', 'box', 'clock', 'cooler', 'cup', 'file box', 'keyboard', 'laptop', 'monitor',
                           'office phone', 'paper', 'pc', 'plant', 'poster', 'printer', 'socket', 'table lamp', 'trashcan',]
        # label2idx = {label: i for i, label in enumerate(labels)}

    def len(self): pass
    def get(self, idx): pass

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.root, self.file_names[idx])
        data = pkl.load(open(data_path, 'rb'))

        if self.transform:
            data = self.transform(data, self.labels)

        return data


def transform_data(data, labels):
    unwanted_cols = ['pos_x', 'pos_y', 'pos_z', 'bbox_x0', 'bbox_y0']

    # make label_idx to one-hot even label_idx is not exist in table
    label_id = data['node_df']['label_id'].values - 1
    label_id = np.eye(len(labels))[label_id]
    label_onehot_df = pd.DataFrame(
        label_id, columns=[f'label_id_{i}' for i in range(len(labels))], index=data['node_df'].index)
    data['node_df'] = pd.concat([data['node_df'], label_onehot_df], axis=1)

    # drop non-feature columns
    data['node_df'] = data['node_df'].drop(columns=['label_name', 'label_id', 'node_id'])
    data['node_df'] = data['node_df'].drop(columns=[col for col in data['node_df'].columns if col.startswith('inst_')])

    data['obj_pose_in_W'] = data['node_df'][['pos_x', 'pos_y', 'pos_z']].values
    # FIXME: drop unwanted node features
    data['node_df'] = data['node_df'].drop(columns=unwanted_cols)

    # FIXME: normalize bbox size
    if 'bbox' in data.keys():
        data['bbox'][['bbox_cx', 'bbox_cy', 'bbox_h', 'bbox_w']] /= 320
    data['edge_df']['bbox_dist'] /= 320 * np.sqrt(2)

    # FIXME: fix edge data type
    data['edge_df']['bbox_dist'] = data['edge_df']['bbox_dist'].astype(float)
    data['edge_df']['node_id_src'] = data['edge_df']['node_id_src'].astype(int)
    data['edge_df']['node_id_dst'] = data['edge_df']['node_id_dst'].astype(int)

    # to tensor
    data['bbox'] = torch.tensor(data['node_df'][['bbox_cx', 'bbox_cy', 'bbox_h', 'bbox_w']].values, dtype=torch.float)
    data['node_attr'] = torch.tensor(data['node_df'].values, dtype=torch.float)
    data['edge_index'] = torch.tensor(data['edge_df'][['node_id_src', 'node_id_dst']].values.T, dtype=torch.int64)
    data['edge_attr'] = torch.tensor(data['edge_df'][[col for col in data['edge_df'].columns if col not in [
                                     'node_id_src', 'node_id_dst']]].values, dtype=torch.float)

    for k, v in data.items():
        if type(v) == np.ndarray:
            data[k] = torch.from_numpy(v)
        elif type(v) == list:
            data[k] = torch.tensor(v)

    # data['e1i'] = torch.from_numpy(np.asarray(data['e1i'], dtype=int))
    # data['e2i'] = torch.from_numpy(np.asarray(data['e2i'], dtype=int))
    # data['e1j'] = torch.from_numpy(np.asarray(data['e1j'], dtype=int))
    # data['e2j'] = torch.from_numpy(np.asarray(data['e2j'], dtype=int))

    # data['bbox_embs'] = torch.tensor(data['bbox_embs'], dtype=torch.float)
    # data['g1_rgb_emb'] = torch.tensor(data['g1_rgb_emb'], dtype=torch.float)
    # data['g2_rgb_emb'] = torch.tensor(data['g2_rgb_emb'], dtype=torch.float)

    # data['g1_camera_pose'] = torch.from_numpy(np.asarray(data['g1_camera_pose'], dtype=float))
    # data['g2_camera_pose'] = torch.from_numpy(np.asarray(data['g2_camera_pose'], dtype=float))
    # data['g1_camera_intrinsics'] = torch.from_numpy(np.asarray(data['g1_camera_intrinsics'], dtype=float))
    # data['g2_camera_intrinsics'] = torch.from_numpy(np.asarray(data['g2_camera_intrinsics'], dtype=float))

    del data['node_df']
    del data['edge_df']
    return data
