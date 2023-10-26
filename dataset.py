import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import glob
import pickle as pkl

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as VF
from torch.utils.data import Dataset
from torch_geometric.data import Dataset

from solo2graph import Graph, PairedGraph


class CustomDataset(Dataset):
    def __init__(self, root, transform=None, max_len=None, min_overlap=1):
        self.root = root
        self.transform = transform
        self.max_len = max_len
        self.file_names = [os.path.basename(name) for name in glob.glob(os.path.join(root, '*.pkl'))]
        np.random.shuffle(self.file_names)

        if min_overlap > 1:
            self.filter_by_overlap(min_overlap=min_overlap)
        if max_len is not None and len(self.file_names) > max_len:
            self.file_names = np.random.choice(self.file_names, max_len, replace=False)

    def filter_by_overlap(self, min_overlap=1):
        new_file_names = []
        cnt = 0
        for name in self.file_names:
            pg_path = os.path.join(self.root, name)
            pg = pkl.load(open(pg_path, 'rb'))
            if len(pg.e1i) >= min_overlap:
                new_file_names.append(name)
                cnt += 1
            if self.max_len is not None and cnt >= self.max_len:
                break
        self.file_names = new_file_names

    def __getitem__(self, idx):
        pg_path = os.path.join(self.root, self.file_names[idx])
        data = pkl.load(open(pg_path, 'rb'))

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.file_names)

    def len(self): pass
    def get(self, idx): pass


def transform_data(pg):
    data = {}

    # node attr
    data['node_position'] = torch.cat((
        torch.tensor(pg.node_feat['bbox_cx'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox_cy'], dtype=torch.float)
    ), dim=1) / 320
    data['node_bbox'] = torch.cat((
        torch.tensor(pg.node_feat['bbox_cx'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox_cy'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox_h'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox_w'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox_h'] * pg.node_feat['bbox_w'], dtype=torch.float) / (320 ** 2),  # size
    ), dim=1) / 320
    # data['node_bbox3d'] = torch.cat((
    #     torch.tensor(pg.node_feat['bbox3d_tx'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_ty'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_tz'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_qx'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_qy'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_qz'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_qw'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_sx'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_sy'], dtype=torch.float),
    #     torch.tensor(pg.node_feat['bbox3d_sz'], dtype=torch.float),
    # ), dim=1)

    # data['node_img'] = torch.tensor(pg.node_feat['bbox_norm_image'], dtype=torch.float)
    data['node_text'] = torch.tensor(pg.node_feat['bbox_text'], dtype=torch.float)
    data['node_norm_text'] = torch.tensor(pg.node_feat['bbox_norm_text'], dtype=torch.float)

    # edge attr
    data['edge_index'] = torch.tensor(pg.edge_index, dtype=torch.long)
    # edge index
    data['edge_attr'] = torch.tensor(pg.edge_attr, dtype=torch.float)
    data['edge_attr'][:, 0] /= np.sqrt(320)

    data['n1'] = torch.tensor(pg.n1, dtype=torch.long)
    data['n2'] = torch.tensor(pg.n2, dtype=torch.long)
    data['e1i'] = torch.tensor(pg.e1i, dtype=torch.long)
    data['e2i'] = torch.tensor(pg.e2i, dtype=torch.long)
    data['e1j'] = torch.tensor(pg.e1j, dtype=torch.long)
    data['e2j'] = torch.tensor(pg.e2j, dtype=torch.long)

    data['g1_step'] = torch.tensor(pg.g1.step, dtype=torch.long)
    data['g2_step'] = torch.tensor(pg.g2.step, dtype=torch.long)
    data['g1_camera_pose'] = torch.tensor(pg.g1.camera_pose, dtype=torch.float)
    data['g2_camera_pose'] = torch.tensor(pg.g2.camera_pose, dtype=torch.float)

    # data['g1_camera_pose'] = torch.from_numpy(np.asarray(data['g1_camera_pose'], dtype=float))
    # data['g2_camera_pose'] = torch.from_numpy(np.asarray(data['g2_camera_pose'], dtype=float))
    # data['g1_camera_intrinsics'] = torch.from_numpy(np.asarray(data['g1_camera_intrinsics'], dtype=float))
    # data['g2_camera_intrinsics'] = torch.from_numpy(np.asarray(data['g2_camera_intrinsics'], dtype=float))
    return data
