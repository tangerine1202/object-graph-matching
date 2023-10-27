import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import glob
import pickle as pkl

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as VF
from torch_geometric.data import Dataset
from torch.utils.data import IterableDataset

from solo2graph import MapGraph, QueryGraph, PairedGraph


class PairListDataset(Dataset):
    def __init__(self, root, pair_fname, transform=None, graph_dir='graph', min_overlap=1):
        super(PairListDataset).__init__()
        self.root = root
        self.graph_root = os.path.join(root, graph_dir)
        self.transform = transform
        self.min_overlap = min_overlap
        self.pair_fname = pair_fname

        self.pair_df = pd.read_csv(os.path.join(root, pair_fname))
        self.pair_df = self.pair_df[self.pair_df['n_overlap'] >= min_overlap]

    def __getitem__(self, idx):
        g1_fname = self.pair_df.iloc[idx]['g1_fname']
        g2_fname = self.pair_df.iloc[idx]['g2_fname']
        g1_graph = pkl.load(open(os.path.join(self.graph_root, g1_fname), 'rb'))
        g2_graph = pkl.load(open(os.path.join(self.graph_root, g2_fname), 'rb'))
        paired_graph = PairedGraph(g1_graph, g2_graph)
        if self.transform:
            paired_graph = self.transform(paired_graph)
        return paired_graph

    def __len__(self):
        return len(self.pair_df)

    def get(self, idx): pass
    def len(self): pass


def transform_3D_mq_data(pg):
    data = {}
    mg = pg.g1
    qg = pg.g2

    # node attr
    data['node_bbox3d'] = torch.cat((
        torch.tensor(pg.node_feat['bbox3d_tx'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_ty'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_tz'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_qx'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_qy'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_qz'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_qw'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_sx'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_sy'], dtype=torch.float),
        torch.tensor(pg.node_feat['bbox3d_sz'], dtype=torch.float),
    ), dim=1)

    data['node_text'] = torch.tensor(pg.node_feat['text_embs'], dtype=torch.float)
    data['node_norm_text'] = torch.tensor(pg.node_feat['norm_text_embs'], dtype=torch.float)

    # edge attr
    data['edge_index'] = torch.tensor(pg.edge_index, dtype=torch.long)
    # edge index
    data['edge_attr'] = torch.tensor(pg.edge_attr['3d'], dtype=torch.float)

    data['n1'] = torch.tensor(pg.n1, dtype=torch.long)
    data['n2'] = torch.tensor(pg.n2, dtype=torch.long)
    data['e1i'] = torch.tensor(pg.e1i, dtype=torch.long)
    data['e2i'] = torch.tensor(pg.e2i, dtype=torch.long)
    data['e1j'] = torch.tensor(pg.e1j, dtype=torch.long)
    data['e2j'] = torch.tensor(pg.e2j, dtype=torch.long)

    data['query_step'] = torch.tensor(qg.step, dtype=torch.long)
    data['query_camera_pose'] = torch.tensor(qg.camera_pose, dtype=torch.float)
    data['query_camera_intrinsics'] = torch.tensor(qg.camera_intrinsics, dtype=torch.float)
    return data


def transform_2D_qq_data(pg):
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

    data['node_text'] = torch.tensor(pg.node_feat['text_embs'], dtype=torch.float)
    data['node_norm_text'] = torch.tensor(pg.node_feat['norm_text_embs'], dtype=torch.float)

    # edge attr
    data['edge_index'] = torch.tensor(pg.edge_index, dtype=torch.long)
    # edge index
    data['edge_attr'] = torch.tensor(pg.edge_attr['2d'], dtype=torch.float)
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

    # data['g1_camera_intrinsics'] = torch.from_numpy(np.asarray(data['g1_camera_intrinsics'], dtype=float))
    # data['g2_camera_intrinsics'] = torch.from_numpy(np.asarray(data['g2_camera_intrinsics'], dtype=float))
    return data
