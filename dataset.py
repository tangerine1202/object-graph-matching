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
        qry_fname = self.pair_df.iloc[idx]['qry_fname']
        map_fname = self.pair_df.iloc[idx]['map_fname']
        qry_graph = pkl.load(open(os.path.join(self.graph_root, qry_fname), 'rb'))
        map_graph = pkl.load(open(os.path.join(self.graph_root, map_fname), 'rb'))
        paired_graph = PairedGraph(qry_graph, map_graph)
        if self.transform:
            paired_graph = self.transform(paired_graph)
        return paired_graph

    def __len__(self):
        return len(self.pair_df)

    def get(self, idx): pass
    def len(self): pass


def transform_3D_qm_data(pg):
    data = {}
    qry_g = pg.g1
    map_g = pg.g2

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
    data['edge_index'] = torch.tensor(pg.edge_index['3d'], dtype=torch.long)
    # edge index
    data['edge_attr'] = torch.tensor(pg.edge_attr['3d'], dtype=torch.float)

    data['n1'] = torch.tensor(pg.n1, dtype=torch.long)
    data['n2'] = torch.tensor(pg.n2, dtype=torch.long)
    data['e1i'] = torch.tensor(pg.e1i, dtype=torch.long)
    data['e2i'] = torch.tensor(pg.e2i, dtype=torch.long)
    data['e1j'] = torch.tensor(pg.e1j, dtype=torch.long)
    data['e2j'] = torch.tensor(pg.e2j, dtype=torch.long)

    data['qry_step'] = torch.tensor(qry_g.step, dtype=torch.long)
    data['qry_camera_pose'] = torch.tensor(qry_g.camera_pose, dtype=torch.float)
    data['qry_camera_intrinsics'] = torch.tensor(qry_g.camera_intrinsics, dtype=torch.float)
    return data


def transform_2D_qq_data(pg):
    data = {}
    qry_g = pg.g1
    map_g = pg.g2

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
    data['edge_index'] = torch.tensor(pg.edge_index['2d'], dtype=torch.long)
    # edge index
    data['edge_attr'] = torch.tensor(pg.edge_attr['2d'], dtype=torch.float)
    data['edge_attr'][:, 0] /= np.sqrt(320)

    data['n1'] = torch.tensor(pg.n1, dtype=torch.long)
    data['n2'] = torch.tensor(pg.n2, dtype=torch.long)
    data['e1i'] = torch.tensor(pg.e1i, dtype=torch.long)
    data['e2i'] = torch.tensor(pg.e2i, dtype=torch.long)
    data['e1j'] = torch.tensor(pg.e1j, dtype=torch.long)
    data['e2j'] = torch.tensor(pg.e2j, dtype=torch.long)

    data['qry_step'] = torch.tensor(qry_g.step, dtype=torch.long)
    data['map_step'] = torch.tensor(map_g.step, dtype=torch.long)
    data['qry_camera_pose'] = torch.tensor(qry_g.camera_pose, dtype=torch.float)
    data['map_camera_pose'] = torch.tensor(map_g.camera_pose, dtype=torch.float)
    return data
