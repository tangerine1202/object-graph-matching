import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import copy
from glob import glob
from pprint import pprint
import pickle as pkl
import shutil

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.transforms.functional as VF
from torchvision.models import resnet50, ResNet50_Weights

from solo_tool import Solo


device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


visual_preprocess = ResNet50_Weights.DEFAULT.transforms(antialias=True)
visual_encoder = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device)


def frame_to_data(f, solo, k=5, bidirectional=False):
    seq_path = f.sequence_path
    cap = f.captures[0]
    metrics = f.metrics
    anno_defs = solo.annotation_definitions
    annos = cap.annotations
    data_dict = {}

    inst = annos['instance segmentation']
    inst.create_masks(seq_path)
    inst_df = inst.instances_df

    bbox = annos['bounding box']
    bbox_df = bbox.values_df

    meta = metrics['metadata']
    env_meta = meta.env_metadata
    meta_df = meta.instances_df

    # prepare object dataframe
    obj_df = meta_df.copy()
    # filter invisible objects
    if not inst.has_instance:
        return False, None
    obj_df = obj_df[obj_df['instanceId'].isin(inst_df['instanceId'])]  # filter out invisible objects
    # rename columns
    obj_df = obj_df.rename(columns={'object_labelName': 'label_name'})
    # add columns
    obj_df['label_id'] = obj_df['label_name'].apply(lambda x: anno_defs['bounding box'].name2id[x])
    pos_df = obj_df['object_absPos'].apply(pd.Series).rename(columns={0: 'pos_x', 1: 'pos_y', 2: 'pos_z'})
    obj_df = pd.concat((obj_df, pos_df), axis=1).drop(columns=['object_absPos'])
    # merge annotations
    bbox_df_for_merge = bbox_df.copy() \
        .drop(columns=['labelName', 'labelId']) \
        .add_prefix('bbox_') \
        .rename(columns={'bbox_instanceId': 'instanceId'})
    inst_df_for_merge = inst_df.copy() \
        .drop(columns=['labelName', 'labelId', 'color']) \
        .add_prefix('inst_') \
        .rename(columns={'inst_instanceId': 'instanceId'})
    obj_df = pd.merge(obj_df, bbox_df_for_merge, how='inner', left_on='instanceId',
                      right_on='instanceId', suffixes=('', '_duplicated'))
    obj_df = pd.merge(obj_df, inst_df_for_merge, how='inner', left_on='instanceId',
                      right_on='instanceId', suffixes=('', '_duplicated'))
    obj_df = obj_df.rename(columns={'instanceId': 'inst_id'})

    # filter out small object
    mask = obj_df['bbox_w'] * obj_df['bbox_h'] > 500
    obj_df = obj_df[mask].reset_index(drop=True)
    if len(obj_df) <= 1:
        return False, None

    # extract visual features
    bbox_embs = []
    rgb_path = f'{seq_path}/{cap.filename}'
    rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    rgb_img = VF.to_tensor(rgb_img)
    for i, row in obj_df.iterrows():
        top, left, w, h = int(row['bbox_y0']), int(row['bbox_x0']), int(row['bbox_w']), int(row['bbox_h'])
        crop_img = rgb_img[:, top:top + h, left:left + w]
        with torch.no_grad():
            proc_crop_img = visual_preprocess(crop_img.to(device).unsqueeze(0))
            bbox_emb = visual_encoder(proc_crop_img).squeeze().cpu().numpy()
        bbox_embs.append(bbox_emb)
    bbox_embs = np.stack(bbox_embs, axis=0)

    with torch.no_grad():
        proc_rgb_img = visual_preprocess(rgb_img.to(device).unsqueeze(0))
        rgb_emb = visual_encoder(proc_rgb_img).squeeze().cpu().numpy()

    graph_dict = obj_df_to_graph(obj_df, k=k, bidirectional=bidirectional)

    data_dict = {
        'step': f.step,
        'camera_pose': cap.camera_pose,
        'camera_intrinsics': cap.projectionMatrix,
        'bbox_embs': bbox_embs,
        'rgb_emb': rgb_emb,
        'node_df': graph_dict['node_df'],
        'edge_df': graph_dict['edge_df'],
        'total_node_count': len(graph_dict['node_df']),
        'total_edge_count': len(graph_dict['edge_df']),
        'node2inst': graph_dict['node2inst'],
    }

    return True, data_dict


def obj_df_to_graph(obj_df, k=5, bidirectional=False):
    node_df = obj_df.copy()
    node_df = pd.concat([pd.Series(node_df.index, name='node_id'), node_df], axis=1)
    edge_df = calc_edge_pair(node_df, k=k, bidirectional=bidirectional)
    node2inst = node_df[['node_id', 'inst_id']].to_dict()['inst_id']

    data_dict = {
        'node_df': node_df,
        'edge_df': edge_df,
        'node2inst': node2inst,
    }

    return data_dict


def calc_edge_pair(node_df, k=5, bidirectional=False):
    if len(node_df) < 2:
        return pd.DataFrame(columns=['node_id_src', 'node_id_dst', 'bbox_dist'])
    if len(node_df) <= k:
        k = len(node_df) - 1

    # construct edge pair id
    node_df_for_cross = node_df.copy()[['node_id', 'bbox_cx', 'bbox_cy']]
    cross_df = node_df_for_cross.merge(node_df_for_cross, how='cross', suffixes=('_src', '_dst'))
    # construct edge features
    diff = cross_df[['bbox_cx_dst', 'bbox_cy_dst']].values - cross_df[['bbox_cx_src', 'bbox_cy_src']].values
    cross_df['bbox_dist'] = np.linalg.norm(diff, axis=1)
    theta = np.arctan2(diff[:, 1], diff[:, 0])
    cross_df['bbox_sin'] = np.sin(theta)
    cross_df['bbox_cos'] = np.cos(theta)

    # bbox dist as edge weight
    dist_pivot = cross_df.pivot(index='node_id_dst', columns='node_id_src', values='bbox_dist')
    # find first 3 pair with minimum weight
    knn_bbox_dist = dist_pivot.apply(lambda x: x.nsmallest(k + 1).index).T
    edge_df = knn_bbox_dist.melt(id_vars=[0], value_vars=[i for i in range(1, k + 1)]
                                 )[[0, 'value']].rename(columns={0: 'node_id_src', 'value': 'node_id_dst'})

    # drop duplicated bi-directional edge
    if bidirectional:
        edge_df['small2large_id'] = edge_df.apply(lambda x: (x['node_id_src'], x['node_id_dst']) if (
            x['node_id_src'] < x['node_id_dst']) else (x['node_id_dst'], x['node_id_src']), axis=1)
        edge_df = edge_df.drop_duplicates(subset=['small2large_id'])
        edge_df = edge_df.drop(columns=['small2large_id'])

    edge_df = edge_df.merge(cross_df[['node_id_src', 'node_id_dst', 'bbox_dist', 'bbox_sin', 'bbox_cos']], on=[
                            'node_id_src', 'node_id_dst'], how='inner')

    return edge_df


def pair_graph(g1, g2):
    g1 = copy.deepcopy(g1)
    g2 = copy.deepcopy(g2)
    all_inst_ids = list(set(g1['node2inst'].values()) | set(g2['node2inst'].values()))
    anchor_inst_ids = list(set(g1['node2inst'].values()) & set(g2['node2inst'].values()))

    inst2g1node = {inst_id: node_id for node_id, inst_id in g1['node2inst'].items()}
    inst2g2node = {inst_id: (node_id + g1['total_node_count']) for node_id, inst_id in g2['node2inst'].items()}

    g1['node_df']['node_id'] = g1['node_df']['inst_id'].apply(lambda x: inst2g1node[x])
    g2['node_df']['node_id'] = g2['node_df']['inst_id'].apply(lambda x: inst2g2node[x])
    g1['edge_df']['node_id_src'] = g1['edge_df']['node_id_src'].apply(lambda x: inst2g1node[g1['node2inst'][x]])
    g1['edge_df']['node_id_dst'] = g1['edge_df']['node_id_dst'].apply(lambda x: inst2g1node[g1['node2inst'][x]])
    g2['edge_df']['node_id_src'] = g2['edge_df']['node_id_src'].apply(lambda x: inst2g2node[g2['node2inst'][x]])
    g2['edge_df']['node_id_dst'] = g2['edge_df']['node_id_dst'].apply(lambda x: inst2g2node[g2['node2inst'][x]])
    g_nodes = pd.concat([g1['node_df'], g2['node_df']], axis=0)
    g_edges = pd.concat([g1['edge_df'], g2['edge_df']], axis=0)
    g_bbox_embs = np.concatenate([g1['bbox_embs'], g2['bbox_embs']], axis=0)

    e1i = [inst2g1node[inst_id] for inst_id in anchor_inst_ids]
    e1j = [inst2g1node[inst_id] for inst_id in g1['node2inst'].values() if inst_id not in anchor_inst_ids]
    e2i = [inst2g2node[inst_id] for inst_id in anchor_inst_ids]
    e2j = [inst2g2node[inst_id] for inst_id in g2['node2inst'].values() if inst_id not in anchor_inst_ids]

    assert np.all([g_nodes['inst_id'].iloc[idx1] == g_nodes['inst_id'].iloc[idx2] for idx1, idx2 in zip(e1i, e2i)])

    total_node_count = g1['total_node_count'] + g2['total_node_count']
    overlap_count = len(e1i)

    return overlap_count, {
        'node_df': g_nodes,
        'edge_df': g_edges,
        'bbox_embs': g_bbox_embs,
        'g1_rgb_emb': g1['rgb_emb'],
        'g2_rgb_emb': g2['rgb_emb'],
        'e1i': e1i,
        'e1j': e1j,
        'e2i': e2i,
        'e2j': e2j,
        'e1i_count': len(e1i),
        'e1j_count': len(e1j),
        'e2i_count': len(e2i),
        'e2j_count': len(e2j),
        'total_obj_count': len(all_inst_ids),
        'total_node_count': total_node_count,
        'g1_node_count': g1['total_node_count'],
        'g2_node_count': g2['total_node_count'],
        'g1_edge_count': g1['total_edge_count'],
        'g2_edge_count': g2['total_edge_count'],
        'g1_camera_pose': g1['camera_pose'],
        'g2_camera_pose': g2['camera_pose'],
        'g1_camera_intrinsics': g1['camera_intrinsics'],
        'g2_camera_intrinsics': g2['camera_intrinsics'],
        'g1_step': g1['step'],
        'g2_step': g2['step'],
    }


if __name__ == '__main__':
    # SOLO_NAME = 'poisson3r8_vis'
    # SCENE = 'SimpleOffice'
    SCENE = 'WP16'
    SOLO_NAME = 'D_pois3r8'
    DATA_DIR = f'data/{SCENE}/{SOLO_NAME}'

    SINGLE_GRAPH_PATH = f'{DATA_DIR}/single_graph'
    PAIRED_GRAPH_PATH = f'{DATA_DIR}/paired_graph'
    if os.path.exists(SINGLE_GRAPH_PATH):
        print(f'remove {SINGLE_GRAPH_PATH}')
        shutil.rmtree(SINGLE_GRAPH_PATH)
    os.mkdir(SINGLE_GRAPH_PATH)
    if os.path.exists(PAIRED_GRAPH_PATH):
        print(f'remove {PAIRED_GRAPH_PATH}')
        shutil.rmtree(PAIRED_GRAPH_PATH)
    os.mkdir(PAIRED_GRAPH_PATH)

    solo = Solo(DATA_DIR)
    k = 5
    bidirectional = False

    for f in tqdm(solo.frames()):
        valid, data_dict = frame_to_data(f, solo, k, bidirectional)
        if not valid:
            print(f'frame {f.frame} is invalid')
            continue
        frame_path = os.path.join(SINGLE_GRAPH_PATH, f'{f.frame}.pkl')
        pkl.dump(data_dict, open(frame_path, 'wb'))

    paths = glob(os.path.join(SINGLE_GRAPH_PATH, '*.pkl'))
    for p1 in tqdm(paths):
        for p2 in paths:
            if p1 == p2:
                continue
            g1 = pkl.load(open(p1, 'rb'))
            g2 = pkl.load(open(p2, 'rb'))
            fname1 = os.path.basename(p1).split('.')[0]
            fname2 = os.path.basename(p2).split('.')[0]
            frame_path = os.path.join(PAIRED_GRAPH_PATH, f'{fname1}_{fname2}.pkl')

            overlap_count, g = pair_graph(g1, g2)
            num_edges = len(g['edge_df'])
            if overlap_count < 1 or num_edges < 1:
                continue
            if overlap_count < 4:
                continue
                # no enough matching for p3p

            pkl.dump(g, open(frame_path, 'wb'))
