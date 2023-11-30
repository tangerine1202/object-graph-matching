import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from pprint import pprint
import pickle as pkl
import glob
import shutil
import copy
import json

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_R
from tqdm.auto import tqdm
np.set_printoptions(suppress=True)

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d

from preprocess import MapGraph, QueryGraph
from preprocess.scannet_utils import ScanNetLoader

from preprocess.utils import (
    GloveFeatureExtractor,
    LAVISFeatureExtractor,
    comp_bbox3d_edge_with_min_knn,
    comp_bbox2d_edge_with_min_knn,
)

from utils import (
    comp_graph_overlap,
    invert_Rt,
)


def encode_text_features(label_names, label_ids, type, n_labels=None):
    if type == 'blipv2':
        feats = []
        for label_name in label_names:
            text_features = lavis_feature_extractor.extract_text_features(label_name)
            text_feature = text_features['text']
            norm_text_feature = text_features['norm_text']
            feats.append(text_feature)
        feats = np.vstack(feats)
    elif type == 'glove':
        feats = []
        for label_name in label_names:
            text_feature = glove_feature_extractor.extract_feature(label_name)
            feats.append(text_feature)
        feats = np.vstack(feats)
    elif type == 'onehot':
        assert n_labels is not None
        feats = np.zeros((len(label_ids), n_labels))
        feats[np.arange(len(label_ids)), label_ids - 1] = 1
    return feats

# %%


def scannet_to_node_map_graph(scene_id: str):
    # scene_id = SCENE_ID
    # camera pose
    bbox3d_df, align_matrix = scannet_loader.get_bbox3d_and_align_matrix(scene_id)
    cam_t, cam_R = scannet_loader.get_cam_pose(scene_id, 0, align_matrix)
    if np.any(np.isnan(cam_t)) or np.any(np.isnan(cam_R.as_quat())) or np.any(np.isinf(cam_t)) or np.any(np.isinf(cam_R.as_quat())):
        return None
    cam_pose = np.concatenate([cam_t, cam_R.as_quat()], axis=0)
    # camera intrinsics
    intrinsics = scannet_loader.get_intrinsics(scene_id)

    # instance id
    map_df = bbox3d_df.copy()
    map_df[['dx', 'dy', 'dz']] = map_df[['dx', 'dy', 'dz']].values / 2
    map_df = map_df.rename(columns={
        'x': 'bbox3d_tx', 'y': 'bbox3d_ty', 'z': 'bbox3d_tz',
        'dx': 'bbox3d_sx', 'dy': 'bbox3d_sy', 'dz': 'bbox3d_sz',
        'c': 'label_id',
    })
    # bbox3d rotation
    # NOTE: the bbox3d is axis-aligned, so the rotation is identity
    bbox3d_q = np.tile([0, 0, 0, 1], (len(map_df), 1))
    map_df = map_df.assign(
        bbox3d_qx=bbox3d_q[:, 0], bbox3d_qy=bbox3d_q[:, 1], bbox3d_qz=bbox3d_q[:, 2], bbox3d_qw=bbox3d_q[:, 3])

    masks = [(label_id in scannet_loader.label_table.index) for label_id in map_df['label_id'].values]
    map_df = map_df[masks]
    if len(map_df) == 0:
        return None
    label_ids = map_df['label_id'].values
    label_names = scannet_loader.get_label_names(label_ids, 'nyu40class')
    map_df['label_name'] = label_names

    # text features
    onehot_label_embs = encode_text_features(
        label_names, label_ids, 'onehot', n_labels=40)
    glove_label_embs = encode_text_features(label_names, label_ids, 'glove')
    blip_label_embs = encode_text_features(label_names, label_ids, 'blipv2')

    inst_ids = np.arange(len(map_df))
    node_ids = np.arange(len(map_df))
    inst2node = dict(zip(inst_ids, node_ids))
    map_df['inst_id'] = inst_ids
    map_df['node_id'] = node_ids

    features = {
        **{k: np.asarray(v).reshape(-1, 1) for k, v in map_df.items()},
        'blip_label_embs': blip_label_embs,
        'glove_label_embs': glove_label_embs,
        'onehot_label_embs': onehot_label_embs,
    }

    data = {
        'features': features,
        'total_obj_cnt': len(map_df),
        'inst_ids': inst_ids,
        'node_ids': node_ids,
        'inst2node': inst2node,
        'camera_pose': cam_pose,
        'camera_intrinsics': intrinsics,
    }
    return data

# %%


def scannet_to_node_qry_graph(scene_id: str, frame_id: int):
    # camera pose
    glb_bbox3d_df, align_matrix = scannet_loader.get_bbox3d_and_align_matrix(scene_id)
    t_wc, R_wc = scannet_loader.get_cam_pose(scene_id, frame_id, align_matrix)
    if np.any(np.isnan(t_wc)) or np.any(np.isnan(R_wc.as_quat())) or np.any(np.isinf(t_wc)) or np.any(np.isinf(R_wc.as_quat())):
        return None
    # R_cw, t_cw = invert_Rt(R_wc, t_wc)
    cam_pose = np.concatenate([t_wc, R_wc.as_quat()], axis=0)
    # camera intrinsics
    intrinsics = scannet_loader.get_intrinsics(scene_id)

    # instance id
    glb_inst_ids = np.arange(len(glb_bbox3d_df))
    glb_bbox3d_df['inst_id'] = glb_inst_ids

    # in view
    masks = scannet_loader.get_in_view_masks(glb_bbox3d_df, t_wc, R_wc, intrinsics)
    if masks.sum() == 0:
        return None

    qry_df = glb_bbox3d_df[masks].copy()
    qry_df = qry_df.rename(columns={
        'c': 'label_id',
    })

    # bbox3d_world_t_df = qry_df[['x', 'y', 'z']].copy()
    # bbox3d_world_s_df = qry_df[['dx', 'dy', 'dz']].copy() / 2
    # bbox3d_world_q_df = pd.DataFrame(np.tile([0, 0, 0, 1], (len(qry_df), 1)), index=qry_df.index, columns=['qx', 'qy', 'qz', 'qw'])
    # bbox3d_world_df = pd.concat([bbox3d_world_t_df, bbox3d_world_q_df, bbox3d_world_s_df], axis=1)
    # bbox3d_local_df = pd.DataFrame(transform_bbox3d(bbox3d_world_df.values, R_cw, t_cw), columns=[
    #     'bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz',
    #     'bbox3d_qx', 'bbox3d_qy', 'bbox3d_qz', 'bbox3d_qw',
    #     'bbox3d_sx', 'bbox3d_sy', 'bbox3d_sz'], index=qry_df.index)
    # assert len(qry_df) == len(bbox3d_local_df), f'{len(qry_df)} != {len(bbox3d_local_df)}'
    # qry_df = pd.concat([qry_df, bbox3d_local_df], axis=1)
    qry_df = qry_df.drop(columns=['x', 'y', 'z', 'dx', 'dy', 'dz'])

    # bbox, x0, y0, w, h, cx, cy
    bbox2d_df = scannet_loader.get_bbox2d(scene_id, frame_id).drop(columns=['c'])
    bbox2d_df = bbox2d_df.add_prefix('bbox_')
    assert len(qry_df) == len(bbox2d_df), f'{len(qry_df)} != {len(bbox2d_df)}'
    bbox2d_df.index = qry_df.index
    qry_df = pd.concat([qry_df, bbox2d_df], axis=1)

    # 3D from depth
    depth_img = scannet_loader.get_depth_img_in_m(scene_id, frame_id, inpaint=True)
    cx = bbox2d_df['bbox_cx'].values
    cy = bbox2d_df['bbox_cy'].values
    z_from_depth = depth_img[cy.astype(int), cx.astype(int)]
    w_residual = cx - depth_img.shape[1] / 2
    h_residual = cy - depth_img.shape[0] / 2
    x_from_depth = z_from_depth * w_residual / intrinsics[0, 0]
    y_from_depth = z_from_depth * h_residual / intrinsics[1, 1]
    bbox3d_from_depth_df = pd.DataFrame(
        np.vstack((x_from_depth, y_from_depth, z_from_depth)).T,
        columns=['tx', 'ty', 'tz'], index=qry_df.index)
    bbox3d_from_depth_df = bbox3d_from_depth_df.add_prefix('bbox3d_from_depth_')
    assert len(qry_df) == len(bbox3d_from_depth_df), f'{len(qry_df)} != {len(bbox3d_from_depth_df)}'
    qry_df = pd.concat([qry_df, bbox3d_from_depth_df], axis=1)

    masks = [(label_id in scannet_loader.label_table.index) for label_id in qry_df['label_id'].values]
    qry_df = qry_df[masks]
    if len(qry_df) == 0:
        return None
    label_ids = qry_df['label_id'].values
    label_names = scannet_loader.get_label_names(label_ids, 'nyu40class')
    qry_df['label_name'] = label_names
    # print(qry_df)

    # node ids
    node_ids = np.arange(len(qry_df))
    inst_ids = qry_df['inst_id'].values
    inst2node = dict(zip(inst_ids, node_ids))

    # text features
    onehot_label_embs = encode_text_features(
        label_names, label_ids, 'onehot', n_labels=40)
    glove_label_embs = encode_text_features(label_names, label_ids, 'glove')
    blip_label_embs = encode_text_features(label_names, label_ids, 'blipv2')

    features = {
        **{k: np.asarray(v).reshape(-1, 1) for k, v in qry_df.items()},
        'blip_label_embs': blip_label_embs,
        'glove_label_embs': glove_label_embs,
        'onehot_label_embs': onehot_label_embs,
    }

    data = {
        'step': frame_id,
        'features': features,
        'total_obj_cnt': len(qry_df),
        'inst_ids': inst_ids,
        'node_ids': node_ids,
        'inst2node': inst2node,
        'camera_pose': cam_pose,
        'camera_intrinsics': intrinsics,
        'img_size': depth_img.shape,
    }
    return data

# %%


def gen_map_graph(scene_id, edge3d_k):
    data = scannet_to_node_map_graph(scene_id)
    if data is None:
        return None
    node_ids = data['node_ids']
    # edge 3d
    xyz_cols = ['bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz']
    bbox3d_t_df = pd.DataFrame({k: data['features'][k][:, 0] for k in xyz_cols})
    edge_index_3d, edge_attr_3d = comp_bbox3d_edge_with_min_knn(bbox3d_t_df, xyz_cols, node_ids, k=edge3d_k)

    data.update({
        'edge': {
            '3d': {
                'index': edge_index_3d,
                'attr': edge_attr_3d,
            },
        }
    })
    return MapGraph(data)

# %%


def gen_query_graph(scene_id, frame_id, edge2d_k, edge3d_k, min_bbox_size=0):
    data = scannet_to_node_qry_graph(scene_id, frame_id)
    if data is None:
        return None
    node_ids = data['node_ids']
    # edge 3d
    # xyz_cols = ['bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz']
    # bbox3d_t_df = pd.DataFrame({k: data['features'][k][:, 0] for k in xyz_cols})
    # edge_index_3d, edge_attr_3d = comp_bbox3d_edge_with_min_knn(bbox3d_t_df, xyz_cols, node_ids, k=edge3d_k)
    # edge 2d
    cxcy_cols = ['bbox_cx', 'bbox_cy']
    bbox2d_cxcy_df = pd.DataFrame({k: data['features'][k][:, 0] for k in cxcy_cols})
    edge_index_2d, edge_attr_2d = comp_bbox2d_edge_with_min_knn(bbox2d_cxcy_df, cxcy_cols, node_ids, k=edge2d_k)
    # edge 3d from depth
    xyz_cols = ['bbox3d_from_depth_tx', 'bbox3d_from_depth_ty', 'bbox3d_from_depth_tz']
    bbox3d_from_depth_t_df = pd.DataFrame({k: data['features'][k][:, 0] for k in xyz_cols})
    edge_index_3d_from_depth, edge_attr_3d_from_depth = comp_bbox3d_edge_with_min_knn(
        bbox3d_from_depth_t_df, xyz_cols, node_ids, k=edge3d_k)

    data.update({
        'edge': {
            '2d': {
                'index': edge_index_2d,
                'attr': edge_attr_2d,
            },
            # '3d': {
            #     'index': edge_index_3d,
            #     'attr': edge_attr_3d,
            # },
            '3d_from_depth': {
                'index': edge_index_3d_from_depth,
                'attr': edge_attr_3d_from_depth,
            },
        }
    })
    return QueryGraph(data)

# %%


def get_frame_ids(scene_id: str):
    frame_paths = glob.glob(os.path.join(args.scannet_path, 'posed_images', scene_id, '*.jpg'))
    frame_ids = list(set([int(os.path.basename(frame_path).split('.')[0]) for frame_path in frame_paths]))
    return frame_ids

# %%


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_path', type=str, default='data/ScanNet/scannet_by_tr3d_with_DepthImage')
    parser.add_argument('--graph_dir_path', type=str, default='data/scannet_graph')
    parser.add_argument('--glove_path', type=str, default='glove/glove.6B.100d.txt')
    parser.add_argument('--qry_edge2d_k', type=int, default=-1, help='KNN edge for 2d query graph')
    parser.add_argument('--qry_edge3d_k', type=int, default=-1, help='KNN edge for 3d query graph')
    parser.add_argument('--map_edge3d_k', type=int, default=3, help='KNN edge for 3d map graph')
    parser.add_argument('--n_scenes', type=int, help='number of scenes to process, (max 706)')
    args = parser.parse_args()

    GRAPH_PATH = os.path.join(args.graph_dir_path, f'{args.n_scenes}_scenes')
    if os.path.exists(GRAPH_PATH):
        shutil.rmtree(GRAPH_PATH)
    os.makedirs(GRAPH_PATH)

    scannet_loader = ScanNetLoader(args.scannet_path)
    lavis_feature_extractor = LAVISFeatureExtractor()
    glove_feature_extractor = GloveFeatureExtractor(args.glove_path)

    meta_path = os.path.join(args.scannet_path, 'meta_data')
    scene_paths = [line.rstrip() for line in open(os.path.join(meta_path, 'scannetv2_train.txt'))]
    scene_ids = [scene_id for scene_id in scene_paths if scene_id.endswith('_00')]
    selected_scene_ids = scene_ids[:args.n_scenes]
    print('total scenes:', len(scene_ids))
    print('used  scenes:', len(selected_scene_ids))

    paired_df = pd.DataFrame(columns=['qry_fname', 'map_fname', 'n_overlap'])

    for scene_id in tqdm(selected_scene_ids):
        map_fname = f'{scene_id}.pkl'
        map_fpath = os.path.join(GRAPH_PATH, map_fname)
        map_graph = gen_map_graph(scene_id, edge3d_k=args.map_edge3d_k)
        if map_graph is None:
            # print(scene_id, 'is None')
            continue
        pkl.dump(map_graph, open(map_fpath, 'wb'))

        qry_frame_ids = get_frame_ids(scene_id)
        for frame_id in tqdm(qry_frame_ids):
            qry_fname = f'{scene_id}_{frame_id:0>5}.pkl'
            qry_fpath = os.path.join(GRAPH_PATH, qry_fname)
            qry_graph = gen_query_graph(scene_id, frame_id, edge2d_k=args.qry_edge2d_k, edge3d_k=args.qry_edge3d_k)
            if qry_graph is None:
                # print(scene_id, frame_id, 'is None')
                continue
            pkl.dump(qry_graph, open(qry_fpath, 'wb'))

            e1i, *_ = comp_graph_overlap(map_graph, qry_graph)
            n_overlap = len(e1i)
            paired_df = pd.concat([paired_df, pd.DataFrame(
                [[qry_fname, map_fname, n_overlap]], columns=paired_df.columns)])
    paired_df.to_csv(os.path.join(GRAPH_PATH, 'qm_paired_list.csv'), index=False)
