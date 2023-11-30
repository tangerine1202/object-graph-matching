import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from ultralytics import RTDETR
import networkx as nx

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.set_printoptions(suppress=True)

from preprocess import MapGraph, QueryGraph
from preprocess.scannet_utils import ScanNetLoader, get_fov, create_color_map, create_camera_frame, create_cuboid

from preprocess.utils import (
    GloveFeatureExtractor,
    LAVISFeatureExtractor,
    comp_bbox3d_edge_with_min_knn,
    comp_bbox2d_edge_with_min_knn,
)

from utils import (
    comp_graph_overlap,
    transform_bbox3d,
    invert_Rt,
    compute_IoU_x0y0wh,
    compute_IoU_xyxy
)

from eva import (
    compute_pose_error,
)


def encode_text_features(label_names, label_ids, type, n_labels=None):
    if type == 'onehot':
        assert n_labels is not None
        feats = np.zeros((len(label_ids), n_labels))
        feats[np.arange(len(label_ids)), label_ids - 1] = 1
    elif type == 'blipv2':
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
    return feats


def detection_to_bbox2d_df(rgb_path, orig_img_size, img_size):
    sx = img_size[0] / orig_img_size[0]
    sy = img_size[1] / orig_img_size[1]
    result = rtdetr_model(rgb_path)[0]
    bbox2d_from_detect_df = pd.DataFrame(result.boxes.xyxy.cpu().numpy(), columns=['x0', 'y0', 'x1', 'y1'])
    bbox2d_from_detect_df['x0'] = bbox2d_from_detect_df['x0'].values * sx
    bbox2d_from_detect_df['y0'] = bbox2d_from_detect_df['y0'].values * sy
    bbox2d_from_detect_df['x1'] = bbox2d_from_detect_df['x1'].values * sx
    bbox2d_from_detect_df['y1'] = bbox2d_from_detect_df['y1'].values * sy
    bbox2d_from_detect_df['w'] = bbox2d_from_detect_df['x1'] - bbox2d_from_detect_df['x0']
    bbox2d_from_detect_df['h'] = bbox2d_from_detect_df['y1'] - bbox2d_from_detect_df['y0']
    bbox2d_from_detect_df['cx'] = (bbox2d_from_detect_df['x0'] + bbox2d_from_detect_df['w']) / 2
    bbox2d_from_detect_df['cy'] = (bbox2d_from_detect_df['y0'] + bbox2d_from_detect_df['h']) / 2
    bbox2d_from_detect_df['label_id'] = result.boxes.cls.cpu().numpy().astype(int)
    bbox2d_from_detect_df['label_name'] = bbox2d_from_detect_df['label_id'].map(str).map(coco_id2name)
    return bbox2d_from_detect_df


def scannet_to_node_qry_graph_from_real(scene_id, frame_id):
    # camera params
    _, align_matrix = scannet_loader.get_bbox3d_and_align_matrix(scene_id)
    t_wc, R_wc = scannet_loader.get_cam_pose(scene_id, frame_id, align_matrix)
    if np.any(np.isnan(t_wc)) or np.any(np.isnan(R_wc.as_quat())) or np.any(np.isinf(t_wc)) or np.any(np.isinf(R_wc.as_quat())):
        return None
    cam_pose = np.concatenate([t_wc, R_wc.as_quat()], axis=0)
    intrinsics = scannet_loader.get_intrinsics(scene_id)
    orig_img_size = scannet_loader.orig_img_size
    img_size = scannet_loader.img_size

    # bbox from detection
    rgb_path = scannet_loader.get_rgb_path(scene_id, frame_id)
    bbox2d_from_detect_df = detection_to_bbox2d_df(rgb_path, orig_img_size, img_size)

    if len(bbox2d_from_detect_df) == 0:
        return None

    qry_df = bbox2d_from_detect_df.add_prefix('bbox_from_detect_') \
        .rename(columns={'bbox_from_detect_label_name': 'label_name',
                         'bbox_from_detect_label_id': 'label_id'}).copy()
    label_names = qry_df['label_name']
    label_ids = qry_df['label_id']

    node_ids = np.arange(len(qry_df))
    inst_ids = np.arange(len(qry_df))
    inst2node = {inst_id: node_id for node_id, inst_id in enumerate(inst_ids)}

    # text features
    onehot_label_embs = encode_text_features(
        label_names, label_ids, 'onehot', n_labels=80)
    glove_label_embs = encode_text_features(label_names, label_ids, 'glove')
    blip_label_embs = encode_text_features(label_names, label_ids, 'blipv2')

    features = {
        **{k: np.asarray(v).reshape(-1, 1) for k, v in qry_df.items()},
        'onehot_label_embs': onehot_label_embs,
        'glove_label_embs': glove_label_embs,
        'blip_label_embs': blip_label_embs,
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
        'img_size': scannet_loader.img_size,
    }

    return data


def gen_query_graph_from_real(scene_id, frame_id, edge2d_k, edge3d_k):
    data = scannet_to_node_qry_graph_from_real(scene_id, frame_id)
    if data is None:
        return None
    node_ids = data['node_ids']

    cxcy_cols = ['bbox_from_detect_cx', 'bbox_from_detect_cy']
    bbox2d_cxcy_df = pd.DataFrame({k: data['features'][k][:, 0] for k in cxcy_cols})
    edge_index_2d, edge_attr_2d = comp_bbox2d_edge_with_min_knn(bbox2d_cxcy_df, cxcy_cols, node_ids, k=edge2d_k)
    # edge 3d from depth
    # xyz_cols = ['bbox3d_from_depth_tx', 'bbox3d_from_depth_ty', 'bbox3d_from_depth_tz']
    # bbox3d_from_depth_t_df = pd.DataFrame({k: data['features'][k][:, 0] for k in xyz_cols})
    # edge_index_3d_from_depth, edge_attr_3d_from_depth = comp_bbox3d_edge_with_min_knn(
    #     bbox3d_from_depth_t_df, xyz_cols, node_ids, k=edge3d_k)

    data.update({
        'edge': {
            '2d': {
                'index': edge_index_2d,
                'attr': edge_attr_2d,
            },
            # '3d_from_depth': {
            #     'index': edge_index_3d_from_depth,
            #     'attr': edge_attr_3d_from_depth,
            # },
        }
    })
    return QueryGraph(data)


def get_frame_ids(scene_id: str):
    frame_paths = glob.glob(os.path.join(args.scannet_path, 'posed_images', scene_id, '*.jpg'))
    frame_ids = list(set([int(os.path.basename(frame_path).split('.')[0]) for frame_path in frame_paths]))
    return frame_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_path', type=str, default='data/ScanNet/scannet_by_tr3d_with_DepthImage')
    parser.add_argument('--map_dir_path', type=str, default='data/scannet_graph/30_scenes')
    parser.add_argument('--paired_list_fname', type=str, default='qm_paired_list.csv')
    parser.add_argument('--graph_dir_path', type=str, default='data/scannet_graph_from_real')
    parser.add_argument('--glove_path', type=str, default='glove/glove.6B.100d.txt')
    parser.add_argument('--qry_edge2d_k', type=int, default=-1, help='KNN edge for 2d query graph')
    parser.add_argument('--qry_edge3d_k', type=int, default=-1, help='KNN edge for 3d query graph')
    parser.add_argument('--n_scenes', type=int, help='number of scenes to process, (max 706)')
    args = parser.parse_args()

    if os.path.exists(args.graph_dir_path):
        confirm = input(f'{args.graph_dir_path} already exists, do you want to remove it? (y/n)')
        if confirm == 'y':
            shutil.rmtree(args.graph_dir_path)
        else:
            exit()
    os.makedirs(args.graph_dir_path)

    scannet_loader = ScanNetLoader(args.scannet_path)
    lavis_feature_extractor = LAVISFeatureExtractor()
    glove_feature_extractor = GloveFeatureExtractor(args.glove_path)
    rtdetr_model = RTDETR('object-detection/rtdetr-l.pt')
    coco_id2name = json.load(open('object-detection/coco_id2name.json'))

    paired_df = pd.DataFrame(columns=['qry_fname', 'map_fname', 'n_overlap'])

    map_fnames = np.unique(pd.read_csv(os.path.join(args.map_dir_path, args.paired_list_fname))['map_fname'])
    map_fname = map_fnames[:args.n_scenes]
    for map_fname in tqdm(map_fnames):
        scene_id = ''.join(map_fname.split('.')[:-1])
        map_fpath = os.path.join(args.map_dir_path, map_fname)
        map_graph = pkl.load(open(map_fpath, 'rb'))

        qry_frame_ids = get_frame_ids(scene_id)
        for frame_id in tqdm(qry_frame_ids):
            qry_fname = f'{scene_id}_{frame_id:0>5}.pkl'
            qry_fpath = os.path.join(args.graph_dir_path, qry_fname)
            qry_graph = gen_query_graph_from_real(
                scene_id, frame_id, edge2d_k=args.qry_edge2d_k, edge3d_k=args.qry_edge3d_k)
            if qry_graph is None:
                continue
            pkl.dump(qry_graph, open(qry_fpath, 'wb'))

            # e1i, *_ = comp_graph_overlap(map_graph, qry_graph)
            # n_overlap = len(e1i)
            n_overlap = -1
            paired_df = pd.concat([paired_df, pd.DataFrame(
                [[qry_fname, map_fname, n_overlap]], columns=paired_df.columns)])
    paired_df.to_csv(os.path.join(args.graph_dir_path, 'qm_paired_list.csv'), index=False)
