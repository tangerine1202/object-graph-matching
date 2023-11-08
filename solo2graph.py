import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
import glob
import copy
from pprint import pprint
import pickle as pkl
import shutil

from tqdm.auto import tqdm

import cv2
from scipy.spatial.transform import Rotation as scipy_R
import numpy as np
import pandas as pd


from solo_tool import Solo
from preprocess import MapGraph, QueryGraph
from utils import (
    comp_graph_overlap,
    transform_bbox3d,
    invert_Rt
)
from preprocess.utils import (
    LAVISFeatureExtractor,
    comp_bbox3d_edge_with_min_knn,
    comp_bbox2d_edge_with_min_knn,
)


def read_depth(step, data_dir):
    path = f'{data_dir}/depth/step{step}.exr'
    depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = depth[:, :, 2]
    return depth


def gen_map_graph(f, solo, edge3d_k=3):
    data = frame_to_map3d_node(f, solo)
    node_ids = data['node_ids']

    xyz_cols = ['bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz']
    bbox3d_t_df = pd.DataFrame({k: data['features'][k][:, 0] for k in xyz_cols})
    edge_index_3d, edge_attr_3d = comp_bbox3d_edge_with_min_knn(bbox3d_t_df, xyz_cols, node_ids, k=edge3d_k)

    data.update({
        'edge': {
            '3d': {
                'index': edge_index_3d,
                'attr': edge_attr_3d,
            }
        }
    })

    return MapGraph(data)


def frame_to_map3d_node(f, solo):
    data_path = solo.path
    step = f.step
    cap = f.captures[0]
    metrics = f.metrics
    anno_defs = solo.annotation_definitions
    annos = cap.annotations

    inst = annos['instance segmentation']
    inst_df = inst.instances_df

    # FIXME: skip frames with no instance
    # if not inst.has_instance:
    #     return False, None

    meta = metrics['metadata']
    env_meta = meta.env_metadata
    meta_df = meta.instances_df

    # === map ===
    map_df = meta_df.copy()
    # rename label columns
    map_df = map_df.rename(columns={'object_labelName': 'label_name'})
    map_df['label_id'] = map_df['label_name'].apply(lambda x: anno_defs['instance segmentation'].name2id[x])

    # bbox3d
    world_bbox3d_t_df = map_df['object_translation'].apply(pd.Series).rename(
        columns={0: 'bbox3d_tx', 1: 'bbox3d_ty', 2: 'bbox3d_tz'})
    world_bbox3d_q_df = map_df['object_rotation'].apply(pd.Series).rename(
        columns={0: 'bbox3d_qx', 1: 'bbox3d_qy', 2: 'bbox3d_qz', 3: 'bbox3d_qw'})
    world_bbox3d_s_df = map_df['object_size'].apply(pd.Series).rename(
        columns={0: 'bbox3d_sx', 1: 'bbox3d_sy', 2: 'bbox3d_sz'})
    map_df = pd.concat((map_df, world_bbox3d_t_df, world_bbox3d_q_df, world_bbox3d_s_df), axis=1)
    map_df = map_df.drop(columns=['object_translation', 'object_rotation', 'object_size'])

    # rename instance id after merging label
    map_df = map_df.rename(columns={'instanceId': 'inst_id'})

    # semantic label
    embs = {}
    for _, obj in map_df.iterrows():
        label_name = obj['label_name']
        text_features = lavis_feature_extractor.extract_text_features(label_name)
        for k, v in text_features.items():
            if k not in embs:
                embs[k] = []
            embs[k].append(v)
    embs = {k: np.vstack(v) for k, v in embs.items()}

    features = {
        **{k: np.asarray(v).reshape(-1, 1) for k, v in map_df.items()},
        **{f'{k}_embs': v for k, v in embs.items()}
    }

    node_ids = map_df.index.to_list()
    inst_ids = map_df['inst_id'].to_list()
    inst2node = {inst_id: node_id for node_id, inst_id in enumerate(inst_ids)}

    data_dict = {
        'node_ids': node_ids,
        'inst_ids': inst_ids,
        'inst2node': inst2node,
        'features': features,
        'total_obj_cnt': len(map_df),
    }
    return data_dict


def get_camera_intrinsics(cap, env_meta):
    # camera intrinsics
    img_dim = cap.dimension
    horizontalFOV = env_meta['camera']['horizontalFOV']
    verticalFOV = env_meta['camera']['verticalFOV']
    normalized_x_scale = np.tan(np.deg2rad(horizontalFOV / 2))
    normalized_y_scale = np.tan(np.deg2rad(verticalFOV / 2))
    fx = img_dim[0] / (2 * normalized_x_scale)
    fy = img_dim[1] / (2 * normalized_y_scale)
    cx = img_dim[0] / 2
    cy = img_dim[1] / 2
    intrinsics = np.array([
        [fx, 0, cx],
        [0, -fy, cy],
        [0, 0, 1],
    ])
    return intrinsics


def gen_query_graph(f, solo, edge2d_k, edge3d_k, min_bbox_size=0):
    valid, data = frame_to_query_node(f, solo, min_bbox_size=min_bbox_size)
    if not valid:
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
    edge_index_3d, edge_attr_3d = comp_bbox3d_edge_with_min_knn(bbox3d_from_depth_t_df, xyz_cols, node_ids, k=edge3d_k)

    data.update({
        'edge': {
            '2d': {
                'index': edge_index_2d,
                'attr': edge_attr_2d,
            },
            '3d': {
                'index': edge_index_3d,
                'attr': edge_attr_3d,
            },
        }
    })
    return QueryGraph(data)


def frame_to_query_node(f, solo, min_bbox_size=0):
    data_path = solo.path
    step = f.step
    cap = f.captures[0]
    metrics = f.metrics
    anno_defs = solo.annotation_definitions
    annos = cap.annotations

    inst = annos['instance segmentation']
    inst_df = inst.instances_df

    bbox = annos['bounding box']
    bbox_df = bbox.values_df

    # FIXME: there is small difference between this bbox3d and the one in metadata.
    #        we convert the one in metadata to the camera view for now.
    # bbox3d = annos['bounding box 3D']
    # bbox3d_df = bbox3d.values_df

    meta = metrics['metadata']
    env_meta = meta.env_metadata
    meta_df = meta.instances_df

    camera_intrinsics = get_camera_intrinsics(cap, env_meta)
    # hFOV = env_meta['camera']['horizontalFOV']
    # vFOV = env_meta['camera']['verticalFOV']

    # FIXME: skip frames with no instance
    if not inst.has_instance:
        return False, None

    R_wc = scipy_R.from_quat(cap.camera_pose[3:])
    t_wc = cap.camera_pose[:3]
    R_cw, t_cw = invert_Rt(R_wc, t_wc)
    # === query ===
    query_df = meta_df.copy()

    # filter out invisible objects
    query_df = query_df[query_df['instanceId'].isin(inst_df['instanceId'])]
    # rename label columns
    query_df = query_df.rename(columns={'object_labelName': 'label_name'})
    query_df['label_id'] = query_df['label_name'].apply(lambda x: anno_defs['instance segmentation'].name2id[x])

    # NOTE: local bbox3d from world bbox3d
    world_bbox3d_t_df = query_df['object_translation'].apply(pd.Series).rename(
        columns={0: 'w_bbox3d_tx', 1: 'w_bbox3d_ty', 2: 'w_bbox3d_tz'})
    world_bbox3d_q_df = query_df['object_rotation'].apply(pd.Series).rename(
        columns={0: 'w_bbox3d_qx', 1: 'w_bbox3d_qy', 2: 'w_bbox3d_qz', 3: 'w_bbox3d_qw'})
    world_bbox3d_s_df = query_df['object_size'].apply(pd.Series).rename(
        columns={0: 'w_bbox3d_sx', 1: 'w_bbox3d_sy', 2: 'w_bbox3d_sz'})
    bbox3d_world_df = pd.concat((world_bbox3d_t_df, world_bbox3d_q_df, world_bbox3d_s_df), axis=1)
    bbox3d_local_df = pd.DataFrame(transform_bbox3d(bbox3d_world_df.values, R_cw, t_cw), columns=[
        'bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz',
        'bbox3d_qx', 'bbox3d_qy', 'bbox3d_qz', 'bbox3d_qw',
        'bbox3d_sx', 'bbox3d_sy', 'bbox3d_sz'], index=query_df.index)
    query_df = pd.concat((query_df, bbox3d_local_df), axis=1)
    query_df = query_df.drop(columns=['object_translation', 'object_rotation', 'object_size'])

    # NOTE: local bbox3d from perception
    # world_bbox3d_s_df = query_df['object_size'].apply(pd.Series).rename(
    #     columns={0: 'w_bbox3d_sx', 1: 'w_bbox3d_sy', 2: 'w_bbox3d_sz'})
    # local_bbox3d_s_df = pd.DataFrame(R_cw.apply(world_bbox3d_s_df.values), columns=[
    #     'bbox3d_sx', 'bbox3d_sy', 'bbox3d_sz'], index=query_df.index)
    # bbox3d_df.update(local_bbox3d_s_df)

    # depth of bbox center
    depth = read_depth(step, data_path)
    bbox_depth = np.asarray([depth[cy, cx] for cx, cy in zip(
        bbox_df['cx'].values.astype(int), bbox_df['cy'].values.astype(int))])
    z_from_2d = bbox_depth
    w_residual = bbox_df['cx'] - cap.dimension[0] / 2
    h_residual = bbox_df['cy'] - cap.dimension[1] / 2
    x_from_2d = z_from_2d * w_residual / camera_intrinsics[0, 0]
    y_from_2d = z_from_2d * h_residual / camera_intrinsics[1, 1]
    xyz_from_2d_cols = ['tx', 'ty', 'tz']
    bbox3d_from_2d_df = pd.DataFrame(np.vstack((x_from_2d, y_from_2d, z_from_2d)).T, columns=xyz_from_2d_cols)
    bbox3d_from_2d_df['instanceId'] = bbox_df['instanceId']
    bbox3d_from_2d_df = bbox3d_from_2d_df.set_index('instanceId', drop=True)

    # merge annotations
    inst_df_for_merge = inst_df.copy() \
        .drop(columns=['labelName', 'labelId', 'color']) \
        .add_prefix('inst_')
    inst_df_for_merge['instanceId'] = inst_df_for_merge['inst_instanceId']
    bbox_df_for_merge = bbox_df.copy() \
        .drop(columns=['labelName', 'labelId']) \
        .add_prefix('bbox_')
    bbox_df_for_merge['instanceId'] = bbox_df_for_merge['bbox_instanceId']
    # NOTE: compute bbox3d from world bbox3d
    # bbox3d_df_for_merge = bbox3d_df.copy() \
    #     .drop(columns=['labelName', 'labelId']) \
    #     .add_prefix('bbox3d_') \
    #     .rename(columns={'bbox3d_instanceId': 'instanceId'})
    # NOTE: bbox3d from 2d
    bbox3d_from_2d_df_for_merge = bbox3d_from_2d_df.copy() \
        .add_prefix('bbox3d_from_depth_') \
        .rename(columns={'bbox3d_from_depth_instanceId': 'instanceId'})
    query_df = pd.merge(query_df, inst_df_for_merge, how='inner', left_on='instanceId',
                        right_on='instanceId', suffixes=('', '_duplicated'))
    query_df = pd.merge(query_df, bbox_df_for_merge, how='inner', left_on='instanceId',
                        right_on='instanceId', suffixes=('', '_duplicated'))
    query_df = pd.merge(query_df, bbox3d_from_2d_df_for_merge, how='inner', left_on='instanceId',
                        right_on='instanceId', suffixes=('', '_duplicated'))
    # NOTE: compute bbox3d from world bbox3d
    # query_df = pd.merge(query_df, bbox3d_df_for_merge, how='inner', left_on='instanceId',
    #                     right_on='instanceId', suffixes=('', '_duplicated'))
    query_df = query_df.rename(columns={'instanceId': 'inst_id'})

    # filter out small object
    mask = (query_df['bbox_w'] * query_df['bbox_h']) > min_bbox_size
    query_df = query_df[mask].reset_index(drop=True)
    if len(query_df) <= 1:
        return False, None

    # extract semantic label features
    embs = {}
    for _, obj in query_df.iterrows():
        label_name = obj['label_name']
        text_features = lavis_feature_extractor.extract_text_features(label_name)
        for k, v in text_features.items():
            if k not in embs:
                embs[k] = []
            embs[k].append(v)
    embs = {k: np.vstack(v) for k, v in embs.items()}

    features = {
        **{k: np.array(v).reshape(-1, 1) for k, v in query_df.items()},
        **{f'{k}_embs': v for k, v in embs.items()},
    }

    node_ids = query_df.index.to_list()
    inst_ids = query_df['inst_id'].to_list()
    inst2node = {inst_id: node_id for node_id, inst_id in enumerate(inst_ids)}

    data_dict = {
        'node_ids': node_ids,
        'inst_ids': inst_ids,
        'inst2node': inst2node,
        'features': features,
        'total_obj_cnt': len(query_df),
        'step': f.step,
        'camera_pose': cap.camera_pose,
        'camera_intrinsics': get_camera_intrinsics(cap, env_meta),
        'camera_hFOV': env_meta['camera']['horizontalFOV'],
        'camera_vFOV': env_meta['camera']['verticalFOV'],
        'img_size': cap.dimension,
    }

    return True, data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        type=str,
                        help='path to solo output',
                        required=True)
    # rearrange solo
    parser.add_argument('--rearrange',
                        action='store_true',
                        help='whether to rearrange the output')
    parser.add_argument('--move',
                        action='store_true',
                        help='move files instead of copying')
    # graph
    parser.add_argument('--graph-dir',
                        type=str,
                        default='graph',
                        help='name of the single graph directory')
    parser.add_argument('--skip-graph-gen',
                        action='store_true',
                        help='skip single graph generation')
    parser.add_argument('--min-bbox-size',
                        type=int,
                        default=0,
                        help='minimum bbox size to be included in the graph')
    parser.add_argument('--qry-edge2d-k',
                        type=int,
                        default=-1,
                        help='KNN edge for 2d query graph')
    parser.add_argument('--qry-edge3d-k',
                        type=int,
                        default=-1,
                        help='KNN edge for 3d query graph')
    parser.add_argument('--map-edge3d-k',
                        type=int,
                        default=3,
                        help='KNN edge for 3d map graph')

    # map graph
    parser.add_argument('--q2q',
                        action='store_true',
                        help='whether to use query to query graph')
    parser.add_argument('--q2m',
                        action='store_true',
                        help='whether to use query to query graph')
    parser.add_argument('--map-filename',
                        type=str,
                        default='map',
                        help='name of the map graph file')

    # paired graph
    parser.add_argument('--pair-file-name',
                        type=str,
                        default='paired_list.csv',
                        help='name of the list of paired graph')
    args = parser.parse_args()

    GRAPH_PATH = os.path.join(args.path, args.graph_dir)
    lavis_feature_extractor = LAVISFeatureExtractor()

    if not args.skip_graph_gen:
        if os.path.exists(GRAPH_PATH):
            print(f'remove {GRAPH_PATH}')
            shutil.rmtree(GRAPH_PATH)
        os.mkdir(GRAPH_PATH)

        solo = Solo(args.path, is_reorganized=(not args.rearrange), move=args.move)

        is_first = True
        for f in tqdm(solo.frames()):
            # map graph
            if is_first:
                is_first = False
                map_graph = gen_map_graph(f, solo, edge3d_k=args.map_edge3d_k)
                map_graph_path = os.path.join(GRAPH_PATH, f'{args.map_filename}.pkl')
                pkl.dump(map_graph, open(map_graph_path, 'wb'))

            # query graph
            qry_graph = gen_query_graph(f, solo,
                                        edge2d_k=args.qry_edge2d_k,
                                        edge3d_k=args.qry_edge3d_k,
                                        min_bbox_size=args.min_bbox_size)
            if qry_graph is None:
                print(f'step {f.step} is invalid')
                continue
            qry_graph_path = os.path.join(GRAPH_PATH, f'step{f.step}.pkl')
            pkl.dump(qry_graph, open(qry_graph_path, 'wb'))

    # paired graph
    if args.q2q:
        PAIR_FILE_NAME = f'qq_{args.pair_file_name}'
        qry_fnames = [os.path.basename(name) for name in glob.glob(
            os.path.join(GRAPH_PATH, '*.pkl')) if args.map_filename not in name]
        paired_df = pd.DataFrame(columns=['qry_fname', 'map_fname', 'n_overlap'])
        for i in tqdm(range(len(qry_fnames))):
            for j in range(i + 1, len(qry_fnames)):
                qry_fname = qry_fnames[i]
                map_fname = qry_fnames[j]
                qry_graph = pkl.load(open(os.path.join(GRAPH_PATH, qry_fname), 'rb'))
                map_graph = pkl.load(open(os.path.join(GRAPH_PATH, map_fname), 'rb'))
                e1i, _, _, _ = comp_graph_overlap(qry_graph, map_graph)
                n_overlap = len(e1i)
                paired_df = pd.concat((paired_df, pd.DataFrame({'qry_fname': qry_fname,
                                                                'map_fname': map_fname,
                                                                'n_overlap': n_overlap}, index=[0])), ignore_index=True)
        paired_df.to_csv(os.path.join(args.path, PAIR_FILE_NAME), index=False)

    if args.q2m:
        PAIR_FILE_NAME = f'qm_{args.pair_file_name}'
        map_fname = f'{args.map_filename}.pkl'
        qry_fnames = [
            os.path.basename(name) for name in glob.glob(os.path.join(GRAPH_PATH, '*.pkl')) if map_fname not in name]

        map_graph = pkl.load(open(os.path.join(GRAPH_PATH, map_fname), 'rb'))
        paired_df = pd.DataFrame(columns=['qry_fname', 'map_fname', 'n_overlap'])
        for qry_fname in tqdm(qry_fnames):
            qry_graph = pkl.load(open(os.path.join(GRAPH_PATH, qry_fname), 'rb'))
            e1i, _, _, _ = comp_graph_overlap(qry_graph, map_graph)
            n_overlap = len(e1i)
            paired_df = pd.concat((paired_df, pd.DataFrame({'qry_fname': qry_fname,
                                                            'map_fname': map_fname,
                                                            'n_overlap': n_overlap}, index=[0])), ignore_index=True)
        paired_df.to_csv(os.path.join(args.path, PAIR_FILE_NAME), index=False)
