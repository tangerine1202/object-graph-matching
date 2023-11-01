import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import copy
import glob
from pprint import pprint
import pickle as pkl
import shutil

from tqdm.auto import tqdm

from scipy.spatial.transform import Rotation as scipy_R
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

from lavis.models import load_model_and_preprocess

from solo_tool import Solo
from utils import comp_graph_overlap, transform_bbox3d, corners_of_bbox3d, invert_Rt


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def extract_text_features(text):
    text_input = txt_processors["eval"](text)
    sample = {"text_input": [text_input]}
    features_text = model.extract_features(sample, mode="text")  # torch.Size([1, words, 768])
    feat_txt = features_text.text_embeds[:, 0, :].cpu().numpy()
    feat_norm_txt = features_text.text_embeds_proj[:, 0, :].cpu().numpy()
    return {
        'text': feat_txt,
        'norm_text': feat_norm_txt,
    }


def extract_multimodal_features(raw_image, text):
    """
    LAVIS Unified Feature Extraction Interface

    The multimodal feature can be used for multimodal classification.
    The low-dimensional unimodal features can be used to compute cross-modal similarity.

    ref: https://github.com/salesforce/LAVIS#unified-feature-extraction-interface
    """
    if isinstance(raw_image, np.ndarray):
        raw_image = Image.fromarray(raw_image)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](text)
    sample = {"image": image, "text_input": [text_input]}

    features_multimodal = model.extract_features(sample)  # torch.Size([1, 32, 768]), 32 is the number of queries
    features_image = model.extract_features(sample, mode="image")  # torch.Size([1, 32, 768])
    features_text = model.extract_features(sample, mode="text")  # torch.Size([1, words, 768])

    # multimodal feature
    feat_multimodal = features_multimodal.multimodal_embeds[:, 0, :].cpu().numpy()
    # uni-modal features
    feat_img = features_image.image_embeds[:, 0, :].cpu().numpy()
    feat_txt = features_text.text_embeds[:, 0, :].cpu().numpy()
    # normalized low-dimensional uni-modal features
    feat_norm_img = features_image.image_embeds_proj[:, 0, :].cpu().numpy()  # norm_img: torch.Size([1, 197, 256])
    feat_norm_txt = features_text.text_embeds_proj[:, 0, :].cpu().numpy()  # norm_tex: torch.Size([1, words, 256])

    return {
        'multimodal': feat_multimodal,
        'image': feat_img,
        'text': feat_txt,
        'norm_image': feat_norm_img,
        'norm_text': feat_norm_txt,
    }


def comp_bbox2d_dist_and_angle(bbox2d_df, xy_cols):
    df = bbox2d_df[xy_cols]
    cross_df = df.merge(df, how='cross', suffixes=('_src', '_dst'))
    # distance
    diff = cross_df[[f'{col}_dst' for col in xy_cols]].values - \
        cross_df[[f'{col}_src' for col in xy_cols]].values  # (n*n, 2)
    dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)  # (n*n, 1)
    # angle
    theta = np.arctan2(diff[:, 1], diff[:, 0]).reshape(-1, 1)  # (n*n, 1)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    edge_attr = pd.DataFrame(np.concatenate((dist, sin_theta, cos_theta), axis=1), columns=[
        'bbox2d_dist', 'bbox2d_sin', 'bbox2d_cos'])  # (n*n, 3)
    return edge_attr


def comp_bbox3d_dist_and_quat(bbox3d_df, xyz_cols):
    df = bbox3d_df[xyz_cols]
    cross_df = df.merge(df, how='cross', suffixes=('_src', '_dst'))
    # distance
    diff = cross_df[[f'{col}_dst' for col in xyz_cols]].values - \
        cross_df[[f'{col}_src' for col in xyz_cols]].values  # (n*n, 3)
    dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)  # (n*n, 1)
    # quaternion
    axis = np.cross(cross_df[[f'{col}_src' for col in xyz_cols]].values,
                    cross_df[[f'{col}_dst' for col in xyz_cols]].values)  # (n*n, 3)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)  # (n*n, 3)
    cos_theta = np.sum(cross_df[[f'{col}_src' for col in xyz_cols]].values *
                       cross_df[[f'{col}_dst' for col in xyz_cols]].values, axis=1)  # (n*n, )
    cos_theta /= np.linalg.norm(cross_df[[f'{col}_src' for col in xyz_cols]].values, axis=1) * \
        np.linalg.norm(cross_df[[f'{col}_dst' for col in xyz_cols]].values, axis=1)  # (n*n, )
    cos_theta = cos_theta.reshape(-1, 1)  # (n*n, 1)
    assert np.allclose(cos_theta, np.clip(cos_theta, -1, 1)
                       ), f'cos_theta out of [-1, 1]: ({cos_theta.min()}, {cos_theta.max()})'
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta)  # (n*n, 1)
    half_angle = 0.5 * angle
    sin_half_angle = np.sin(half_angle)
    xyz = sin_half_angle * axis
    w = np.cos(half_angle)
    edge_attr = pd.DataFrame(np.concatenate((dist, xyz, w), axis=1), columns=[
        'bbox3d_dist', 'bbox3d_qx', 'bbox3d_qy', 'bbox3d_qz', 'bbox3d_qw'])  # (n*n, 5)
    return edge_attr


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


class MapGraph:
    def __init__(self, f, solo):
        self.data = self.frame_to_data(f, solo)

        self.total_inst_cnt = self.data['total_obj_cnt']
        self.inst_ids = self.data['inst_ids']
        self.inst2node = self.data['inst2node']
        self.node_ids = self.data['node_ids']
        self.node_feat = self.data['features']

        edge_index_3d, edge_attr_3d = self.comp_bbox3d_edge_with_min_knn()
        self.edge_index = {
            '3d': edge_index_3d
        }
        self.edge_attr = {
            '3d': edge_attr_3d
        }

    def __len__(self):
        return len(self.node_ids)

    def comp_bbox3d_edge_with_min_knn(self, k=3):
        # edge attr
        cols = ['bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz']
        node_df = pd.DataFrame({k: self.node_feat[k][:, 0] for k in cols})
        edge_attr = comp_bbox3d_dist_and_quat(node_df, cols)

        # edge index from edge_attr
        edge_attr['src'] = np.arange(len(self.node_ids)).repeat(len(self.node_ids))
        edge_attr['dst'] = np.tile(np.arange(len(self.node_ids)), len(self.node_ids))
        # exclude diagonal edges
        mask = np.eye(len(self.node_ids), dtype=bool).reshape(-1)  # (n*n, 1)
        edge_attr = edge_attr[~mask]

        # KNN edge
        pivot = edge_attr.pivot(index='dst', columns='src', values='bbox3d_dist')
        knn_dist = pivot.apply(lambda x: x.nsmallest(k).index)
        edge_index = knn_dist.melt().rename(columns={'variable': 'src', 'value': 'dst'})
        # update edge_attr
        edge_attr = edge_attr.merge(edge_index, on=['src', 'dst'])
        edge_index = edge_attr[['src', 'dst']].T
        edge_attr = edge_attr.drop(columns=['src', 'dst'])
        return edge_index, edge_attr

    def frame_to_data(self, f, solo):
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

        # rename instance id after merging tabel
        map_df = map_df.rename(columns={'instanceId': 'inst_id'})

        # semantic label
        embs = {}
        for _, obj in map_df.iterrows():
            label_name = obj['label_name']
            text_features = extract_text_features(label_name)
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


class QueryGraph:
    def __init__(self, f, solo, min_bbox_size=0, drop_no_instance=True):
        self.min_bbox_size = min_bbox_size
        valid, data = self.frame_to_data(f, solo)
        self.valid = valid
        self.data = data
        if not valid and drop_no_instance:
            self.data = None
            return

        self.step = self.data['step']
        self.camera_pose = self.data['camera_pose']
        self.camera_intrinsics = self.data['camera_intrinsics']

        self.total_inst_cnt = self.data['total_obj_cnt']
        self.inst_ids = self.data['inst_ids']
        self.inst2node = self.data['inst2node']

        self.node_ids = self.data['node_ids']
        self.node_feat = self.data['features']

        edge_index_2d, edge_attr_2d = self.comp_bbox2d_edge_with_dense()
        edge_index_3d, edge_attr_3d = self.comp_bbox3d_edge_with_dense()
        self.edge_index = {
            '2d': edge_index_2d,
            '3d': edge_index_3d,
        }
        self.edge_attr = {
            '2d': edge_attr_2d,
            '3d': edge_attr_3d,
        }

    def __len__(self):
        return len(self.node_ids)

    def comp_bbox2d_edge_with_dense(self):
        cols = ['bbox_cx', 'bbox_cy']
        node_df = pd.DataFrame({k: self.node_feat[k][:, 0] for k in cols})
        edge_attr = comp_bbox2d_dist_and_angle(node_df, cols)
        # edge index from edge_attr
        edge_attr['src'] = np.arange(len(self.node_ids)).repeat(len(self.node_ids))
        edge_attr['dst'] = np.tile(np.arange(len(self.node_ids)), len(self.node_ids))
        # remove self loop
        mask = np.eye(len(self.node_ids), dtype=bool).reshape(-1)
        edge_attr = edge_attr[~mask]

        edge_index = edge_attr[['src', 'dst']].T
        edge_attr = edge_attr.drop(columns=['src', 'dst'])
        return edge_index, edge_attr

    def comp_bbox3d_edge_with_dense(self):
        cols = ['bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz']
        node_df = pd.DataFrame({k: self.node_feat[k][:, 0] for k in cols})
        edge_attr = comp_bbox3d_dist_and_quat(node_df, cols)
        # edge index from edge_attr
        edge_attr['src'] = np.arange(len(self.node_ids)).repeat(len(self.node_ids))
        edge_attr['dst'] = np.tile(np.arange(len(self.node_ids)), len(self.node_ids))
        # remove self loop
        mask = np.eye(len(self.node_ids), dtype=bool).reshape(-1)  # (n*n, 1)
        edge_attr = edge_attr[~mask]

        edge_index = edge_attr[['src', 'dst']].T
        edge_attr = edge_attr.drop(columns=['src', 'dst'])
        return edge_index, edge_attr

    def comp_bbox3d_edge_with_min_knn(self, k=3):
        # edge attr
        cols = ['bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz']
        node_df = pd.DataFrame({k: self.node_feat[k][:, 0] for k in cols})
        edge_attr = comp_bbox3d_dist_and_quat(node_df, cols)
        # edge index from edge_attr
        edge_attr['src'] = np.arange(len(self.node_ids)).repeat(len(self.node_ids))
        edge_attr['dst'] = np.tile(np.arange(len(self.node_ids)), len(self.node_ids))
        # remove diagonal edges
        mask = np.eye(len(self.node_ids), dtype=bool).reshape(-1)  # (n*n, 1)
        edge_attr = edge_attr[~mask]

        # KNN edge
        pivot = edge_attr.pivot(index='dst', columns='src', values='bbox3d_dist')
        knn_dist = pivot.apply(lambda x: x.nsmallest(k).index)
        edge_index = knn_dist.melt().rename(columns={'variable': 'src', 'value': 'dst'})

        edge_attr = edge_attr.merge(edge_index, on=['src', 'dst'])
        edge_index = edge_attr[['src', 'dst']].T
        edge_attr = edge_attr.drop(columns=['src', 'dst'])
        return edge_index, edge_attr

    def frame_to_data(self, f, solo):
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
        query_df = pd.merge(query_df, inst_df_for_merge, how='inner', left_on='instanceId',
                            right_on='instanceId', suffixes=('', '_duplicated'))
        query_df = pd.merge(query_df, bbox_df_for_merge, how='inner', left_on='instanceId',
                            right_on='instanceId', suffixes=('', '_duplicated'))
        # NOTE: compute bbox3d from world bbox3d
        # obj_df = pd.merge(obj_df, bbox3d_df_for_merge, how='inner', left_on='instanceId',
        #                   right_on='instanceId', suffixes=('', '_duplicated'))
        query_df = query_df.rename(columns={'instanceId': 'inst_id'})

        # filter out small object
        mask = (query_df['bbox_w'] * query_df['bbox_h']) > self.min_bbox_size
        query_df = query_df[mask].reset_index(drop=True)
        if len(query_df) <= 1:
            return False, None

        # extract semantic label features
        embs = {}
        for _, obj in query_df.iterrows():
            label_name = obj['label_name']
            text_features = extract_text_features(label_name)
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
    parser.add_argument('--keep-no-instance',
                        action='store_true',
                        help='skip frames with no instance')

    # map graph
    parser.add_argument('--no-q2q',
                        action='store_true',
                        help='whether to use query to query graph')
    parser.add_argument('--no-q2m',
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

    if not args.skip_graph_gen:
        if os.path.exists(GRAPH_PATH):
            print(f'remove {GRAPH_PATH}')
            shutil.rmtree(GRAPH_PATH)
        os.mkdir(GRAPH_PATH)

        solo = Solo(args.path, is_reorganized=(not args.rearrange), move=args.move)

        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

        is_first = True
        for f in tqdm(solo.frames()):
            # map graph
            if is_first:
                is_first = False
                map_graph = MapGraph(f, solo)
                map_graph_path = os.path.join(GRAPH_PATH, f'{args.map_filename}.pkl')
                pkl.dump(map_graph, open(map_graph_path, 'wb'))

            # query graph
            qry_graph = QueryGraph(f, solo, min_bbox_size=args.min_bbox_size,
                                   drop_no_instance=(not args.keep_no_instance))
            if not qry_graph.valid and not args.keep_no_instance:
                print(f'step {f.step} is invalid')
                continue
            qry_graph_path = os.path.join(GRAPH_PATH, f'step{f.step}.pkl')
            pkl.dump(qry_graph, open(qry_graph_path, 'wb'))

    # paired graph
    if not args.no_q2q:
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

    if not args.no_q2m:
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
