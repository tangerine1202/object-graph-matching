import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import copy
import glob
from pprint import pprint
import pickle as pkl
import shutil

from tqdm.auto import tqdm

from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

from lavis.models import load_model_and_preprocess

from solo_tool import Solo


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


def world2local_bbox3d(cam_pose, world_t, world_q, world_s):
    assert cam_pose.shape == (7,), f'cam_pose shape should be (7, ), but got {cam_pose.shape}'
    assert world_t.shape[1] == 3, f'world_t shape should be (n, 3), but got {world_t.shape}'
    assert world_q.shape[1] == 4, f'world_q shape should be (n, 4), but got {world_q.shape}'
    assert world_s.shape[1] == 3, f'world_s shape should be (n, 3), but got {world_s.shape}'
    # convert bbox3d from global to local
    cam_t = cam_pose[:3].reshape(1, 3)  # (1, 3)
    cam_R = R.from_quat(cam_pose[3:])
    local_t = cam_R.inv().apply(world_t - cam_t)  # (n, 3)
    local_r = cam_R.inv() * R.from_quat(world_q)
    local_s = np.array(world_s)
    bbox3d_df = pd.DataFrame(np.concatenate((local_t, local_r.as_quat(), local_s), axis=1),
                             columns=['bbox3d_tx', 'bbox3d_ty', 'bbox3d_tz',
                                      'bbox3d_qx', 'bbox3d_qy', 'bbox3d_qz', 'bbox3d_qw',
                                      'bbox3d_sx', 'bbox3d_sy', 'bbox3d_sz'])
    return bbox3d_df


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
        local_bbox3d_df = world2local_bbox3d(
            cap.camera_pose, world_bbox3d_t_df, world_bbox3d_q_df, world_bbox3d_s_df)
        local_bbox3d_df.index = query_df.index
        query_df = pd.concat((query_df, local_bbox3d_df), axis=1)
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
            'camera_intrinsics': cap.projectionMatrix,
        }

        return True, data_dict


class PairedGraph:
    def __init__(self, qry_g, map_g):
        self.g1 = qry_g
        self.g2 = map_g
        self.n1 = len(qry_g)
        self.n2 = len(map_g)
        self.n = self.n1 + self.n2

        # concat node features
        self.node_feat = {}
        for feat_name in self.g1.node_feat.keys():
            if feat_name not in self.g2.node_feat:
                # print(f'Warning: {feat_name} not in g2')
                continue
            self.node_feat[feat_name] = np.concatenate(
                [self.g1.node_feat[feat_name], self.g2.node_feat[feat_name]], axis=0)

        # edge index
        self.edge_index = {}
        for edge_type in self.g1.edge_index.keys():
            if edge_type not in self.g2.edge_index:
                # print(f'Warning: {edge_type} not in g2')
                continue
            self.edge_index[edge_type] = np.concatenate(
                [self.g1.edge_index[edge_type], self.g2.edge_index[edge_type] + self.n1], axis=1)
        # edge index -- cross graph
        # self.comp_cross_edge()
        # self.adj = np.zeros((self.n, self.n))
        # self.adj[:self.n1, :self.n1] = self.g1.adj
        # self.adj[self.n1:, self.n1:] = self.g2.adj
        # self.adj[self.n1:, :self.n1] = self.adj_g2tog1
        # self.adj[:self.n1, self.n1:] = self.adj_g1tog2
        # self.edge_index = self.adj2edge_index(self.adj)

        # edge attr
        self.edge_attr = {}
        for edge_type in self.g1.edge_attr.keys():
            if edge_type not in self.g2.edge_attr:
                # print(f'Warning: {edge_type} not in g2')
                continue
            self.edge_attr[edge_type] = np.concatenate(
                [self.g1.edge_attr[edge_type], self.g2.edge_attr[edge_type]], axis=0)

        for edge_type in self.edge_attr.keys():
            assert self.edge_attr[edge_type].shape[0] == self.edge_index[edge_type].shape[1], \
                f'edge_type {edge_type} shape of edge_attr {self.edge_attr[edge_type].shape} ' \
                f'does not match edge_index shape {self.edge_index[edge_type].shape}'

        # ground truth matching
        self.comp_GT_matching()

    # def comp_cross_edge(self):
    #     # compute edge between g1 and g2
    #     self.edge_g1tog2 = np.array([[g1_node_id, g2_node_id]
    #                                 for g1_node_id in self.g1.node_ids for g2_node_id in self.g2.node_ids]).T
    #     self.edge_g2tog1 = np.array([[g2_node_id, g1_node_id]
    #                                 for g2_node_id in self.g2.node_ids for g1_node_id in self.g1.node_ids]).T
    #     adj_g1tog2 = np.zeros((len(self.g1), len(self.g2)))
    #     adj_g2tog1 = np.zeros((len(self.g2), len(self.g1)))
    #     adj_g1tog2[self.edge_g1tog2[0], self.edge_g1tog2[1]] = 1
    #     adj_g2tog1[self.edge_g2tog1[0], self.edge_g2tog1[1]] = 1
    #     # self.adj_g1tog2 = adj_g1tog2
    #     # self.adj_g2tog1 = adj_g2tog1

    # def adj2edge_index(self, adj):
    #     edge_index = []
    #     for i in range(adj.shape[0]):
    #         for j in range(adj.shape[1]):
    #             if adj[i, j] == 1:
    #                 edge_index.append([i, j])
    #     edge_index = np.array(edge_index).T
    #     return edge_index

    def comp_GT_matching(self):
        self.all_inst_ids = list(set(self.g1.inst_ids) | set(self.g2.inst_ids))
        self.anchor_inst_ids = list(set(self.g1.inst_ids) & set(self.g2.inst_ids))
        self.e1i = [self.g1.inst2node[inst_id] for inst_id in self.anchor_inst_ids]
        self.e1j = [self.g1.inst2node[inst_id] for inst_id in self.g1.inst_ids if inst_id not in self.anchor_inst_ids]
        self.e2i = [self.g2.inst2node[inst_id] for inst_id in self.anchor_inst_ids]
        self.e2j = [self.g2.inst2node[inst_id] for inst_id in self.g2.inst_ids if inst_id not in self.anchor_inst_ids]

    def __len__(self):
        return self.n


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

        solo = Solo(args.path, is_reorganized=args.rearrange, move=args.move)

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
                paired_graph = PairedGraph(qry_graph, map_graph)
                n_overlap = len(paired_graph.e1i)
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
            paired_graph = PairedGraph(qry_graph, map_graph)
            n_overlap = len(paired_graph.e1i)
            paired_df = pd.concat((paired_df, pd.DataFrame({'qry_fname': qry_fname,
                                                            'map_fname': map_fname,
                                                            'n_overlap': n_overlap}, index=[0])), ignore_index=True)
        paired_df.to_csv(os.path.join(args.path, PAIR_FILE_NAME), index=False)
