import os
import numpy as np
import pandas as pd
import PIL.Image as Image
from lavis.models import load_model_and_preprocess
import torch

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


# LAVIS Unified Feature Extraction Interface


class LAVISFeatureExtractor:
    def __init__(self):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors

    def extract_text_features(self, text):
        text_input = self.txt_processors["eval"](text)
        sample = {"text_input": [text_input]}
        features_text = self.model.extract_features(sample, mode="text")  # torch.Size([1, words, 768])
        feat_txt = features_text.text_embeds[:, 0, :].cpu().numpy()
        feat_norm_txt = features_text.text_embeds_proj[:, 0, :].cpu().numpy()
        return {
            'text': feat_txt,
            'norm_text': feat_norm_txt,
        }

    def extract_multimodal_features(self, raw_image, text):
        """
        LAVIS Unified Feature Extraction Interface

        The multimodal feature can be used for multimodal classification.
        The low-dimensional unimodal features can be used to compute cross-modal similarity.

        ref: https://github.com/salesforce/LAVIS#unified-feature-extraction-interface
        """
        if isinstance(raw_image, np.ndarray):
            raw_image = Image.fromarray(raw_image)
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = self.txt_processors["eval"](text)
        sample = {"image": image, "text_input": [text_input]}

        # torch.Size([1, 32, 768]), 32 is the number of queries
        features_multimodal = self.model.extract_features(sample)
        features_image = self.model.extract_features(sample, mode="image")  # torch.Size([1, 32, 768])
        features_text = self.model.extract_features(sample, mode="text")  # torch.Size([1, words, 768])

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

# edge from 2D, 3D bounding box


def comp_bbox2d_edge_with_min_knn(df, cxcy_cols, node_ids, k=3):
    """
    df: node features DataFrame, need to contain cxcy_cols
    cxcy_cols: the columns of bbox2d center. e.g. bbox2d_cx, bbox2d_cy
    k: the number of nearest neighbors, -1 means dense connection
    """
    node_df = df[cxcy_cols].copy()
    edge_attr = comp_bbox2d_dist_and_angle(node_df, cxcy_cols)
    # edge index from edge_attr
    edge_attr['src'] = np.arange(len(node_ids)).repeat(len(node_ids))
    edge_attr['dst'] = np.tile(np.arange(len(node_ids)), len(node_ids))
    # remove self loop
    mask = np.eye(len(node_ids), dtype=bool).reshape(-1)
    edge_attr = edge_attr[~mask]

    # KNN edge
    if k > 0:
        pivot = edge_attr.pivot(index='dst', columns='src', values='bbox2d_dist')
        knn_dist = pivot.apply(lambda x: x.nsmallest(k).index)
        edge_index = knn_dist.melt().rename(columns={'variable': 'src', 'value': 'dst'})
        edge_attr = edge_attr.merge(edge_index, on=['src', 'dst'])

    edge_index = edge_attr[['src', 'dst']].T
    edge_attr = edge_attr.drop(columns=['src', 'dst'])
    assert edge_index.shape[0] == 2, f'edge_index.shape[0] = {edge_index.shape[0]}'
    return edge_index, edge_attr


def comp_bbox3d_edge_with_min_knn(df, xyz_cols, node_ids, k=3):
    """
    df: node features DataFrame, need to contain xyz_cols
    xyz_cols: the columns of bbox3d center. e.g. bbox3d_tx, bbox3d_ty, bbox3d_tz
    k: the number of nearest neighbors, -1 means dense connection
    """
    # edge attr
    node_df = df[xyz_cols].copy()
    edge_attr = comp_bbox3d_dist_and_quat(node_df, xyz_cols)

    # edge index from edge_attr
    edge_attr['src'] = np.arange(len(node_ids)).repeat(len(node_ids))
    edge_attr['dst'] = np.tile(np.arange(len(node_ids)), len(node_ids))
    # remove self loop
    mask = np.eye(len(node_ids), dtype=bool).reshape(-1)  # (n*n, 1)
    edge_attr = edge_attr[~mask]

    # KNN edge
    if k > 0:
        pivot = edge_attr.pivot(index='dst', columns='src', values='bbox3d_dist')
        knn_dist = pivot.apply(lambda x: x.nsmallest(k).index)
        edge_index = knn_dist.melt().rename(columns={'variable': 'src', 'value': 'dst'})
        edge_attr = edge_attr.merge(edge_index, on=['src', 'dst'])

    # update edge_attr
    edge_index = edge_attr[['src', 'dst']].T
    edge_attr = edge_attr.drop(columns=['src', 'dst'])
    assert edge_index.shape[0] == 2, f'edge_index.shape[0] = {edge_index.shape[0]}'
    return edge_index, edge_attr


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
