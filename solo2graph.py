import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import copy
from glob import glob
from pprint import pprint
import pickle as pkl
import shutil

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
# import torchvision.transforms.functional as VF
# from torchvision.models import resnet50, ResNet50_Weights

from lavis.models import load_model_and_preprocess

from solo_tool import Solo


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def extract_features(raw_image, text):
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
    # use features_multimodal[:,0,:] for multimodal classification tasks
    feat_multimodal = features_multimodal.multimodal_embeds[:, 0, :].cpu().numpy()

    # uni-modal features
    feat_img = features_image.image_embeds[:, 0, :].cpu().numpy()
    feat_txt = features_text.text_embeds[:, 0, :].cpu().numpy()

    # normalized low-dimensional uni-modal features
    # norm_img: torch.Size([1, 197, 256])
    # norm_tex: torch.Size([1, words, 256])
    feat_norm_img = features_image.image_embeds_proj[:, 0, :].cpu().numpy()
    feat_norm_txt = features_text.text_embeds_proj[:, 0, :].cpu().numpy()
    # similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
    # similarity = (features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t())

    return {
        'multimodal': feat_multimodal,
        'image': feat_img,
        'text': feat_txt,
        'norm_image': feat_norm_img,
        'norm_text': feat_norm_txt,
    }


class Graph:
    def __init__(self, f, solo, drop_no_instance=True):
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
        self.edge_index = self.data['edge_index']
        self.node_feat = self.data['features']

        self.node_attr = self.comp_node_attr()
        self.edge_attr = self.comp_edge_attr()
        # self.adj = self.edge_index2adj()

    def __len__(self):
        return len(self.node_ids)

    def comp_node_attr(self):
        attr = np.empty((len(self.node_ids), 0))
        for feature in self.node_feat.values():
            attr = np.concatenate((attr, feature), axis=1)
        return attr

    def comp_edge_attr(self):
        cols = ['bbox_cx', 'bbox_cy']
        node_df = pd.DataFrame({k: self.node_feat[k][:, 0] for k in cols})
        node_df['node_id'] = self.node_ids
        cross_df = node_df.merge(node_df, how='cross', suffixes=('_src', '_dst'))
        # construct edge features
        diff = cross_df[['bbox_cx_dst', 'bbox_cy_dst']].values - cross_df[['bbox_cx_src', 'bbox_cy_src']].values
        cross_df['bbox_dist'] = np.linalg.norm(diff, axis=1)
        theta = np.arctan2(diff[:, 1], diff[:, 0])
        cross_df['bbox_sin'] = np.sin(theta)
        cross_df['bbox_cos'] = np.cos(theta)
        edge_attr = cross_df[['bbox_dist', 'bbox_sin', 'bbox_cos']]
        # exclude diagonal edges
        mask = np.eye(len(self.node_ids), dtype=bool).reshape(-1)
        edge_attr = edge_attr[~mask].values
        return edge_attr

    def edge_index2adj(self):
        adj = np.zeros((len(self.node_ids), len(self.node_ids)))
        adj[self.edge_index[0], self.edge_index[1]] = 1
        return adj

    def frame_to_data(self, f, solo):
        data_dict = {}
        data_path = solo.output_path
        step = f.step
        cap = f.captures[0]
        metrics = f.metrics
        anno_defs = solo.annotation_definitions
        annos = cap.annotations

        inst = annos['instance segmentation']
        inst_df = inst.instances_df

        # FIXME: skip frames with no instance
        if not inst.has_instance:
            return False, None

        bbox = annos['bounding box']
        bbox_df = bbox.values_df

        meta = metrics['metadata']
        env_meta = meta.env_metadata
        meta_df = meta.instances_df

        # prepare object dataframe
        obj_df = meta_df.copy()
        # filter invisible objects
        obj_df = obj_df[obj_df['instanceId'].isin(inst_df['instanceId'])]
        # rename columns
        obj_df = obj_df.rename(columns={'object_labelName': 'label_name'})
        obj_df['label_id'] = obj_df['label_name'].apply(lambda x: anno_defs['bounding box'].name2id[x])

        # add columns
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
        # mask = obj_df['bbox_w'] * obj_df['bbox_h'] > 500
        # obj_df = obj_df[mask]

        obj_df = obj_df.reset_index(drop=True)

        if len(obj_df) <= 1:
            return False, None

        # extract  features
        bbox_embs = {}
        rgb_path = f'{data_path}/rgb/step{step}.png'
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        for _, obj in obj_df.iterrows():
            top, left, w, h = int(obj['bbox_y0']), int(obj['bbox_x0']), int(obj['bbox_w']), int(obj['bbox_h'])
            crop_img = rgb_img[top:top + h, left:left + w, :]
            features = extract_features(crop_img, obj['label_name'])
            for k, v in features.items():
                if k not in bbox_embs:
                    bbox_embs[k] = []
                bbox_embs[k].append(v)
        bbox_embs = {k: np.vstack(v) for k, v in bbox_embs.items()}

        features = {
            **{k: np.asarray(v).reshape(-1, 1) for k, v in obj_df.items()},
            **{f'bbox_{k}': v for k, v in bbox_embs.items()}
        }

        node_ids = obj_df.index.to_list()
        edge_index = np.array([[i, j] for i in node_ids for j in node_ids if i != j]).T
        inst_ids = obj_df['inst_id'].to_list()
        inst2node = {inst_id: node_id for node_id, inst_id in enumerate(inst_ids)}

        data_dict = {
            'node_ids': node_ids,
            'edge_index': edge_index,
            'inst_ids': inst_ids,
            'inst2node': inst2node,
            'features': features,
            'total_obj_cnt': len(obj_df),
            'step': f.step,
            'camera_pose': cap.camera_pose,
            'camera_intrinsics': cap.projectionMatrix,
        }

        return True, data_dict


class PairedGraph:
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2
        self.n1 = len(g1)
        self.n2 = len(g2)
        self.n = self.n1 + self.n2

        self.node_attr = np.concatenate([g1.node_attr, g2.node_attr], axis=0)
        self.node_feat = {}
        self.comp_node_feat()
        # edge index
        self.comp_inter_edge()
        # self.adj = np.zeros((self.n, self.n))
        # self.adj[:self.n1, :self.n1] = self.g1.adj
        # self.adj[self.n1:, self.n1:] = self.g2.adj
        # self.adj[self.n1:, :self.n1] = self.adj_g2tog1
        # self.adj[:self.n1, self.n1:] = self.adj_g1tog2
        # self.edge_index = self.adj2edge_index(self.adj)
        self.edge_index = np.concatenate([self.g1.edge_index, self.g2.edge_index + self.n1], axis=1)
        # edge attr
        # self.edge_attr = self.comp_edge_attr()
        self.edge_attr = np.concatenate([self.g1.edge_attr, self.g2.edge_attr], axis=0)
        assert len(self.edge_attr) == len(self.edge_index[0])

        self.comp_matching()

    def comp_node_feat(self):
        for name in self.g1.node_feat.keys():
            self.node_feat[name] = np.concatenate([self.g1.node_feat[name], self.g2.node_feat[name]], axis=0)

    def comp_edge_attr(self):
        pass

    def comp_inter_edge(self):
        # compute edge between g1 and g2
        self.edge_g1tog2 = np.array([[g1_node_id, g2_node_id]
                                    for g1_node_id in self.g1.node_ids for g2_node_id in self.g2.node_ids]).T
        self.edge_g2tog1 = np.array([[g2_node_id, g1_node_id]
                                    for g2_node_id in self.g2.node_ids for g1_node_id in self.g1.node_ids]).T
        adj_g1tog2 = np.zeros((len(self.g1), len(self.g2)))
        adj_g2tog1 = np.zeros((len(self.g2), len(self.g1)))
        adj_g1tog2[self.edge_g1tog2[0], self.edge_g1tog2[1]] = 1
        adj_g2tog1[self.edge_g2tog1[0], self.edge_g2tog1[1]] = 1
        # self.adj_g1tog2 = adj_g1tog2
        # self.adj_g2tog1 = adj_g2tog1

    def adj2edge_index(self, adj):
        edge_index = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == 1:
                    edge_index.append([i, j])
        edge_index = np.array(edge_index).T
        return edge_index

    def comp_matching(self):
        self.all_inst_ids = list(set(self.g1.inst_ids) | set(self.g2.inst_ids))
        self.anchor_inst_ids = list(set(self.g1.inst_ids) & set(self.g2.inst_ids))
        self.e1i = [g1.inst2node[inst_id] for inst_id in self.anchor_inst_ids]
        self.e1j = [g1.inst2node[inst_id] for inst_id in self.g1.inst_ids if inst_id not in self.anchor_inst_ids]
        self.e2i = [g2.inst2node[inst_id] for inst_id in self.anchor_inst_ids]
        self.e2j = [g2.inst2node[inst_id] for inst_id in self.g2.inst_ids if inst_id not in self.anchor_inst_ids]

    def __len__(self):
        return self.n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        type=str,
                        help='path to solo output',
                        required=True)
    parser.add_argument('--reorganized',
                        action='store_true',
                        help='whether the data has been reorganized')
    parser.add_argument('--move',
                        action='store_true',
                        help='move files instead of copying')
    parser.add_argument('--output_dir',
                        type=str,
                        default='data',
                        help='name of the reorganized output directory')
    parser.add_argument('--single_graph_dir',
                        type=str,
                        default='graph',
                        help='name of the single graph directory')
    parser.add_argument('--keep_no_instance',
                        action='store_true',
                        help='skip frames with no instance')
    parser.add_argument('--paired_graph_dir',
                        type=str,
                        default='paired_graph',
                        help='name of the paired graph directory')
    args = parser.parse_args()

    SINGLE_GRAPH_PATH = os.path.join(args.path, args.single_graph_dir)
    PAIRED_GRAPH_PATH = os.path.join(args.path, args.paired_graph_dir)

    if os.path.exists(SINGLE_GRAPH_PATH):
        print(f'remove {SINGLE_GRAPH_PATH}')
        shutil.rmtree(SINGLE_GRAPH_PATH)
    os.mkdir(SINGLE_GRAPH_PATH)

    if os.path.exists(PAIRED_GRAPH_PATH):
        print(f'remove {PAIRED_GRAPH_PATH}')
        shutil.rmtree(PAIRED_GRAPH_PATH)
    os.mkdir(PAIRED_GRAPH_PATH)

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

    solo = Solo(args.path, args.output_dir, is_reorganized=args.reorganized, move=args.move)

    for f in tqdm(solo.frames()):
        graph = Graph(f, solo, drop_no_instance=(not args.keep_no_instance))
        if not graph.valid and not args.keep_no_instance:
            print(f'step {f.step} is invalid')
            continue
        graph_path = os.path.join(SINGLE_GRAPH_PATH, f'step{f.step}.pkl')
        pkl.dump(graph, open(graph_path, 'wb'))

    paths = glob(os.path.join(SINGLE_GRAPH_PATH, '*.pkl'))
    for p1 in tqdm(paths):
        for p2 in paths:
            if p1 == p2:
                continue
            g1 = pkl.load(open(p1, 'rb'))
            g2 = pkl.load(open(p2, 'rb'))
            fname1 = os.path.basename(p1).split('.')[0]
            fname2 = os.path.basename(p2).split('.')[0]
            paired_graph_path = os.path.join(PAIRED_GRAPH_PATH, f'{fname1}_{fname2}.pkl')

            pg = PairedGraph(g1, g2)
            num_overlap = len(pg.e1i)
            num_edges = len(pg.edge_index.T)
            if num_overlap < 1 or num_edges < 1:
                continue
            if num_overlap < 4:
                continue
                # no enough matching for p3p

            pkl.dump(pg, open(paired_graph_path, 'wb'))
