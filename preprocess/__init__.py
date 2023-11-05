import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
from pprint import pprint
import pickle as pkl
import glob
import shutil
import copy

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_R
from tqdm.auto import tqdm


class MapGraph:
    def __init__(self, data):
        self.data = data

        # graph meta
        self.total_inst_cnt = self.data['total_obj_cnt']
        self.inst_ids = self.data['inst_ids']
        self.node_ids = self.data['node_ids']
        self.inst2node = self.data['inst2node']
        # node
        self.node_feat = self.data['features']
        # edge
        edge_index_3d, = self.data['edge']['3d']['index'],
        edge_attr_3d = self.data['edge']['3d']['attr']
        self.edge_index = {
            '3d': edge_index_3d
        }
        self.edge_attr = {
            '3d': edge_attr_3d
        }

    def __len__(self):
        return len(self.node_ids)


class QueryGraph:
    def __init__(self, data):
        self.data = data

        self.step = self.data['step']
        # camera
        self.camera_pose = self.data['camera_pose']
        self.camera_intrinsics = self.data['camera_intrinsics']
        self.img_size = self.data['img_size']

        # graph meta
        self.total_inst_cnt = self.data['total_obj_cnt']
        self.inst_ids = self.data['inst_ids']
        self.node_ids = self.data['node_ids']
        self.inst2node = self.data['inst2node']
        # node
        self.node_feat = self.data['features']
        # edge
        edge_index_2d = self.data['edge']['2d']['index']
        edge_index_3d = self.data['edge']['3d']['index']
        edge_attr_2d = self.data['edge']['2d']['attr']
        edge_attr_3d = self.data['edge']['3d']['attr']
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
