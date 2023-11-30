import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import json

from tqdm.auto import tqdm

from scipy.spatial.transform import Rotation as scipy_R
import numpy as np
import pandas as pd
import cv2

from preprocess import MapGraph, QueryGraph


def read_calib(calib_path):
    calib = json.load(open(calib_path))
    calib['K'] = np.array(calib['K']).reshape(3, 3).T
    calib['R'] = np.array(calib['Rtilt']).reshape(3, 3)
    # FIXME: read from the json
    calib['t'] = np.array([0, 0, 0])
    return calib


def change_3d_coordinate_left_up_front(calib, bbox3d):
    """
    # change 3D coordinate from right-front-up to left-up-front
    # K: intrinsic matrix
    # calib: calibration matrix
    # bbox3d: 3D bbox in right-front-up coordinate
    # return: 3D bbox in left-up-front coordinate
    """
    change_axis = np.array([[-1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
    K = calib['K']
    K[0, 0] *= -1
    K[1, 1] *= -1

    t = calib['t'] @ change_axis
    R_xyz = scipy_R.from_matrix(calib['R']).as_euler('xyz')
    R = scipy_R.from_euler('xyz', [-R_xyz[0], R_xyz[2], R_xyz[1]])

    # bbox3d
    # translation & size
    bbox3d_t = bbox3d.iloc[:, :3] @ change_axis
    bbox3d_s = bbox3d.iloc[:, 3:6] @ change_axis
    # rotation
    bbox3d_orientation = bbox3d[['Orientation1', 'Orientation2']]
    # rotation
    bbox3d_q = np.empty((len(bbox3d), 4))
    for i in range(len(bbox3d)):
        yaw_in_rad = np.arctan2(bbox3d_orientation.iloc[i]['Orientation2'], bbox3d_orientation.iloc[i]['Orientation1'])
        q = scipy_R.from_euler('y', yaw_in_rad)
        bbox3d_q[i] = q.as_quat()

    new_bbox3d = pd.DataFrame(np.hstack([bbox3d_t, bbox3d_q, bbox3d_s]),
                              columns=['tx', 'ty', 'tz',
                                       'qx', 'qy', 'qz', 'qw',
                                       'sx', 'sy', 'sz'])
    return {
        'K': K,
        'R': R,
        't': t,
        'bbox3d': new_bbox3d,
    }


if __name__ == '__main__':
    FNAME = '000001'
    DATA_DIR = './data/sunrgbd_trainval'
    IMAGE_PATH = os.path.join(DATA_DIR, 'image', f'{FNAME}.jpg')
    CALIB_PATH = os.path.join(DATA_DIR, 'calib_json', f'{FNAME}.json')
    LABEL_PATH = os.path.join(DATA_DIR, 'label_v1.csv')

    rgb = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    calib = read_calib(CALIB_PATH)
    labels = pd.read_csv(LABEL_PATH)
    labels = labels[labels['ImageID'] == int(FNAME)]

    bbox_col_rename = {
        '2D_x0': 'x0',
        '2D_y0': 'y0',
        '2D_x_size': 'w',
        '2D_y_size': 'h'
    }
    bbox = labels[bbox_col_rename.keys()].rename(columns=bbox_col_rename)

    change_coord_dict = change_3d_coordinate_left_up_front(
        calib, labels[['3D_x', '3D_y', '3D_z',
                       '3D_x_size', '3D_y_size', '3D_z_size',
                       'Orientation1', 'Orientation2']])
    K = change_coord_dict['K']
    R = change_coord_dict['R']
    t = change_coord_dict['t']
    bbox3d = change_coord_dict['bbox3d']
