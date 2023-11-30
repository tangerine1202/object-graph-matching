import warnings
from itertools import combinations
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as scipy_R
from scipy.optimize import minimize as scipy_minimize


def read_scannet_img(scene_id, frame_id, scannet_path='data/ScanNet/scannet_by_tr3d_with_DepthImage'):
    img_path = f'{scannet_path}/posed_images/{scene_id}/{frame_id}.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return img


def data_dict_to_device(data_dict, device):
    return {k: v.to(device) for k, v in data_dict.items()}


def invert_Rt(R, t):
    assert isinstance(R, scipy_R)
    assert t.shape == (3,)
    R_inv = R.inv()
    t_inv = -(R_inv.apply(t))
    return R_inv, t_inv


def adjust_intrinsics_with_img_size(K, orig_img_size, img_size):
    # Calculate scale factors
    s = np.array(img_size) / np.array(orig_img_size)
    # Scale the intrinsic parameters
    K_new = K.copy()
    K_new[0, :] *= s[0]
    K_new[1, :] *= s[1]
    return K_new

# ----- Graph -----


def comp_graph_overlap(g1, g2):
    # all_inst_ids = list(set(g1.inst_ids) | set(g2.inst_ids))
    anchor_inst_ids = list(set(g1.inst_ids) & set(g2.inst_ids))
    e1i = [g1.inst2node[inst_id] for inst_id in anchor_inst_ids]
    e1j = [g1.inst2node[inst_id] for inst_id in g1.inst_ids if inst_id not in anchor_inst_ids]
    e2i = [g2.inst2node[inst_id] for inst_id in anchor_inst_ids]
    e2j = [g2.inst2node[inst_id] for inst_id in g2.inst_ids if inst_id not in anchor_inst_ids]
    return e1i, e1j, e2i, e2j

# ----- bbox3d -----


def transform_bbox3d(bbox3d, R, t):
    assert bbox3d.shape[1] == 10
    assert isinstance(R, scipy_R)
    assert t.shape == (3,)
    t_local = bbox3d[:, :3].copy()
    q_local = bbox3d[:, 3:7].copy()
    s_local = bbox3d[:, 7:].copy()
    t_local = R.apply(t_local) + t
    # transform q_local with R
    q_local = np.asarray([(R * scipy_R.from_quat(q)).as_quat() for q in q_local])
    s_local = s_local.copy()
    bbox3d_local = np.concatenate([t_local, q_local, s_local], axis=1)
    return bbox3d_local


def corners_of_bbox3d(bbox3d):
    assert bbox3d.shape[1] == 10
    N = bbox3d.shape[0]
    corners = np.zeros((N, 8, 3))
    for i in range(len(bbox3d)):
        t = bbox3d[i, :3]
        r = scipy_R.from_quat(bbox3d[i, 3:7])
        s = bbox3d[i, 7:]
        corners[i] = np.array([
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]) * s
        corners[i] = r.apply(corners[i]) + t
    return corners


def project_bbox3d_to_2d_xyxy(bbox3d, K):
    N = bbox3d.shape[0]
    bbox = np.zeros((N, 4))
    corners = corners_of_bbox3d(bbox3d)
    for i in range(N):
        corners_h = corners[i] @ K.T
        in_view_mask = (corners_h[:, 2:] > 0).flatten()
        if sum(in_view_mask) == 0:
            bbox[i] = np.array([-1, -1, -1, -1])
            continue
        corners_h = corners_h[:, :2] / corners_h[:, 2:]
        bbox[i] = np.array([
            corners_h[:, 0].min(),
            corners_h[:, 1].min(),
            corners_h[:, 0].max(),
            corners_h[:, 1].max(),
        ])
    return bbox

# ----- IoU -----


def compute_intersect_area(xyxy_a, xyxy_b):
    ax0, ay0, ax1, ay1 = xyxy_a
    bx0, by0, bx1, by1 = xyxy_b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return area


def compute_union_area(xyxy_a, xyxy_b):
    ax0, ay0, ax1, ay1 = xyxy_a
    bx0, by0, bx1, by1 = xyxy_b
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    area_intersect = compute_intersect_area(xyxy_a, xyxy_b)
    area_union = area_a + area_b - area_intersect
    return area_union


def compute_IoU_x0y0wh(xywh_a, xywh_b):
    xyxy_a = xywh_a.copy()
    xyxy_a[2:] += xyxy_a[:2]
    xyxy_b = xywh_b.copy()
    xyxy_b[2:] += xyxy_b[:2]
    return compute_IoU_xyxy(xyxy_a, xyxy_b)


def compute_IoU_xyxy(xyxy_a, xyxy_b):
    area_intersect = compute_intersect_area(xyxy_a, xyxy_b)
    area_union = compute_union_area(xyxy_a, xyxy_b)
    iou = area_intersect / area_union
    return iou
