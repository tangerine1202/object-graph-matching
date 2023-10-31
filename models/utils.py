import warnings
from itertools import combinations
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as scipy_R
from scipy.optimize import minimize as scipy_minimize


# def compute_pose_from_2Dto3D_bbox(pred_dict, data_dict):
#     # FIXME: using GT matching to test the accuracy of pose estimation
#     pred_e1i = data_dict['e1i'][0].cpu().numpy()
#     pred_e2i = data_dict['e2i'][0].cpu().numpy()
#     # pred_e1i = np.array([idx for idx, v in enumerate(pred_dict['matches0']) if v != -1])
#     # pred_e2i = np.array([v.item() for idx, v in enumerate(pred_dict['matches0']) if v != -1])
#     corrs = np.stack([pred_e1i, pred_e2i], axis=1)  # (n, 2)
#     if len(pred_e1i) == 0:
#         pred_pose = None
#     else:
#         bbox = data_dict['qry_node_bbox'][0][corrs[:, 0], :4]
#         bbox3d = data_dict['map_node_bbox3d'][0][corrs[:, 1]]
#         cam_pose = data_dict['qry_camera_pose'][0]
#         K = data_dict['qry_camera_intrinsics'][0]
#         if isinstance(bbox, torch.Tensor):
#             bbox = bbox.cpu().numpy()
#         if isinstance(bbox3d, torch.Tensor):
#             bbox3d = bbox3d.cpu().numpy()
#         if isinstance(cam_pose, torch.Tensor):
#             cam_pose = cam_pose.cpu().numpy()
#         if isinstance(K, torch.Tensor):
#             K = K.cpu().numpy()
#         # NOTE: we want to estimate the camera pose, so do not use the camera pose as the initial pose
#         # FIXME: init with GT pose eto evaluate the accuracy of pose estimation
#         # bbox3d = transform_bbox3d(bbox3d, scipy_R.from_quat(cam_pose[3:]), cam_pose[:3])
#         pred_R, pred_t = pose_by_minimize_bbox3d_projection(bbox3d, bbox, K)
#         pred_pose = np.concatenate([pred_t, pred_R.as_quat()])
#     return pred_pose


# def transform_bbox3d(bbox3d, R, t):
#     R_inv = R.inv()
#     t_local = bbox3d[:, :3].copy()
#     q_local = bbox3d[:, 3:7].copy()
#     s_local = bbox3d[:, 7:].copy()
#     t_local = (t_local - t) @ R_inv.as_matrix()
#     q_local = np.array([(R_inv * scipy_R.from_quat(q)).as_quat() for q in q_local]).reshape(-1, 4)
#     s_local = s_local @ R_inv.as_matrix()
#     bbox3d_local = np.concatenate([t_local, q_local, s_local], axis=1)
#     return bbox3d_local


# def pose_by_minimize_bbox3d_projection(bbox3d, bbox2d, K, max_iter=None):
#     def bbox3d_projection_cost_function(transformation_flat, bbox3d, bbox2d_ccwh, K):
#         transformation = transformation_flat.reshape(4, 4)
#         R, t = transformation[:3, :3], transformation[:3, 3]
#         transformed_bbox3d = transform_bbox3d(bbox3d, scipy_R.from_matrix(np.array(R)), t)
#         transformed_bbox2d_xyxy = project_bbox3d_to_2d(transformed_bbox3d, K)
#         bbox2d_wh = bbox2d_ccwh[:, 2:]
#         bbox2d_xyxy = bbox2d_ccwh.copy()
#         bbox2d_xyxy[:, :2] -= bbox2d_wh
#         bbox2d_xyxy[:, 2:] += bbox2d_wh
#         # calculate IoU of transformed_bbox2d and bbox2d
#         total_intersect_area = 0
#         total_union_area = 0
#         for i in range(len(bbox2d_xyxy)):
#             intersect_area = compute_intersect_area(bbox2d_xyxy[i], transformed_bbox2d_xyxy[i])
#             union_area = compute_union_area(bbox2d_xyxy[i], transformed_bbox2d_xyxy[i])
#             total_intersect_area += intersect_area
#             total_union_area += union_area
#         iou = total_intersect_area / total_union_area
#         error = 1 - iou
#         return error

#         # error = transformed_bbox2d_xyxy - bbox2d_xyxy
#         # return np.sum(np.linalg.norm(error, axis=1))

#     T_init_flat = np.eye(4).flatten()
#     options = None if max_iter is None else {'maxiter': max_iter}

#     result = scipy_minimize(
#         bbox3d_projection_cost_function, T_init_flat, args=(bbox3d, bbox2d, K),
#         method='L-BFGS-B', options=options)
#     print('obj:', result.fun)

#     T = result.x.reshape(4, 4)
#     R, t = scipy_R.from_matrix(T[:3, :3]), T[:3, 3]
#     # error = result.fun
#     return R, t


# def project_bbox3d_to_2d(bbox3d, K):
#     def corners_of_bbox3d(bbox3d):
#         corners = np.zeros((8, 3))
#         t = bbox3d[:3]
#         q = bbox3d[3:7]
#         s = bbox3d[7:]
#         r = scipy_R.from_quat(q)
#         corners = np.array([
#             [1, 1, 1],
#             [1, 1, -1],
#             [1, -1, 1],
#             [1, -1, -1],
#             [-1, 1, 1],
#             [-1, 1, -1],
#             [-1, -1, 1],
#             [-1, -1, -1],
#         ]) * s
#         corners = r.apply(corners) + t
#         return corners

#     N = bbox3d.shape[0]
#     bbox = np.zeros((N, 4))
#     for i in range(N):
#         corners = corners_of_bbox3d(bbox3d[i])
#         corners_h = (K @ corners.T).T
#         corners_h = corners_h[:, :2] / corners_h[:, 2:]
#         bbox[i] = np.array([
#             corners_h[:, 0].min(),
#             corners_h[:, 1].min(),
#             corners_h[:, 0].max(),
#             corners_h[:, 1].max(),
#         ])
#         bbox[i] = np.clip(bbox[i], 0, 320)
#     return bbox


# def compute_pose_from_bbox3d(pred_dict, data_dict):
#     pred_e1i = np.array([idx for idx, v in enumerate(pred_dict['matches0']) if v != -1])
#     pred_e2i = np.array([v.item() for idx, v in enumerate(pred_dict['matches0']) if v != -1])
#     corrs = np.stack([pred_e1i, pred_e2i], axis=1)  # (n, 2)
#     if len(pred_e1i) == 0:
#         pred_pose = None
#     else:
#         if isinstance(data_dict['node_bbox3d'], torch.Tensor):
#             bbox3d_t1 = data_dict['node_bbox3d'][0].cpu().numpy()
#             bbox3d_t2 = data_dict['node_bbox3d'][0].cpu().numpy()
#         bbox3d_t1 = bbox3d_t1[:data_dict['n1'].item(), :3]
#         bbox3d_t2 = bbox3d_t2[data_dict['n1'].item():, :3]
#         pred_R, pred_t = pose_by_ICP_with_corrs_init(bbox3d_t1, bbox3d_t2, corrs)
#         pred_pose = np.concatenate([pred_t, pred_R.as_quat()])
#     return pred_pose


# def compute_intersect_area(xyxy_a, xyxy_b):
#     ax0, ay0, ax1, ay1 = xyxy_a
#     bx0, by0, bx1, by1 = xyxy_b
#     x0 = max(ax0, bx0)
#     y0 = max(ay0, by0)
#     x1 = min(ax1, bx1)
#     y1 = min(ay1, by1)
#     area = max(0, x1 - x0) * max(0, y1 - y0)
#     return area


# def compute_union_area(xyxy_a, xyxy_b):
#     ax0, ay0, ax1, ay1 = xyxy_a
#     bx0, by0, bx1, by1 = xyxy_b
#     area_a = (ax1 - ax0) * (ay1 - ay0)
#     area_b = (bx1 - bx0) * (by1 - by0)
#     area_intersect = compute_intersect_area(xyxy_a, xyxy_b)
#     area_union = area_a + area_b - area_intersect
#     return area_union


def pose_by_ICP_with_corrs_init(src, tgt, corrs=None, max_distance=1):
    if corrs is None:
        assert src.shape == tgt.shape
        corrs = np.stack([np.arange(len(src)), np.arange(len(tgt))], axis=1)
    assert corrs.shape[1] == 2
    src = src[corrs[:, 0]]
    tgt = tgt[corrs[:, 1]]
    # init with SVD
    R_init, t_init = pose_by_minimize_t_with_RANSAC_SVD(src, tgt, max_iters=30)
    # ICP
    if len(src) < 3:
        return R_init, t_init
    trans_init = np.eye(4)
    trans_init[:3, :3] = R_init.as_matrix()
    trans_init[:3, 3] = t_init
    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src)
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, max_distance, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),)
    R = scipy_R.from_matrix(np.array(reg_p2p.transformation[:3, :3]))
    t = np.array(reg_p2p.transformation[:3, 3])
    return R, t


def pose_by_minimize_t_with_RANSAC_SVD(src, tgt, corrs=None, max_distance=0.1, min_inliers=3,
                                       max_iters=100, corrs_ratio=None, outlier_ratio=None):
    """
    # corrs_ratio: the ratio of correspondences that are inliers
    # outlier_ratio: the ratio of outliers in the data
    """
    if corrs is None:
        assert src.shape == tgt.shape
        corrs = np.stack([np.arange(len(src)), np.arange(len(tgt))], axis=1)
    assert corrs.shape[1] == 2

    sample_size = 3
    src = src[corrs[:, 0]]  # (n, 3)
    tgt = tgt[corrs[:, 1]]

    best_R, best_t = pose_by_minimize_t_with_SVD(src, tgt)
    best_inliers = np.ones(len(src), dtype=bool)

    if len(src) < sample_size:
        return best_R, best_t

    if corrs_ratio is not None and outlier_ratio is not None:
        max_iters = min(max_iters, int(np.log(1 - corrs_ratio) / np.log(1 - (1 - outlier_ratio)**sample_size)))
    elif corrs_ratio is not None or outlier_ratio is not None:
        warnings.warn('corrs_ratio and outlier_ratio should be both set or both None')

    # if combination is less than max_iters, use all combinations instead of random sampling
    max_combs = np.math.comb(len(src), sample_size)
    if max_combs <= max_iters:
        max_iters = max_combs
        combs = np.asarray(list(combinations(range(len(src)), sample_size)))

    for i in range(max_iters):
        if max_combs <= max_iters:
            indices = combs[i]
            src_sample = src[indices]
            tgt_sample = tgt[indices]
        else:
            # Randomly select correspondence pairs
            random_indices = np.random.choice(len(src), sample_size, replace=False)
            src_sample = src[random_indices]
            tgt_sample = tgt[random_indices]

        # Compute the transformation
        R, t = pose_by_minimize_t_with_SVD(src_sample, tgt_sample)

        # Calculate the error for all correspondences
        errors = np.linalg.norm((src @ R.as_matrix().T + t) - tgt, axis=1)

        # Label correspondences as inliers or outliers based on the threshold
        inliers = np.where(errors < max_distance)[0]

        if len(inliers) >= min_inliers and len(inliers) > len(best_inliers):
            best_R, best_t = R, t
            best_inliers = inliers

    return best_R, best_t  # , best_inliers


def pose_by_minimize_t_with_SVD(src, tgt, corrs=None):
    if corrs is None:
        assert src.shape == tgt.shape
        corrs = np.stack([np.arange(len(src)), np.arange(len(tgt))], axis=1)
    assert corrs.shape[1] == 2

    src = src[corrs[:, 0]]  # (n, 3)
    tgt = tgt[corrs[:, 1]]
    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)
    src_centered = src - src_center
    tgt_centered = tgt - tgt_center

    H = src_centered.T @ tgt_centered  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = tgt_center - R @ src_center

    R = scipy_R.from_matrix(R)
    return R, t


def pose_by_minimize_t_with_iter(src, tgt, corrs=None, max_iter=None):
    def icp_cost_function(transformation_flat, src, tgt):
        transformation = transformation_flat.reshape(4, 4)
        R, t = transformation[:3, :3], transformation[:3, 3]
        transformed_src = src @ R.T + t  # (n, 3)
        error = transformed_src - tgt
        return np.sum(np.linalg.norm(error, axis=1))

    if corrs is None:
        assert src.shape == tgt.shape
        corrs = np.stack([np.arange(len(src)), np.arange(len(tgt))], axis=1)
    assert corrs.shape[1] == 2
    src = src[corrs[:, 0]]
    tgt = tgt[corrs[:, 1]]

    T_init_flat = np.eye(4).flatten()
    options = {'maxiter': max_iter} if max_iter is not None else {}

    result = scipy_minimize(
        icp_cost_function, T_init_flat, args=(src, tgt),
        method='L-BFGS-B', options=options)

    T = result.x.reshape(4, 4)
    R, t = scipy_R.from_matrix(T[:3, :3]), T[:3, 3]
    # error = result.fun
    return R, t
