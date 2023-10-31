import warnings
from itertools import combinations
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as scipy_R
from scipy.optimize import minimize as scipy_minimize

# ----- Graph -----


def comp_graph_overlap(g1, g2):
    all_inst_ids = list(set(g1.inst_ids) | set(g2.inst_ids))
    anchor_inst_ids = list(set(g1.inst_ids) & set(g2.inst_ids))
    e1i = [g1.inst2node[inst_id] for inst_id in anchor_inst_ids]
    e1j = [g1.inst2node[inst_id] for inst_id in g1.inst_ids if inst_id not in anchor_inst_ids]
    e2i = [g2.inst2node[inst_id] for inst_id in anchor_inst_ids]
    e2j = [g2.inst2node[inst_id] for inst_id in g2.inst_ids if inst_id not in anchor_inst_ids]
    return e1i, e1j, e2i, e2j


# ----- Pose Estimation -----

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
