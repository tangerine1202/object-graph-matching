import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import warnings
from torch_geometric.nn import GATv2Conv
import torch_geometric.nn as pygnn
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as scipy_R
from scipy.optimize import minimize as scipy_minimize
from solo2graph import MapGraph, QueryGraph, PairedGraph

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'torch device: {device}')


class CustomModel(nn.Module):
    def __init__(self, edge_attr_dim, emb_dim=64, sinkhorn_iters=50, match_threshold=0.2, bin_score=1.0):
        super(CustomModel, self).__init__()
        self.edge_attr_dim = edge_attr_dim
        self.emb_dim = emb_dim

        # matching (SuperGlue method)
        self.sinkhorn_iters = sinkhorn_iters
        self.match_threshold = match_threshold
        self.bin_score = torch.tensor(bin_score).to(device)

        # self.position_encoder = nn.Sequential(
        #     nn.Conv1d(2, 64, kernel_size=1, bias=True),
        #     nn.InstanceNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, emb_dim, kernel_size=1, bias=True),
        # )
        # self.bbox_encoder = nn.Sequential(
        #     nn.Conv1d(5, 64, kernel_size=1, bias=True),
        #     nn.InstanceNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, emb_dim, kernel_size=1, bias=True),
        # )
        self.bbox3d_encoder = nn.Sequential(
            nn.InstanceNorm1d(10),
            nn.Conv1d(10, 64, kernel_size=1, bias=True),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, emb_dim, kernel_size=1, bias=True),
        )
        self.txt_encoder = nn.Sequential(
            nn.Conv1d(768, 128, kernel_size=1, bias=True),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, emb_dim, kernel_size=1, bias=True),
        )
        self.edge_attr_encoder = nn.Sequential(
            # nn.InstanceNorm1d(edge_attr_dim), # NOTE: do not use InstanceNorm1d for bbox3d
            nn.Conv1d(edge_attr_dim, 64, kernel_size=1, bias=True),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, emb_dim, kernel_size=1, bias=True),
        )
        self.layers = pygnn.Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(emb_dim * 2, emb_dim, edge_dim=emb_dim), 'x, edge_index, edge_attr -> x'),
            (nn.ReLU(inplace=True)),
            (GATv2Conv(emb_dim, emb_dim, edge_dim=emb_dim), 'x, edge_index, edge_attr -> x'),
            (nn.ReLU(inplace=True)),
            (nn.Linear(emb_dim, emb_dim), 'x -> x'),
        ])

    def forward(self, data_dict):
        edge_index = data_dict['edge_index'].squeeze(0)
        edge_attr = data_dict['edge_attr']

        # node_position = data_dict['node_position']
        # node_bbox = data_dict['node_bbox']
        node_bbox3d = data_dict['node_bbox3d']
        node_text = data_dict['node_text']

        # node_position = self.position_encoder(node_position.transpose(1, 2)).transpose(1, 2).squeeze(0)
        # node_bbox = self.bbox_encoder(node_bbox.transpose(1, 2)).transpose(1, 2).squeeze(0)
        node_bbox3d = self.bbox3d_encoder(node_bbox3d.transpose(1, 2)).transpose(1, 2).squeeze(0)
        node_text = self.txt_encoder(node_text.transpose(1, 2)).transpose(1, 2).squeeze(0)

        node_attr = torch.cat((
            # node_position,
            # node_bbox,
            node_bbox3d,
            node_text
        ), dim=1)

        edge_attr = self.edge_attr_encoder(edge_attr.transpose(1, 2)).transpose(1, 2).squeeze(0)
        # fusion
        node_attr = self.layers(node_attr, edge_index, edge_attr).unsqueeze(0)

        # matching (SuperGlue method)
        # ref: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
        mdesc0 = node_attr[:, :data_dict['n1']].transpose(1, 2)
        mdesc1 = node_attr[:, data_dict['n1']:].transpose(1, 2)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.emb_dim**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores,
            self.bin_score,
            iters=self.sinkhorn_iters,
        )

        # Get the matches with score above "match_threshold".
        matches = matches_from_scores(scores, self.match_threshold)

        pred_dict = {
            'matches0': matches['matches0'],
            'matches1': matches['matches1'],
            'matching_scores0': matches['matching_scores0'],
            'matching_scores1': matches['matching_scores1'],
            'scores': scores,
        }

        if not self.training:
            # compute pose
            pred_pose = compute_pose(pred_dict, data_dict)
            pred_dict['pose'] = pred_pose

        return pred_dict


def compute_pose(pred_dict, data_dict):
    pred_e1i = np.array([idx for idx, v in enumerate(pred_dict['matches0']) if v != -1])
    pred_e2i = np.array([v.item() for idx, v in enumerate(pred_dict['matches0']) if v != -1])
    corrs = np.stack([pred_e1i, pred_e2i], axis=1)  # (n, 2)
    if len(pred_e1i) == 0:
        pred_pose, error = None, None
    else:
        if isinstance(data_dict['node_bbox3d'], torch.Tensor):
            bbox3d_t1 = data_dict['node_bbox3d'][0].cpu().numpy()
            bbox3d_t2 = data_dict['node_bbox3d'][0].cpu().numpy()
        bbox3d_t1 = bbox3d_t1[:data_dict['n1'].item(), :3]
        bbox3d_t2 = bbox3d_t2[data_dict['n1'].item():, :3]
        pred_R, pred_t = pose_by_ICP_with_SVD_init(bbox3d_t1, bbox3d_t2, corrs)

        pred_pose = np.concatenate([pred_t, pred_R.as_quat()])
    return pred_pose


def pose_by_ICP_with_SVD_init(src, tgt, corrs=None, max_distance=1):
    if corrs is None:
        assert src.shape == tgt.shape
        corrs = np.stack([np.arange(len(src)), np.arange(len(tgt))], axis=1)
    assert corrs.shape[1] == 2
    src = src[corrs[:, 0]]
    tgt = tgt[corrs[:, 1]]
    # init with SVD
    R_init, t_init = pose_by_minimize_t_with_SVD(src, tgt)
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


def pose_by_minimize_t_with_RANSAC_SVD(src, tgt, corrs, max_distance=0.1, min_inliers=3,
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
    if corrs_ratio is not None and outlier_ratio is not None:
        max_iters = max(max_iters, int(np.log(1 - corrs_ratio) / np.log(1 - (1 - outlier_ratio)**sample_size)))
    elif corrs_ratio is not None or outlier_ratio is not None:
        warnings.warn('corrs_ratio and outlier_ratio should be both set or both None')

    src = src[corrs[:, 0]]  # (n, 3)
    tgt = tgt[corrs[:, 1]]

    best_R, best_t = pose_by_minimize_t_with_SVD(src, tgt)
    best_inliers = np.ones(len(src), dtype=bool)

    if len(src) < sample_size:
        return best_R, best_t

    for _ in range(max_iters):
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


def matches_from_scores(scores, match_threshold: float):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > match_threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    return {
        'matches0': indices0[0],  # use -1 for invalid match
        'matches1': indices1[0],  # use -1 for invalid match
        'matching_scores0': mscores0[0],
        'matching_scores1': mscores1[0],
    }


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
