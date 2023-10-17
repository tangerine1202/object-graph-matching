import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from torch_geometric.nn import GATv2Conv
import torch_geometric.nn as pygnn
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from scipy.optimize import minimize as scipy_minimize

if torch.backends.mps.is_available():
    device = torch.device('mps')
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device('cuda')
print(f'torch device: {device}')


class CustomModel(nn.Module):
    def __init__(self, node_attr_dim, visual_dim, edge_attr_dim, emb_dim=64, sinkhorn_iters=50, match_threshold=0.2, bin_score=1.0):
        super(CustomModel, self).__init__()
        self.node_attr_dim = node_attr_dim
        self.visual_dim = visual_dim
        self.edge_attr_dim = edge_attr_dim
        self.emb_dim = emb_dim
        # matching (SuperGlue method)
        self.sinkhorn_iters = sinkhorn_iters
        # FIXME: adjust the match_threshold
        self.match_threshold = match_threshold
        # FIXME: what is the meaning of bin_score?
        self.bin_score = torch.tensor(bin_score).to(device)

        self.node_attr_encoder = nn.Sequential(
            nn.Linear(node_attr_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.bbox_encoder = nn.Sequential(
            nn.Linear(visual_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.rgb_encoder = nn.Sequential(
            nn.Linear(visual_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        self.fusion = pygnn.Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(emb_dim * 1, emb_dim, edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            (nn.ReLU(inplace=True)),
            (GATv2Conv(emb_dim, emb_dim, edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            (nn.ReLU(inplace=True)),
        ])
        self.aggr = pygnn.Sequential('x, ptr', [
            (pygnn.aggr.MaxAggregation(), 'x, ptr=ptr -> x'),
            (nn.ReLU(inplace=True)),
            (nn.Linear(emb_dim, emb_dim), 'x -> x'),
            (nn.ReLU(inplace=True)),
        ])

        self.pose_lin = nn.Linear(emb_dim, 7)

    def forward(self, data_dict):
        edge_index = data_dict['edge_index']
        edge_attr = data_dict['edge_attr']
        edge_index = edge_index.squeeze(0)
        edge_attr = edge_attr.squeeze(0)

        node_attr = F.normalize(data_dict['node_attr'], dim=-1)
        node_attr_embs = self.node_attr_encoder(node_attr)

        bbox_embs = F.normalize(data_dict['bbox_embs'], dim=-1)
        bbox_embs = self.bbox_encoder(bbox_embs)

        g1_rgb_emb = F.normalize(data_dict['g1_rgb_emb'], dim=-1)
        g1_rgb_emb = self.rgb_encoder(g1_rgb_emb)
        g2_rgb_emb = F.normalize(data_dict['g2_rgb_emb'], dim=-1)
        g2_rgb_emb = self.rgb_encoder(g2_rgb_emb)

        # append global visual features to node visual features
        # visual_global_emb = torch.cat(
        #   [g1_rgb_emb.repeat(data_dict['g1_node_count'], 1),
        #    g2_rgb_emb.repeat(data_dict['g2_node_count'], 1)], dim=0)
        # visual_global_emb = visual_global_emb.unsqueeze(0)

        fused_node_embs = torch.cat([node_attr_embs], dim=-1)
        fused_node_embs = fused_node_embs.squeeze(0)
        fused_node_embs = self.fusion(fused_node_embs, edge_index, edge_attr)
        fused_node_embs = fused_node_embs.unsqueeze(0)

        # matching (SuperGlue method)
        # ref: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
        mdesc0 = fused_node_embs[:, :data_dict['g1_node_count']].transpose(1, 2)
        mdesc1 = fused_node_embs[:, data_dict['g1_node_count']:].transpose(1, 2)

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

        # P3P pose estimation
        # pred_pose0 = None
        # pred_pose1 = None
        # if data_dict['g1_node_count'] >= 4 and data_dict['g2_node_count'] >= 4:
        #     topk_indices0 = torch.topk(matches['matching_scores0'], 4)[1]
        #     topk_indices1 = torch.topk(matches['matching_scores1'], 4)[1] + data_dict['g1_node_count'].item()
        #     p2d_0 = data_dict['bbox'][0, topk_indices0, :2]
        #     p2d_1 = data_dict['bbox'][0, topk_indices1, :2]
        #     p3d_0 = data_dict['obj_pose_in_W'][0, topk_indices0]
        #     p3d_1 = data_dict['obj_pose_in_W'][0, topk_indices1]
        #     pred_pose0 = p3p(p2d_0, p3d_0, data_dict['g1_camera_intrinsics'][0])
        #     pred_pose1 = p3p(p2d_1, p3d_1, data_dict['g2_camera_intrinsics'][0])
        #     pred_pose0 = torch.tensor(pred_pose0, dtype=torch.float).unsqueeze(0)
        #     pred_pose1 = torch.tensor(pred_pose1, dtype=torch.float).unsqueeze(0)

        # aggr_ptr = torch.tensor([0, data_dict['g1_node_count'], data_dict['total_node_count']], dtype=torch.long)
        # graph_emb = self.aggr(fused_node_embs, ptr=aggr_ptr)

        # pose1 = self.pose_lin(graph_emb[:, 0, :])
        # pose2 = self.pose_lin(graph_emb[:, 1, :])

        return {
            'matches0': matches['matches0'],
            'matches1': matches['matches1'],
            'matching_scores0': matches['matching_scores0'],
            'matching_scores1': matches['matching_scores1'],
            'scores': scores,
            'node_attr_embs': node_attr_embs,
            'bbox_embs': bbox_embs,
            'joint_embs': fused_node_embs,
            # 'pose0': pred_pose0,
            # 'pose1': pred_pose1,
        }


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


def p3p(image_points, world_points, K):
    assert image_points.shape[0] == 4 and world_points.shape[0] == 4, "P3P requires exactly 3 point correspondences."
    image_points = image_points.cpu().numpy()
    world_points = world_points.cpu().numpy()
    K = K.cpu().numpy()

    def objective_function(vars):
        R, t = vars[:9].reshape(3, 3), vars[9:]
        t = t[:, np.newaxis]
        projected_points = K @ (R @ world_points.T + t)
        projected_points /= projected_points[2]  # Normalize by the depth
        projected_points = projected_points[:2].T  # Take the first two rows and transpose
        error = np.sum((projected_points - image_points)**2)
        return error

    # Initial estimate for the camera pose (R, t)
    initial_guess = np.concatenate([np.eye(3).flatten(), np.zeros(3)])

    result = scipy_minimize(objective_function, initial_guess, method='L-BFGS-B')
    if result.success:
        estimated_vars = result.x
        R, t = estimated_vars[:9].reshape(3, 3), estimated_vars[9:]
        # q = R2Quaternion(R)
        q = np.zeros(4)
        pose = np.concatenate([t, q], axis=0)
        return pose
    else:
        # raise RuntimeError("P3P optimization failed.")
        return None


# def p3p(image_points, world_points, camera_intrinsics):
#     assert len(image_points) == 4 and len(world_points) == 4, "P3P requires exactly 4 point correspondences."

#     # Convert input data to PyTorch tensors
#     camera_intrinsics = camera_intrinsics.clone().float()
#     image_points = image_points.clone().float()
#     world_points = torch.concat([world_points.clone(), torch.ones(len(world_points), 1)], dim=1).float()

#     # Define the camera pose variables
#     R = torch.eye(3, requires_grad=True)
#     t = torch.zeros(3, requires_grad=True)

#     # P3P nonlinear equation system
#     def objective_function():
#         predicted_image_points = project_points(world_points, R, t, camera_intrinsics)
#         residual = predicted_image_points - image_points
#         return torch.sum(residual ** 2)

#     # Create an optimizer to minimize the objective function
#     optimizer = torch.optim.SGD([R, t], lr=0.1)  # You can adjust the learning rate

#     # Optimize the camera pose
#     for _ in range(100):  # You may need to adjust the number of optimization iterations
#         optimizer.zero_grad()
#         loss = objective_function()
#         # Rt = torch.cat([R, t.unsqueeze(1)], dim=1)
#         # projected_points = torch.mm(camera_intrinsics, torch.mm(Rt, world_points.T))  # 3x4 * 4xN = 3xN
#         # projected_points = projected_points / projected_points[2]  # Normalize by the depth
#         # projected_points = projected_points[:2].T  # Take the first two rows and transpose
#         # loss = torch.sum((projected_points - image_points) ** 2)
#         loss.backward()
#         optimizer.step()
#     pose = torch.cat([R2Quaternion(R), t], dim=0)

#     return pose


# def project_points(world_points, R, t, K):
#     # Perform the projection from world coordinates to image coordinates
#     Rt = torch.cat([R, t.unsqueeze(1)], dim=1)
#     projected_points = torch.mm(K, torch.mm(Rt, world_points.T))  # 3x4 * 4xN = 3xN
#     projected_points = projected_points / projected_points[2]  # Normalize by the depth
#     projected_points = projected_points[:2].T  # Take the first two rows and transpose
#     # print('R grad_fn', R.grad_fn)
#     # print('t grad_fn', t.grad_fn)
#     # print('K grad_fn', K.grad_fn)
#     # print('projected_points grad_fn', projected_points.grad_fn)
#     return projected_points


def R2Quaternion(R):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    q = np.zeros(4)
    q[0] = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    q[1] = (R[2, 1] - R[1, 2]) / (4.0 * q[0])
    q[2] = (R[0, 2] - R[2, 0]) / (4.0 * q[0])
    q[3] = (R[1, 0] - R[0, 1]) / (4.0 * q[0])
    return q
