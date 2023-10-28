import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
from torch_geometric.nn import GATv2Conv
import torch_geometric.nn as pygnn
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
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

        return {
            'matches0': matches['matches0'],
            'matches1': matches['matches1'],
            'matching_scores0': matches['matching_scores0'],
            'matching_scores1': matches['matching_scores1'],
            'scores': scores,
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
