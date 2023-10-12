import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCriterion(torch.nn.Module):
    def __init__(self, device, alpha=0.5):
        super(CustomCriterion, self).__init__()
        self.device = device
        self.alpha = alpha
        # self.icl_crit = ICLCriterion(device, temperature=0.1)
        # self.pose_crit = PoseNetCriterion(learn_beta=True)
        self.glue_crit = SuperGlueCriterion()

    def forward(self, pred_dict, data_dict):
        glue = self.glue_crit(pred_dict, data_dict)
        # icl = self.icl_criterion(pred_dict['joint_emb'], data_dict)
        # pose1 = self.pose_crit(pred_dict['pose1'], data_dict['g1_camera_pose'])
        # pose2 = self.pose_crit(pred_dict['pose2'], data_dict['g2_camera_pose'])
        # pose = (pose1 + pose2) / 2
        # loss = self.alpha * glue + (1-self.alpha) * pose
        loss = glue
        return loss


class SuperGlueCriterion(torch.nn.Module):
    # ref: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
    def __init__(self):
        super(SuperGlueCriterion, self).__init__()

    def forward(self, pred_dict, data_dict):
        all_matches = torch.concat([data_dict['e1i'], data_dict['e2i'] -
                                   data_dict['g1_node_count'].item()], dim=0).transpose(0, 1).unsqueeze(0)
        scores = pred_dict['scores']

        # check if indexed correctly
        loss = []
        for i in range(len(all_matches[0])):
            x = all_matches[0][i][0]
            y = all_matches[0][i][1]
            loss.append(-torch.log(scores[0][x][y].exp()))  # check batch size == 1 ?
        # for p0 in unmatched0:
        #     loss += -torch.log(scores[0][p0][-1])
        # for p1 in unmatched1:
        #     loss += -torch.log(scores[0][-1][p1])
        loss_mean = torch.mean(torch.stack(loss))
        loss_mean = torch.reshape(loss_mean, (1, -1))
        # FIXME: check if only the loss is used in backprop
        return loss_mean[0]


class ICLCriterion(nn.Module):
    # ref: SGAligner repo
    def __init__(self, device, temperature=0.1, alpha=0.5):
        super(ICLCriterion, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.device = device

    def forward(self, emb, data_dict):
        emb = F.normalize(emb, dim=1)
        e1i = emb[data_dict['e1i']]
        e2i = emb[data_dict['e2i']]
        e1j = emb[data_dict['e1j']]
        e2j = emb[data_dict['e2j']]

        qm_e1i_e2i = calculate_prob_dist(e1i, e2i, e1j, e2j, self.temp)
        qm_e2i_e1i = calculate_prob_dist(e2i, e1i, e2j, e1j, self.temp)

        lossA = qm_e1i_e2i
        lossB = qm_e2i_e1i

        loss = self.alpha * lossA + (1 - self.alpha) * lossB
        loss = -torch.log(loss).mean()
        return loss


class IALCriterion(nn.Module):
    def __init__(self, device, temperature=1, alpha=0.5):
        super(IALCriterion, self).__init__()
        self.device = device
        self.temp = temperature
        self.alpha = alpha
        self.zoom = 0.1

    def forward(self, src_emb, ref_emb, data_dict):
        '''
        src_emb : joint embedding
        ref_emb : modal embedding
        '''
        src_emb = F.normalize(src_emb, dim=1)
        ref_emb = F.normalize(ref_emb, dim=1)

        o_e1i = src_emb[data_dict['e1i']]
        o_e2i = src_emb[data_dict['e2i']]
        o_e1j = src_emb[data_dict['e1j']]
        o_e2j = src_emb[data_dict['e2j']]

        qo_e1i_e2i = calculate_prob_dist(o_e1i, o_e2i, o_e1j, o_e2j, self.temp)
        qo_e2i_e1i = calculate_prob_dist(o_e2i, o_e1i, o_e2j, o_e1j, self.temp)

        m_e1i = ref_emb[data_dict['e1i']]
        m_e2i = ref_emb[data_dict['e2i']]
        m_e1j = ref_emb[data_dict['e1j']]
        m_e2j = ref_emb[data_dict['e2j']]

        qm_e1i_e2i = calculate_prob_dist(m_e1i, m_e2i, m_e1j, m_e2j, self.temp)
        qm_e2i_e1i = calculate_prob_dist(m_e2i, m_e1i, m_e2j, m_e1j, self.temp)

        klLoss = nn.KLDivLoss(size_average=False, reduction="sum", log_target=True)
        loss_a = klLoss(qm_e1i_e2i.log(), qo_e1i_e2i).mean()
        loss_b = klLoss(qm_e2i_e1i.log(), qo_e2i_e1i).mean()

        loss = self.zoom * (self.alpha * loss_a + (1 - self.alpha) * loss_b)
        return loss


def calculate_prob_dist(e1i, e2i, e1j, e2j, temp):
    # FIXME: remove the squeeze(0) when batch_size>1
    e1i = e1i.squeeze(0)
    e2i = e2i.squeeze(0)
    e1j = e1j.squeeze(0)
    e2j = e2j.squeeze(0)

    deltaM_e1i_e2i = torch.exp(torch.matmul(e1i, torch.transpose(e2i, 0, 1)) / temp)
    deltaM_e1i_e1j = torch.exp(torch.matmul(e1i, torch.transpose(e1j, 0, 1)) / temp)
    deltaM_e1i_e2j = torch.exp(torch.matmul(e1i, torch.transpose(e2j, 0, 1)) / temp)

    deltaM_e1i_e2i_e1j = deltaM_e1i_e2i / (deltaM_e1i_e1j.sum() + 1e-9)
    deltaM_e1i_e2i_e2j = deltaM_e1i_e2i / (deltaM_e1i_e2j.sum() + 1e-9)
    q_e1i_e2i_inverse = 1.0 + 1.0 / (deltaM_e1i_e2i_e1j + 1e-9) + 1.0 / (deltaM_e1i_e2i_e2j + 1e-9)
    q_e1i_e2i = 1.0 / (q_e1i_e2i_inverse + 1e-9)

    return q_e1i_e2i


class PoseNetCriterion(torch.nn.Module):
    # ref: https://capsulesbot.com/blog/2018/08/24/apolloscape-posenet-pytorch.html
    def __init__(self, beta=512.0, learn_beta=False, sx=0.0, sq=-3.0):
        super(PoseNetCriterion, self).__init__()
        self.loss_fn = torch.nn.L1Loss()
        self.learn_beta = learn_beta
        if not learn_beta:
            self.beta = beta
        else:
            self.beta = 1.0
        self.sx = torch.nn.Parameter(torch.Tensor([sx]), requires_grad=learn_beta)
        self.sq = torch.nn.Parameter(torch.Tensor([sq]), requires_grad=learn_beta)

    def forward(self, x, y):
        # Translation loss
        loss = torch.exp(-self.sx) * self.loss_fn(x[:, :3], y[:, :3])
        # Rotation loss
        loss += torch.exp(-self.sq) * self.beta * self.loss_fn(x[:, 3:], y[:, 3:]) + self.sq
        return loss
