import torch
import numpy as np
import quaternion


def compute_eval(pred_dict, data_dict):
    metrics = {}

    # confusion_matrix
    conf_matrix = compute_confusion_matrix(pred_dict, data_dict)
    metrics['acc'] = conf_matrix['acc']
    metrics['pre'] = conf_matrix['pre']
    metrics['rec'] = conf_matrix['rec']
    metrics['f1'] = conf_matrix['f1']

    # hits@k
    # all_k = [1, 3, 5]
    # for k in all_k:
    #     correct, total = compute_hits_k(pred_dict, data_dict, k)
    #     metrics[f'hits@{k}'] = correct / total

    # translation error
    if 'pose0' in pred_dict and 'pose1' in pred_dict:
        pose1 = pred_dict['pose0']
        pose2 = pred_dict['pose1']
        cam_pose1 = data_dict['g1_camera_pose']
        cam_pose2 = data_dict['g2_camera_pose']
        mse1, mae1 = computer_translation_error(pose1[:, :3], cam_pose1[:, :3])
        mse2, mae2 = computer_translation_error(pose2[:, :3], cam_pose2[:, :3])
        metrics['pos0_mse'] = mse1
        metrics['pos0_mae'] = mae1
        metrics['pos1_mse'] = mse2
        metrics['pos1_mae'] = mae2

    # # rotation error
    # rot1_errors = []
    # rot2_errors = []
    # for i in range(len(pose1)):
    #     rot1_errors.append(compute_rotation_error_in_degree(pose1[i, 3:], cam_pose1[i, 3:]))
    #     rot2_errors.append(compute_rotation_error_in_degree(pose2[i, 3:], cam_pose2[i, 3:]))
    # metrics['rot1_error'] = np.mean(rot1_errors)
    # metrics['rot2_error'] = np.mean(rot2_errors)

    return metrics


def compute_hits_k(pred_dict, data_dict, k=1):
    scores = pred_dict['scores'].squeeze(0)
    e1i_idxs = data_dict['e1i'].squeeze(0)
    e2i_idxs = data_dict['e2i'].squeeze(0) - data_dict['g1_node_count']

    max_score = torch.max(scores)
    rank_list = torch.argsort(1 - scores / max_score, dim=1)
    correct, total = 0, e1i_idxs.shape[0]
    for idx, e1i_idx in enumerate(e1i_idxs):
        e1_idx_rank_list = list(rank_list[e1i_idx])
        e1_idx_rank_list_k = e1_idx_rank_list[:k]
        if e2i_idxs[idx] in e1_idx_rank_list_k:
            correct += 1
    return correct, total


def compute_confusion_matrix(pred_dict, data_dict):
    e1i = data_dict['e1i'].squeeze(0)
    e1j = data_dict['e1j'].squeeze(0)
    e2i = data_dict['e2i'].squeeze(0) - data_dict['g1_node_count']
    e2j = data_dict['e2j'].squeeze(0) - data_dict['g1_node_count']
    e1_gt_matches = [[e1i[i], e2i[i]] for i in range(len(e1i))] \
        + [[e1j[i].item(), -1] for i in range(len(e1j))]
    e2_gt_matches = [[e2i[i], e1i[i]] for i in range(len(e2i))] \
        + [[e2j[i].item(), -1] for i in range(len(e2j))]

    TP, FP, FN, TN = 0, 0, 0, 0
    for match in e1_gt_matches:
        if pred_dict['matches0'][match[0]].item() == match[1]:
            if match[1] == -1:
                TN += 1
            else:
                TP += 1
        else:
            if match[1] == -1:
                FP += 1
            else:
                FN += 1
    for match in e2_gt_matches:
        if pred_dict['matches1'][match[0]].item() == match[1]:
            if match[1] == -1:
                TN += 1
            else:
                TP += 1
        else:
            if match[1] == -1:
                FP += 1
            else:
                FN += 1
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'acc': (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0,
        'pre': TP / (TP + FP) if TP + FP > 0 else 0,
        'rec': TP / (TP + FN) if TP + FN > 0 else 0,
        'f1': 2 * TP / (2 * TP + FP + FN) if TP + FP + FN > 0 else 0,
    }


def computer_translation_error(pred_pos, gt_pos):
    mse = torch.norm(pred_pos - gt_pos, dim=1, p=2).mean()
    mae = torch.norm(pred_pos - gt_pos, dim=1, p=1).mean()
    return mse, mae


def compute_rotation_error_in_degree(q1, q2):
    # maybe consider use the Eular angel difference: https://github.com/sayands/sgaligner/blob/c4ad97d0aeaabb2ea2e415bb834c1618601d252d/utils/registration.py#L48
    # # Normalize both quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    q1 = np.quaternion(*q1)
    q2 = np.quaternion(*q2)
    # Compute the quaternion product
    rotation_error_quaternion = q1 * q2.conjugate()
    # Extract the angle of rotation from the resulting quaternion
    angle_radians = 2 * np.arccos(rotation_error_quaternion.real)
    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees
