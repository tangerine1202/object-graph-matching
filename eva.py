import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as scipy_R


def compute_eval(pred_dict, data_dict, eval_type):
    metrics = {}

    # confusion_matrix
    conf_matrix = compute_confusion_matrix(pred_dict, data_dict)
    metrics['pre'] = conf_matrix['pre']
    metrics['rec'] = conf_matrix['rec']
    metrics['f1'] = conf_matrix['f1']

    gt_pose = data_dict['qry_camera_pose'][0].cpu().numpy()
    if eval_type == '3d':
        if 'pose' in pred_dict and pred_dict['pose'] is not None:
            pred_pose = pred_dict['pose']
            t_rmse, r_err = compute_pose_error(pred_pose, gt_pose)
            metrics['t_rmse'] = t_rmse
            metrics['r_err'] = r_err

    # upcast metric to float to address None value
    for k in metrics:
        metrics[k] = np.float64(metrics[k])

    return metrics


def compute_corr(pred_dict, data_dict):
    e1i = data_dict['e1i'].squeeze(0).cpu().numpy()
    e2i = data_dict['e2i'].squeeze(0).cpu().numpy()
    matches0 = pred_dict['matches0'].squeeze(0).cpu().numpy()

    gt_corr = {e1i[i]: e2i[i] for i in range(len(e1i))}
    corr = {i: matches0[i] for i in range(len(matches0)) if matches0[i] != -1}
    tp_corr = {i: corr[i] for i in corr if i in gt_corr and gt_corr[i] == corr[i]}
    fp_corr = {i: corr[i] for i in corr if i in gt_corr and gt_corr[i] != corr[i]}
    fn_corr = {i: gt_corr[i] for i in gt_corr if i not in corr}
    return tp_corr, fp_corr, fn_corr, gt_corr


def compute_confusion_matrix(pred_dict, data_dict):
    tp_corr, fp_corr, fn_corr, gt_corr = compute_corr(pred_dict, data_dict)

    tp = len(tp_corr)
    fp = len(fp_corr)
    fn = len(fn_corr)
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'pre': tp / (tp + fp) if tp + fp > 0 else 0,
        'rec': tp / (tp + fn) if tp + fn > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0,
    }


def compute_pose_error(pred_pose, gt_pose, deg=True):
    pred_t = pred_pose[:3]
    pred_q = pred_pose[3:]
    gt_t = gt_pose[:3]
    gt_q = gt_pose[3:]
    t_rmse = compute_translation_error(pred_t, gt_t)
    r_err = compute_rotation_error(pred_q, gt_q, deg=deg)
    return t_rmse, r_err


def compute_translation_error(pred_t, gt_t):
    axis = 1 if len(pred_t.shape) == 2 else 0
    rmse = np.linalg.norm(pred_t - gt_t, axis=axis).mean()
    return rmse


def compute_rotation_error(pred_q, gt_q, deg=True):
    if not isinstance(pred_q, scipy_R):
        pred_q = scipy_R(pred_q)
    if not isinstance(gt_q, scipy_R):
        gt_q = scipy_R(gt_q)

    # rotation from pred to gt
    angle = pred_q.inv() * gt_q
    theta = angle.magnitude()
    if deg:
        theta = np.rad2deg(theta)
    return theta
