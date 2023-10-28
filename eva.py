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
        pred_e1i = np.array([idx for idx, v in enumerate(pred_dict['matches0']) if v != -1])
        pred_e2i = np.array([v.item() for idx, v in enumerate(pred_dict['matches0']) if v != -1])
        if len(pred_e1i) == 0:
            fitness, t_rmse, r_err = None, None, None
        else:
            bbox3d_t1 = data_dict['node_bbox3d'][0][:data_dict['n1'].item(), :3].cpu().numpy()
            bbox3d_t2 = data_dict['node_bbox3d'][0][data_dict['n1'].item():, :3].cpu().numpy()
            pred_R, pred_t, reg = icp_on_translation(bbox3d_t1, bbox3d_t2, pred_e1i, pred_e2i)
            pred_pose = np.concatenate([pred_t, pred_R.as_quat()])
            # log pose error
            t_rmse, r_err = compute_pose_error(pred_pose, gt_pose)
            fitness = reg.fitness
        metrics['icp_fitness'] = fitness
        # metrics['icp_inlier_rmse'] = reg.inlier_rmse
        metrics['icp_t_rmse'] = t_rmse
        metrics['icp_r_err'] = r_err

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
    if type(pred_q) is not scipy_R:
        pred_q = scipy_R(pred_q)
    if type(gt_q) is not scipy_R:
        gt_q = scipy_R(gt_q)

    # rotation from pred to gt
    angle = pred_q.inv() * gt_q
    theta = angle.magnitude()
    if deg:
        theta = np.rad2deg(theta)
    return theta


def icp_on_translation(bbox3d_1, bbox3d_2, e1i, e2i, radius_normal=5, radius_feature=10, icp_threshold=1):
    assert len(bbox3d_1.shape) == 2 and len(bbox3d_2.shape) == 2
    assert bbox3d_1.shape[1] == 3 and bbox3d_2.shape[1] == 3
    g1_bbox3d = bbox3d_1[e1i]
    g2_bbox3d = bbox3d_2[e2i]
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(g1_bbox3d[:, :3])
    pcd2.points = o3d.utility.Vector3dVector(g2_bbox3d[:, :3])
    # pcd1.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (len(g1_bbox3d), 1)))
    # pcd2.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 0, 1]]), (len(g2_bbox3d), 1)))

    pcd1, pcd1_fpfh = preprocess_pcd(pcd1, radius_normal=radius_normal, radius_feature=radius_feature)
    pcd2, pcd2_fpfh = preprocess_pcd(pcd2, radius_normal=radius_normal, radius_feature=radius_feature)
    reg_glb = execute_global_registration(pcd1, pcd2, pcd1_fpfh, pcd2_fpfh, distance_threshold=0.05)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, icp_threshold, reg_glb.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    R = scipy_R.from_matrix(np.array(reg_p2p.transformation[:3, :3]))
    t = np.array(reg_p2p.transformation[:3, 3])

    return R, t, reg_p2p


def preprocess_pcd(pcd, radius_normal, radius_feature):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def execute_global_registration(src, tgt, src_fpfh, tgt_fpfh, distance_threshold):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, src_fpfh, tgt_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
