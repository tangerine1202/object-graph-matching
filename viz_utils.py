import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from solo2graph import MapGraph, QueryGraph

from eva import compute_corr


def read_img(step, data_dir, type='sem_seg'):
    return cv2.cvtColor(cv2.imread(f'{data_dir}/{type}/step{step}.png'), cv2.COLOR_BGR2RGB)


def read_graph(step, data_dir):
    return pkl.load(open(f'{data_dir}/graph/step{step}.pkl', 'rb'))


def draw_bbox(img, bbox_xywh, color=(255, 0, 0), thickness=2):
    bbox_xywh = bbox_xywh.astype(np.int32)
    return cv2.rectangle(img, (bbox_xywh[0], bbox_xywh[1]), (bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]), color, thickness)


def viz_corr(
    pred_dict, data_dict,
    data_dir,
    img_type='inst_seg',
    TP_CORR_COLOR=(0, 255, 0),
    FP_CORR_COLOR=(255, 200, 0),
    FN_CORR_COLOR=(0, 200, 255),
    NON_CORR_COLOR=(200, 200, 200),
):

    def concat_imgs(imgs: list, axis=1, pad_px: int = 0, pad_color: list = (255, 255, 255)):
        if pad_px > 0:
            pad_shape = (pad_px, imgs[0].shape[1], imgs[0].shape[2]) if axis == 0 else (
                imgs[0].shape[0], pad_px, imgs[0].shape[2])
            padding = np.full(pad_shape, pad_color, dtype=np.uint8)
            return np.concatenate([np.concatenate([x, padding], axis=axis) for x in imgs[:-1]] + [imgs[-1]], axis=axis)
        else:
            return np.concatenate(imgs, axis=axis)

    def draw_bbox(img, bbox_xywh, color=(255, 0, 0), thickness=2):
        bbox_xywh = bbox_xywh.astype(np.int32)
        return cv2.rectangle(img, (bbox_xywh[0], bbox_xywh[1]), (bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]), color, thickness)

    tp_corr, fp_corr, fn_corr, gt_corr = compute_corr(pred_dict, data_dict)
    qry_step = data_dict['qry_step'].item()
    amp_step = data_dict['map_step'].item()
    qry_img = read_img(qry_step, data_dir, img_type)
    map_img = read_img(amp_step, data_dir, img_type)
    pad_px = 10
    img = concat_imgs([qry_img, map_img], pad_px=pad_px)

    qry_g = read_graph(qry_step, data_dir)
    map_g = read_graph(amp_step, data_dir)
    qry_bbox = pd.DataFrame({k: qry_g.node_feat[k].squeeze(1) for k in ['bbox_cx', 'bbox_cy', 'bbox_w', 'bbox_h']})
    map_bbox = pd.DataFrame({k: map_g.node_feat[k].squeeze(1) for k in ['bbox_cx', 'bbox_cy', 'bbox_w', 'bbox_h']})
    qry_bbox['bbox_x'] = qry_bbox['bbox_cx'] - qry_bbox['bbox_w'] / 2
    qry_bbox['bbox_y'] = qry_bbox['bbox_cy'] - qry_bbox['bbox_h'] / 2
    map_bbox['bbox_x'] = map_bbox['bbox_cx'] - map_bbox['bbox_w'] / 2
    map_bbox['bbox_y'] = map_bbox['bbox_cy'] - map_bbox['bbox_h'] / 2

    for i in range(len(qry_g)):
        bbox1 = qry_bbox[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].iloc[i]
        if i not in tp_corr and i not in fp_corr:  # and i not in fn_corr:
            img = draw_bbox(img, bbox1, color=NON_CORR_COLOR)
    for i in range(len(map_g)):
        bbox2 = map_bbox[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].iloc[i]
        bbox2['bbox_x'] += qry_img.shape[1] + pad_px
        if i not in tp_corr.values() and i not in fp_corr.values():  # and i not in fn_corr.values():
            img = draw_bbox(img, bbox2, color=NON_CORR_COLOR)

    for i, j in fn_corr.items():
        bbox1 = qry_bbox[['bbox_cx', 'bbox_cy', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].iloc[i].astype(int)
        bbox2 = map_bbox[['bbox_cx', 'bbox_cy', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].iloc[j].astype(int)
        bbox2[['bbox_cx', 'bbox_x']] += qry_img.shape[1] + pad_px
        img = cv2.line(img,
                       (bbox1['bbox_cx'], bbox1['bbox_cy']),
                       (bbox2['bbox_cx'], bbox2['bbox_cy']),
                       FN_CORR_COLOR,
                       2)
        draw_bbox(img, bbox1[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']], color=FN_CORR_COLOR)
        draw_bbox(img, bbox2[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']], color=FN_CORR_COLOR)

    for i, j in tp_corr.items():
        bbox1 = qry_bbox[['bbox_cx', 'bbox_cy', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].iloc[i].astype(int)
        bbox2 = map_bbox[['bbox_cx', 'bbox_cy', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].iloc[j].astype(int)
        bbox2[['bbox_cx', 'bbox_x']] += qry_img.shape[1] + pad_px
        img = cv2.line(img,
                       (bbox1['bbox_cx'], bbox1['bbox_cy']),
                       (bbox2['bbox_cx'], bbox2['bbox_cy']),
                       TP_CORR_COLOR,
                       2)
        draw_bbox(img, bbox1[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']], color=TP_CORR_COLOR)
        draw_bbox(img, bbox2[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']], color=TP_CORR_COLOR)

    for i, j in fp_corr.items():
        bbox1 = qry_bbox[['bbox_cx', 'bbox_cy', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].iloc[i].astype(int)
        bbox2 = map_bbox[['bbox_cx', 'bbox_cy', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].iloc[j].astype(int)
        bbox2[['bbox_cx', 'bbox_x']] += qry_img.shape[1] + pad_px
        img = cv2.line(img,
                       (bbox1['bbox_cx'], bbox1['bbox_cy']),
                       (bbox2['bbox_cx'], bbox2['bbox_cy']),
                       FP_CORR_COLOR,
                       2)
        draw_bbox(img, bbox1[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']], color=FP_CORR_COLOR)
        draw_bbox(img, bbox2[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']], color=FP_CORR_COLOR)

    return img, tp_corr, fp_corr, fn_corr, gt_corr
