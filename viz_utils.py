import pickle as pkl
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_R
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import open3d as o3d
import plotly.graph_objects as go
from preprocess import QueryGraph, MapGraph

from eva import compute_corr


def read_img(step, data_dir, type='sem_seg'):
    return cv2.cvtColor(cv2.imread(f'{data_dir}/{type}/step{step}.png'), cv2.COLOR_BGR2RGB)


def read_graph(step, data_dir):
    return pkl.load(open(f'{data_dir}/graph/step{step}.pkl', 'rb'))


def draw_bbox_xywh(img, bbox_xywh, color=(255, 0, 0), thickness=2):
    bbox_xywh = bbox_xywh.astype(np.int32)
    return cv2.rectangle(img, (bbox_xywh[0], bbox_xywh[1]), (bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]), color, thickness)


def draw_bbox_xyxy(img, bbox_xyxy, color=(255, 0, 0), thickness=2):
    bbox_xyxy = bbox_xyxy.astype(np.int32)
    return cv2.rectangle(img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), color, thickness)


def concat_imgs(imgs: list, axis=1, pad_px: int = 0, pad_color: list = (255, 255, 255)):
    if pad_px > 0:
        pad_shape = (pad_px, imgs[0].shape[1], imgs[0].shape[2]) if axis == 0 else (
            imgs[0].shape[0], pad_px, imgs[0].shape[2])
        padding = np.full(pad_shape, pad_color, dtype=np.uint8)
        return np.concatenate([np.concatenate([x, padding], axis=axis) for x in imgs[:-1]] + [imgs[-1]], axis=axis)
    else:
        return np.concatenate(imgs, axis=axis)

# def viz_bbox_x0y0wh_with_plotly(bbox_x0y0wh, labels, fig_size=(500, 500)):


def viz_bbox_x0y0wh_with_plt(bbox_x0y0wh, labels, fig_size=None):
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    for bbox, label in zip(bbox_x0y0wh, labels):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1], label, color='r')
    plt.show()
    return plt, ax


def viz_corr_on_bbox_cxcy(src_img, ref_img, bbox_src_cxcy, bbox_ref_cxcy, corr):
    assert corr.shape[0] == 2, f'corr shape (2, n), but got {corr.shape}'
    assert max(corr[0]) < len(
        bbox_src_cxcy), f'corr[0] should be less than len(bbox_src) {len(bbox_src_cxcy)}, but got {max(corr[0])}'
    assert max(corr[1]) < len(
        bbox_ref_cxcy), f'corr[1] should be less than len(bbox_ref) {len(bbox_ref_cxcy)}, but got {max(corr[1])}'

    imgs = concat_imgs([src_img, ref_img])

    for src_idx, ref_idx in corr.T:
        src_pt = bbox_src_cxcy[src_idx]
        ref_pt = bbox_ref_cxcy[ref_idx]
        # shift
        src_pt[0] += src_img.shape[1]
        cv2.line(imgs, tuple(src_pt.astype(int)), tuple(ref_pt.astype(int)), (255, 255, 0), 1)
        cv2.circle(imgs, tuple(src_pt.astype(int)), 2, (0, 255, 0), 2)
        cv2.circle(imgs, tuple(ref_pt.astype(int)), 2, (255, 0, 0), 2)
    return imgs

# Function to create vertices of a 3D bounding box


def get_bbox3d_corners(center, size, rotation):
    # Local coordinates of a 3D bounding box
    x, y, z = size
    vertices = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]])
    # Apply rotation and translation
    vertices = rotation.apply(vertices) + center
    return vertices


# Function to create edges of a 3D bounding box
def create_bbox3d_edges(corners):
    edges = [
        [corners[i], corners[j]]
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]
    ]
    return edges


def viz_bbox3d_with_plotly(bbox3d, labels, fig_size=(500, 500)):
    # Initialize plotly figure
    fig = go.Figure()

    # Plot each bounding box
    for item, label_name in zip(bbox3d, labels):
        tx, ty, tz, qx, qy, qz, qw, sx, sy, sz = item
        center = np.array([tx, ty, tz])
        size = np.array([sx, sy, sz])
        rotation = scipy_R.from_quat([qx, qy, qz, qw])
        vertices = get_bbox3d_corners(center, size, rotation)
        edges = create_bbox3d_edges(vertices)

        # Add edges to figure
        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[edge[0][0], edge[1][0]],
                    y=[edge[0][1], edge[1][1]],
                    z=[edge[0][2], edge[1][2]],
                    mode='lines',
                    line=dict(color='blue'),
                    name=label_name,
                    showlegend=False
                )
            )

    # Set figure layout
    fig.update_layout(
        title='3D Bounding Boxes',
        width=fig_size[0],
        height=fig_size[1],
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    return fig


def viz_bbox3d_and_cam_with_o3d(bbox3d, cam_pose, labels=None):
    cam_t, cam_R = cam_pose[:3], scipy_R.from_quat(cam_pose[3:])
    # Convert quaternion to rotation matrix
    # Define a color map for categories
    # num_labels = len(np.unique(labels)) + 1
    # color_map = create_color_map(num_labels)
    # Create an Open3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Add camera frame to the visualization
    camera_frame = create_camera_frame(cam_t, cam_R.as_matrix())
    vis.add_geometry(camera_frame)
    # Add objects as cuboids
    for idx, obj in enumerate(bbox3d):
        position = obj[:3]
        rotation = scipy_R.from_quat(obj[3:7])
        size = obj[7:]
        # color = color_map[labels[idx]] if labels is not None else None
        cuboid, cuboid_frame = create_cuboid(position, rotation, size)
        vis.add_geometry(cuboid)
        vis.add_geometry(cuboid_frame)
    # Run the visualizer
    vis.run()
    vis.destroy_window()


def create_color_map(num_categories, alpha=0.5):
    # Use a Matplotlib colormap
    cmap = plt.cm.get_cmap('viridis', num_categories)
    return {category: list(cmap(category))[:3] + [alpha] for category in range(num_categories)}


def create_camera_frame(camera_pos, camera_rot, size=0.5):
    # Create a coordinate frame representing the camera
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    camera_frame.translate(camera_pos, relative=False)
    camera_frame.rotate(camera_rot, center=camera_pos)
    return camera_frame


def create_cuboid(center, rotation, sizes, color=[1, 0, 0]):
    # Create a cuboid
    cuboid_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cuboid_frame.translate(center, relative=False)
    cuboid = o3d.geometry.TriangleMesh.create_box(width=sizes[0] * 2, height=sizes[1] * 2, depth=sizes[2] * 2)
    cuboid.translate(center - np.array(sizes))
    cuboid.rotate(rotation.as_matrix(), center=center)
    cuboid.paint_uniform_color(color)
    return cuboid, cuboid_frame


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
