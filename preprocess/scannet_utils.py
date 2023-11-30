import os
import argparse
from pprint import pprint
import pickle as pkl
import glob
import shutil
import copy

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_R

from tqdm.auto import tqdm

import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.set_printoptions(suppress=True)


class ScanNetLoader():
    def __init__(self, scannet_path: str):
        self.scannet_path = scannet_path
        self.posed_images_path = os.path.join(self.scannet_path, 'posed_images')
        self.instance_data_path = os.path.join(self.scannet_path, 'scannet_instance_data')
        self.orig_img_size = (1296, 968)
        self.img_size = (640, 480)
        label_table_path = os.path.join(self.scannet_path, 'meta_data', 'scannetv2-labels.combined.tsv')
        self.label_table = pd.read_csv(label_table_path, sep='\t', index_col='nyu40id').sort_index()
        # remove duplicated index
        self.label_table = self.label_table[~self.label_table.index.duplicated(keep='first')]

    def _read_intrinsics_file(self, scene_id: str):
        intrinsics_file = os.path.join(self.scannet_path, 'posed_images', scene_id, 'intrinsic.txt')
        intrinsics = np.loadtxt(intrinsics_file)
        return intrinsics

    def _read_cam_pose(self, scene_id: str, frame_id: int):
        cam_pose_path = os.path.join(self.posed_images_path, scene_id, f'{frame_id:0>5}.txt')
        cam_pose = np.loadtxt(cam_pose_path)
        cam_R = scipy_R.from_matrix(cam_pose[:3, :3])
        cam_t = cam_pose[:3, 3]
        return cam_t, cam_R

    def _read_instance_data(self, scene_id: str):
        align_matrix_path = os.path.join(self.instance_data_path, f'{scene_id}_axis_align_matrix.npy')
        aligned_bbox3d_path = os.path.join(self.instance_data_path, f'{scene_id}_aligned_bbox.npy')
        unaligned_bbox3d_path = os.path.join(self.instance_data_path, f'{scene_id}_unaligned_bbox.npy')
        align_matrix = np.load(align_matrix_path)
        aligned_bbox3d = np.load(aligned_bbox3d_path)
        unaligned_bbox3d = np.load(unaligned_bbox3d_path)
        aligned_bbox3d_df = pd.DataFrame(aligned_bbox3d, columns=['x', 'y', 'z', 'dx', 'dy', 'dz', 'c'])
        unaligned_bbox3d_df = pd.DataFrame(unaligned_bbox3d, columns=['x', 'y', 'z', 'dx', 'dy', 'dz', 'c'])
        aligned_bbox3d_df['c'] = aligned_bbox3d_df['c'].astype(int)
        unaligned_bbox3d_df['c'] = unaligned_bbox3d_df['c'].astype(int)
        return align_matrix, aligned_bbox3d_df, unaligned_bbox3d_df

    def _aligned_poses(self, align_matrix, cam_t, cam_R):
        aligned_cam_t = align_matrix[:3, :3] @ cam_t + align_matrix[:3, 3]
        aligned_cam_R = align_matrix[:3, :3] @ cam_R.as_matrix()
        aligned_cam_R = scipy_R.from_matrix(aligned_cam_R)
        return aligned_cam_t, aligned_cam_R

    def _get_raw_intrinsics(self, scene_id: str):
        intrinsics = self._read_intrinsics_file(scene_id)
        return intrinsics[:3, :3]

    def get_rgb_path(self, scene_id: str, frame_id: int):
        rgb_path = os.path.join(self.posed_images_path, scene_id, f'{frame_id:0>5}.jpg')
        return rgb_path

    def get_depth_path(self, scene_id: str, frame_id: int):
        depth_path = os.path.join(self.posed_images_path, scene_id, f'{frame_id:0>5}.png')
        return depth_path

    def _get_raw_rgb_img(self, scene_id: str, frame_id: int):
        rgb_path = self.get_rgb_path(scene_id, frame_id)
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        return rgb_img

    def _get_depth_img(self, scene_id: str, frame_id: int):
        depth_path = self.get_depth_path(scene_id, frame_id)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)[:, :, np.newaxis]
        return depth_img

    def get_rgb_img(self, scene_id: str, frame_id: int):
        rgb_img = self._get_raw_rgb_img(scene_id, frame_id)
        rgb_img = cv2.resize(rgb_img, self.img_size)
        return rgb_img

    def get_depth_img_in_m(self, scene_id: str, frame_id: int, inpaint=False, inpaint_radius=3, inpaint_method=cv2.INPAINT_TELEA):
        depth_img = self._get_depth_img(scene_id, frame_id).astype(np.float32)
        if inpaint:
            depth_img = cv2.inpaint(depth_img, np.uint8(depth_img == 0), inpaint_radius, inpaint_method)
        depth_img /= 1000  # convert to meters
        return depth_img

    def get_intrinsics(self, scene_id: str):
        K = self._get_raw_intrinsics(scene_id)
        # Calculate scale factors
        s = np.array(self.img_size) / np.array(self.orig_img_size)
        # Scale the intrinsic parameters
        K_new = K.copy()
        K_new[0, :] *= s[0]
        K_new[1, :] *= s[1]
        return K_new

    def get_bbox3d_and_align_matrix(self, scene_id: str):
        align_matrix, aligned_bbox3d_df, _ = self._read_instance_data(scene_id)
        return aligned_bbox3d_df, align_matrix

    def get_cam_pose(self, scene_id: str, frame_id: int, align_matrix):
        cam_t, cam_R = self._read_cam_pose(scene_id, frame_id)
        aligned_cam_t, aligned_cam_R = self._aligned_poses(align_matrix, cam_t, cam_R)
        return aligned_cam_t, aligned_cam_R

    def get_in_view_masks(self, bbox3d_df, cam_t, cam_R, intrinsics):
        masks = np.ones(len(bbox3d_df), dtype=bool)
        for i, obj in enumerate(bbox3d_df.values):
            center = obj[:3]
            size = obj[3:6]
            # filter out out-of-view objects
            center_cam_coords = transform_to_camera_coords(center.reshape(1, 3), cam_t, cam_R)
            if center_cam_coords[2] <= 0:
                masks[i] = False
            corners_3d = get_3d_bbox_corners(center, size)
            corners_cam_coords = transform_to_camera_coords(corners_3d, cam_t, cam_R)
            corners_2d = project_to_2d(corners_cam_coords, intrinsics)
            corners_2d = np.clip(corners_2d, 0, np.array(self.img_size) - 1)
            bbox2d = compute_2d_bbox(corners_2d)
            bbox_size = bbox2d[2].astype(int) * bbox2d[3].astype(int)
            if bbox_size < 1 or bbox_size >= (self.img_size[0] - 1) * (self.img_size[1] - 1):
                masks[i] = False
        return masks

    def get_bbox2d(self, scene_id: str, frame_id: int):
        intrinsics = self.get_intrinsics(scene_id)
        aligned_bbox3d_df, align_matrix = self.get_bbox3d_and_align_matrix(scene_id)
        cam_t, cam_R = self.get_cam_pose(scene_id, frame_id, align_matrix)
        in_view_masks = self.get_in_view_masks(aligned_bbox3d_df, cam_t, cam_R, intrinsics)
        aligned_bbox3d_df = aligned_bbox3d_df[in_view_masks]
        bbox2ds = []
        for obj in aligned_bbox3d_df.values:
            center = obj[:3]
            size = obj[3:6]
            label = obj[6]
            # project 3D bounding boxes to 2D
            corners_3d = get_3d_bbox_corners(center, size)
            corners_cam_coords = transform_to_camera_coords(corners_3d, cam_t, cam_R)
            corners_2d = project_to_2d(corners_cam_coords, intrinsics)
            corners_2d = np.clip(corners_2d, 0, np.array(self.img_size) - 1)
            bbox2d = compute_2d_bbox(corners_2d)
            cx = bbox2d[0] + bbox2d[2] / 2
            cy = bbox2d[1] + bbox2d[3] / 2
            bbox2ds.append(bbox2d + [cx, cy, label])
        bbox2d_df = pd.DataFrame(bbox2ds, columns=['x0', 'y0', 'w', 'h', 'cx', 'cy', 'c'])
        return bbox2d_df

    def get_label_names(self, label_ids, type):
        return self.label_table.loc[label_ids, type].values

    def viz_2d_scene(self, scene_id: str, frame_id: int, label_type='nyu40class'):
        rgb_img = self.get_rgb_img(scene_id, frame_id)
        bbox2d_df = self.get_bbox2d(scene_id, frame_id)
        fig, ax = plt.subplots()
        ax.imshow(rgb_img)
        for i, bbox2d in bbox2d_df.iterrows():
            label_id = bbox2d['c']
            label_name = self.get_label_names([label_id], label_type)[0]
            text = f'{label_id}: {label_name}'
            rect = patches.Rectangle((bbox2d['x0'], bbox2d['y0']), bbox2d['w'], bbox2d['h'],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.text(bbox2d['x0'], bbox2d['y0'] + bbox2d['h'], text, color='r')
            ax.add_patch(rect)
        plt.show()
        return fig, ax

    def viz_3d_scene(self, scene_id: str, frame_id: int):
        aligned_bbox3d_df, align_matrix = self.get_bbox3d_and_align_matrix(scene_id)
        cam_t, cam_R = self.get_cam_pose(scene_id, frame_id, align_matrix)
        # Convert quaternion to rotation matrix
        rot_matrix = cam_R.as_matrix()
        # Define a color map for categories
        num_categories = int(np.max(aligned_bbox3d_df.values[:, -1])) + 1
        color_map = create_color_map(num_categories)
        # Create an Open3D visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # Add camera frame to the visualization
        camera_frame = create_camera_frame(cam_t, rot_matrix)
        vis.add_geometry(camera_frame)
        # Add objects as cuboids
        for obj in aligned_bbox3d_df.values:
            position = obj[:3]
            size = obj[3:6]
            category = int(obj[6])
            color = color_map[category]
            cuboid, cuboid_frame = create_cuboid(position, size, color=color)
            vis.add_geometry(cuboid)
            vis.add_geometry(cuboid_frame)
        # Run the visualizer
        vis.run()
        vis.destroy_window()


# ===== 2D projection =====


def get_fov(intrinsics, img_size):
    """
    return: hfov, vfov
    """
    hfov = 2 * np.arctan(img_size[0] / (2 * intrinsics[0, 0]))
    vfov = 2 * np.arctan(img_size[1] / (2 * intrinsics[1, 1]))
    return (hfov, vfov)


def get_3d_bbox_corners(center, size):
    """ Compute 8 corners of a 3D bounding box based on its center and size. """
    dx, dy, dz = size / 2
    corners = np.array([[dx, dy, dz], [dx, dy, -dz], [dx, -dy, dz], [dx, -dy, -dz],
                        [-dx, dy, dz], [-dx, dy, -dz], [-dx, -dy, dz], [-dx, -dy, -dz]])
    return corners + center


def transform_to_camera_coords(corners, cam_t, cam_R):
    """ Transform 3D points from world coordinates to camera coordinates. """
    rotation_matrix = cam_R.inv().as_matrix()
    return (rotation_matrix @ (corners.T - cam_t[:, np.newaxis]))


def project_to_2d(points_3d, intrinsics):
    points_2d = intrinsics @ points_3d
    points_2d = points_2d[:2, :] / points_2d[2, :]
    return points_2d.T


def compute_2d_bbox(projected_corners):
    """ Compute the 2D bounding box from projected 2D corners. """
    min_x, min_y = np.min(projected_corners, axis=0)
    max_x, max_y = np.max(projected_corners, axis=0)
    return [min_x, min_y, max_x - min_x, max_y - min_y]

# ===== 3D visualization =====


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


def create_cuboid(center, sizes, color=[1, 0, 0, 0.5]):
    # Create a cuboid
    cuboid_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cuboid_frame.translate(center, relative=False)
    cuboid = o3d.geometry.TriangleMesh.create_box(width=sizes[0], height=sizes[1], depth=sizes[2])
    cuboid.translate(center - np.array(sizes) / 2)
    cuboid.paint_uniform_color(color[:3])  # Use only RGB components, as alpha is not supported here
    return cuboid, cuboid_frame
