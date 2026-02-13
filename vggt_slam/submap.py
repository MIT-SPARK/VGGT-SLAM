import re
import os
import cv2
import torch
import numpy as np
import scipy
import open3d as o3d
from vggt_slam.slam_utils import decompose_camera

class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id
        self.R_world_map = None
        self.poses = None
        self.frames = None
        self.proj_mats = None
        self.retrieval_vectors = None
        self.colors = None # (S, H, W, 3)
        self.conf = None # (S, H, W)
        self.conf_masks = None # (S, H, W)
        self.conf_threshold = None
        self.pointclouds = None # (S, H, W, 3)
        self.voxelized_points = None
        self.last_non_loop_frame_index = None
        self.frame_ids = None
        self.is_lc_submap = False
        self.img_names = []
        self.semantic_vectors = []
    
    def set_lc_status(self, is_lc_submap):
        self.is_lc_submap = is_lc_submap
    
    def add_all_poses(self, poses):
        self.poses = poses

    def add_all_points(self, points, colors, conf, conf_threshold_percentile, intrinsics_inv):
        self.pointclouds = points
        self.colors = colors
        self.conf = conf
        self.conf_threshold = np.percentile(self.conf, conf_threshold_percentile) + 1e-6
        self.proj_mats = intrinsics_inv
    
    def set_img_names(self, img_names):
        self.img_names = img_names
            
    def add_all_frames(self, frames):
        self.frames = frames
    
    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors
    
    def get_lc_status(self):
        return self.is_lc_submap
    
    def get_id(self):
        return self.submap_id

    def get_conf_threshold(self):
        return self.conf_threshold
    
    def get_conf_masks_frame(self, index):
        return self.conf_masks[index]
    
    def get_frame_at_index(self, index):
        return self.frames[index, ...]
    
    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index
    
    def get_img_names_at_index(self, index):
        return self.img_names[index]

    def get_all_frames(self):
        return self.frames
    
    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors

    def get_first_homography_world(self, graph):
        homography =  graph.get_homography(self.get_id())
        homography = homography / homography[-1,-1]
        return homography

    def get_last_homography_world(self, graph):
        """
        Get the last camera projection matrix of the submap that is not a 
        loop closure frame. 
        Returns a 4x4 matrix normalized so that the last element is 1.
        """
        homography = graph.get_homography(self.get_id() + self.get_last_non_loop_frame_index())
        homography = homography / homography[-1,-1]
        return homography

    def get_first_pose_world(self, graph):
        """
        Get the first camera projection matrix of the submap. 
        Returns a 4x4 matrix normalized so that the last element is 1.
        """
        return np.linalg.inv(self.get_first_homography_world(graph))

    def get_last_pose_world(self, graph):
        """
        Get the last camera projection matrix of the submap that is not a 
        loop closure frame. 
        Returns a 4x4 matrix normalized so that the last element is 1.
        """
        projection_mat = graph.get_projection_matrix(self.get_id() + self.get_last_non_loop_frame_index())
        projection_mat = projection_mat / projection_mat[-1,-1]
        return projection_mat

    def get_all_poses(self):
        return self.poses

    def get_all_poses_world(self, graph, give_camera_mat=False):
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        poses = []
        for index, homography_world in enumerate(homography_list):
            projection_mat = self.proj_mats[index] @ np.linalg.inv(homography_world) # TODO HERE
            projection_mat = projection_mat / projection_mat[-1,-1]
            if give_camera_mat:
                poses.append(projection_mat)
            else:
                cal, rot, trans, scale = decompose_camera(projection_mat[0:3,:])

                pose = np.eye(4)
                pose[0:3, 0:3] = rot
                pose[0:3,3] = trans
                poses.append(pose)
        return np.stack(poses, axis=0)
    
    def get_frame_pointcloud(self, pose_index):
        return self.pointclouds[pose_index]

    def set_frame_ids(self, file_paths):
        """
        Extract the frame number (integer or decimal) from the file names, 
        removing any leading zeros, and add them all to a list.

        Note: This does not include any of the loop closure frames.
        """
        frame_ids = []
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)  # matches integers and decimals
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids

    def set_last_non_loop_frame_index(self, last_non_loop_frame_index):
        self.last_non_loop_frame_index = last_non_loop_frame_index
    
    def set_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors
    
    def set_conf_masks(self, conf_masks):
        self.conf_masks = conf_masks
    
    def set_all_semantic_vectors(self, semantic_vectors):
        self.semantic_vectors = semantic_vectors

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])
    
    def get_frame_ids(self):
        # Note this does not include any of the loop closure frames
        return self.frame_ids

    def filter_data_by_confidence(self, data):
        init_conf_mask = self.conf > self.conf_threshold
        return data[init_conf_mask]

    def get_points_list_in_world_frame(self, graph, rectifing_homographies=None):
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        point_list = []
        frame_id_list = []
        frame_conf_mask = []
        for index in  range(len(self.pointclouds)):
            points = self.pointclouds[index]
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (homography_list[index] @ points_homogeneous.T).T
            points_transformed = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape)

            point_list.append(points_transformed)
            frame_id_list.append(self.frame_ids[index])
            conf_mask = self.conf_masks[index] > self.conf_threshold
            frame_conf_mask.append(conf_mask)
        return point_list, frame_id_list, frame_conf_mask

    def get_points_in_world_frame(self, graph):
        homography_list = [graph.get_homography(i + self.get_id()) for i in range(len(self.poses))]
        points_all = None
        for index in  range(len(self.pointclouds)):
            points = self.pointclouds[index]
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (homography_list[index] @ points_homogeneous.T).T
            points_transformed = (points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape)

            conf_mask = self.conf_masks[index] > self.conf_threshold

            points_transformed = points_transformed[conf_mask]
            if index == 0:
                points_all = points_transformed
            else:
                points_all = np.vstack([points_all, points_transformed])

        return points_all

    def get_voxel_points_in_world_frame(self, voxel_size, nb_points=8, factor_for_outlier_rejection=2.0):
        if self.voxelized_points is None:
            if voxel_size > 0.0:
                points = self.filter_data_by_confidence(self.pointclouds)
                points_flat = points.reshape(-1, 3)
                colors = self.filter_data_by_confidence(self.colors)
                colors_flat = colors.reshape(-1, 3) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_flat)
                pcd.colors = o3d.utility.Vector3dVector(colors_flat)
                self.voxelized_points = pcd.voxel_down_sample(voxel_size=voxel_size)
                if (nb_points > 0):
                    self.voxelized_points, _ = self.voxelized_points.remove_radius_outlier(nb_points=nb_points,
                                                                                           radius=voxel_size * factor_for_outlier_rejection)
            else:
                raise RuntimeError("`voxel_size` should be larger than 0.0.")

        points_flat = np.asarray(self.voxelized_points.points)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        H_world_map = self.poses[0]
        points_transformed = (H_world_map @ points_homogeneous.T).T

        voxelized_points_in_world_frame = o3d.geometry.PointCloud()
        voxelized_points_in_world_frame.points = o3d.utility.Vector3dVector(points_transformed[:, :3] / points_transformed[:, 3:])
        voxelized_points_in_world_frame.colors = self.voxelized_points.colors
        return voxelized_points_in_world_frame
    
    def get_points_colors(self):
        colors = self.filter_data_by_confidence(self.colors)
        return colors.reshape(-1, 3)

    def get_all_semantic_vectors(self):
        return self.semantic_vectors

    
    def get_points_in_mask(self, frame_index, mask, graph):
        points = self.get_points_list_in_world_frame(graph)[0][frame_index]
        points_flat = points.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        points_in_mask = points_flat[mask_flat]
        return points_in_mask
