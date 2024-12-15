import numpy as np
import os
import open3d as o3d

from keypoint_detection.filt_hole import get_region
from keypoint_detection.detect_harris_corners import detect_corners
from util.reconstruct import depth_image_to_point_cloud_np

def get_keypoint_and_segmentation(rgb: np.array,
                                  sigma: float = 0.9,
                                  kernel_size: int = 10,
                                  k: float = 0.1,
                                  thresh: float = 0.3,
                                  max_corner: int = 500,
                                  mode: str = "peghole",
                                  debug: bool = False):
    '''
    Get the peghole or peg keypoints and segmentation map    
    '''
    shape_segmentation = get_region(rgb, mode=mode)

    if debug:
        results_dir = os.path.join(os.getcwd(), './keypoint_detection/results')
        dest_img_path = os.path.join(results_dir, '{}_points_num_{}.jpg'.format(mode, '{}'))
    else:
        dest_img_path = None

    corners = detect_corners(shape_segmentation, dest_img_path, kernel_size, sigma, k, thresh, max_corner)

    return corners, shape_segmentation

def get_keypoint_pcd(keypoints: np.array,
                     depth: np.array,
                     K: np.array,
                     view_matrix: np.array,
                     epsilon: int = 10,
                     store: bool = False,
                     color: np.array = None,):
    
    new_depth = np.full_like(depth, float('inf'))
    for pixel in keypoints:
        region = depth[pixel[1] - epsilon: pixel[1] + epsilon,
                       pixel[0] - epsilon: pixel[0] + epsilon]
        
        pixel_depth_max = np.max(region)
        # print("pixel depth max: ", pixel_depth_max)
        # print(region)
        new_depth[pixel[1], pixel[0]] = pixel_depth_max    

    pc = depth_image_to_point_cloud_np(new_depth, K)    
    pc = view_matrix @ pc # pcd in world
    pc = pc.T[:, :3]

    if store:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)      

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        coordinate = coordinate.sample_points_uniformly(number_of_points=30000)

        if color is not None:
            color = np.repeat([color], pc.shape[0], axis=0)
            pcd.colors = o3d.utility.Vector3dVector(color)
        
        combined_pcd = pcd + coordinate
        o3d.io.write_point_cloud("./keypoints.ply", combined_pcd)
    
    return pc
    



