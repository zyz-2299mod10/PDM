import numpy as np 
import os 
from scipy.spatial import distance 
import cv2 

import torch
import torch.nn.functional as F

import math 
import gzip
import ujson
from numpy.linalg import inv

def prepare_data(file_name):
    def normalize_angle(x):
        x = x % (2 * np.pi)  # force in range [0, 2 pi)
        if x > np.pi:  # move to [-pi, pi)
            x -= 2 * np.pi
        return x

    def extract_yaw_from_matrix(matrix):
        """Extracts the yaw from a CARLA world matrix"""
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        yaw = normalize_angle(yaw)
        return yaw

    
    folder =  f"../data1/{file_name}"
    measurements_list = sorted(os.listdir(f"{folder}/measurements")  )
    bev_list = []
    ego_info_list = []

    for i, frame in enumerate(measurements_list):
        # You can choose the frame you want to compare
        if i == 55 or i == 65:
            bev_path_i = f"{folder}/bev_gt/{frame}"
            bev_path_i = bev_path_i.replace(".json.gz", ".npz")

            bev_image = np.load(f"{bev_path_i}")["arr_0"]
        
            measurement_path = f"{folder}/measurements/{frame}"
            bbox_path = f"{folder}/boxes/{frame}"
            
            with gzip.open(measurement_path, 'rt', encoding='utf-8') as f:
                measurements_i = ujson.load(f)
                
            with gzip.open(bbox_path, 'rt', encoding='utf-8') as f:
                bbox_i = ujson.load(f)
                    
            global_pos = measurements_i['global_pos']

            for box in bbox_i:
                if type(box) != list:       
                    object_class = box['class']
                    if "ego_car" == object_class:  
                        ego_matrix = np.array(box['matrix'])
                        ego_yaw = extract_yaw_from_matrix(ego_matrix)
                        ego_yaw = np.degrees(ego_yaw)
                        break
        
            ego_x = global_pos[0]
            ego_y = global_pos[1]

            bev_list.append(bev_image[2])
            ego_info_list.append([ego_x, ego_y, ego_yaw])

    return bev_list, ego_info_list


def create_affine_mat(x1, y1, theta1, x2, y2, theta2):
    """
    Create an affine transformation matrix to map the BEV representation from one ego vehicle pose to another.
    Please refer to the Documentation of Carla's coordinate system (https://github.com/autonomousvision/carla_garage/blob/main/docs/coordinate_systems.md)
    The Compass's coordinate system is different from the World's coordinate system.

    Hint:
    You can follow OpenCV's or PyTorch's Affine Transformation process and return proper shape of your affine matrix.
    The warp_features function in stp3.py will use this matrix to warp the BEV representation.        
    - [OpenCv Affine Transformations](https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html)
    - [PyTorch Affine Grid](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html)
    
    Parameters:
        x1, y1 (float): Initial position of the ego vehicle in world coordinates (meters).
        theta1 (float): Initial yaw angle of the ego vehicle (degrees).
        x2, y2 (float): Target position of the ego vehicle in world coordinates (meters).
        theta2 (float): Target yaw angle of the ego vehicle (degrees).

    Returns:
        np.ndarray: A affine transformation matrix.
    """
    # carla coordinate system ( meter --> pixel )
    bev_dim_x = int((32.0 - -32.0) / 0.25)
    bev_dim_y = int((32.0 - -32.0) / 0.25)
    
    # TODO_2-1: Implement the affine matrix calculation
    matrix = None
    
    
    raise NotImplementedError()
    return matrix

def warp_features(x, affine_mats):
    """
    Applies affine transformations to feature maps using the provided affine matrices.
    
    Parameters:
        x (torch.Tensor): The input feature map tensor, with shape (N, C, H, W).
        affine_mats (torch.Tensor): A tensor of affine transformation matrices you created by create_affine_mat() function.

    Returns:
        torch.Tensor: The warped BEV map tensor with the same shape as the input `x`.
    """
    # TODO_2-2: Implement the warp_features() function
    
    
    
    raise NotImplementedError
    return x


def process(file_name):
    
    bev_list, ego_info_list = prepare_data(file_name)
    
    bev_0 = bev_list[0]
    bev_1 = bev_list[1]

    # Source x, y, yaw
    ego_x_0, ego_y_0, ego_yaw_0 = ego_info_list[0]

    # Target x, y, yaw
    ego_x_1, ego_y_1, ego_yaw_1 = ego_info_list[1]


    # Adjust shape with the same as model's input
    bev_0_warpped = torch.tensor(bev_0).unsqueeze(0).unsqueeze(0).type(torch.float32)
    
    # Create affine matrix and warp the BEV
    affine_mat = create_affine_mat(ego_x_0, ego_y_0, ego_yaw_0, ego_x_1, ego_y_1, ego_yaw_1).unsqueeze(0)
    bev_0_warpped = warp_features(bev_0_warpped, affine_mat)
    
    # Turn back to numpy for visualization
    bev_0_warpped = bev_0_warpped.squeeze(0).squeeze(0).numpy()


    ## Visualize ##
    # Original BEV
    cv2.imwrite('mask_0_ori.png', bev_0 * 255)
    cv2.imwrite('mask_1_ori.png', bev_1 * 255)

    # Merge two images
    h, w = bev_1.shape
    merge = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    nonzero_indices = lambda arr: arr == 1
    
    merge[nonzero_indices(bev_0_warpped)] = (0, 0, 255)
    merge[nonzero_indices(bev_1)] = ( 0, 255, 0)
    
    # Save the merged BEV image
    cv2.imwrite('merge.png', merge )



if __name__ == "__main__":
    process("Town03_Scenario9_route9_07_24_18_38_57")
