import numpy as np
import open3d as o3d
import argparse
import cv2
import os
import time
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

def cal_intrinsic_matrix(hov, h, w):
    vertical_fov = (h / w * hov) * np.pi / 180
    hov *= np.pi / 180

    f_x = (w / 2.0) / np.tan(hov / 2.0)
    f_y = (h / 2.0) / np.tan(vertical_fov / 2.0)

    K = np.array([[f_x, 0.0, w / 2.0], [0.0, f_y, h / 2.0], [0.0, 0.0, 1.0]])
    return K

def depth_image_to_point_cloud(rgb, depth, z_threshold) -> o3d.geometry.PointCloud:
    '''
    Reconstruct the point cloud by given rgb and depth image
    '''

    # TODO: Get point cloud from rgb and depth image 

    # common setup
    K = cal_intrinsic_matrix(90, 512, 512)
    pcd = o3d.geometry.PointCloud()

    # point cloud
    xmap = np.array([[j for i in range(depth.shape[0])] for j in range(depth.shape[1])])
    ymap = np.array([[i for i in range(depth.shape[0])] for j in range(depth.shape[1])])
    
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    mask_depth = depth <= z_threshold
    choose = mask_depth.flatten().nonzero()[0].astype(np.uint32)
    if len(choose) < 1:
        assert "out of z theshold"

    depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = -depth_masked / 255 * 10 # turn to m
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    points = np.concatenate((pt0, pt1, pt2), axis=1)
    pcd.points = o3d.utility.Vector3dVector(points)

    # color
    rgbs = (rgb[:, :, [2, 1, 0]].astype(np.float32) / 255).reshape(-1, 3)
    rgbs = rgbs[choose, :]
    pcd.colors = o3d.utility.Vector3dVector(rgbs)

    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, 
                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, 
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    return result.transformation


def local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Use Open3D ICP function to implement
    threshold = voxel_size * 0.3
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    return result.transformation

def nearest_neighbor(source, target):
    '''
    Finding the nearset points from given target points
    '''
    target_kdtree = cKDTree(target)
    distances, indices = target_kdtree.query(source, k=1)
    return distances, indices

def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size, max_iters=20, convergence_threshold=1e-6):
    '''
    Implement the point2plane ICP to align the given 2 point clouds
    '''
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    target_normals = np.asarray(target_down.normals)

    trans = trans_init.copy()    
    prev_error = float('inf')
    current_max_iter = 0 
    for iteration in range(max_iters):
        current_max_iter = iteration
        source_transformed = (trans[:3, :3] @ source_points.T).T + trans[:3, 3]

        # finding the nearest neighbor of each points
        distances, indices = nearest_neighbor(source_transformed, target_points)
        valid_mask = distances < voxel_size * 0.3
        valid_source = source_transformed[valid_mask]
        valid_target = target_points[indices[valid_mask]]
        valid_normals = target_normals[indices[valid_mask]]

        # calculate point2plane
        A = np.zeros((len(valid_source), 6))
        b = np.zeros((len(valid_source), 1))

        for i in range(len(valid_source)):
            source_point = valid_source[i]
            target_point = valid_target[i]
            normal = valid_normals[i]

            # normal * (R * source_point + t - target_point)
            cross_prod = np.cross(source_point, normal)
            A[i, :3] = cross_prod  
            A[i, 3:] = normal       
            b[i] = normal.dot(target_point - source_point)

        # solve the LS 
        delta = np.linalg.lstsq(A, b, rcond=None)[0].flatten() 

        delta_rotation = R.from_rotvec(delta[:3]).as_matrix()
        delta_translation = delta[3:]

        delta_transformation = np.eye(4)
        delta_transformation[:3, :3] = delta_rotation
        delta_transformation[:3, 3] = delta_translation

        # update
        trans = delta_transformation @ trans

        # calculate error 
        mean_error = np.mean(distances[valid_mask]**2)
        # print("current error: ", mean_error)

        if np.abs(prev_error - mean_error) < convergence_threshold:
            break
        prev_error = mean_error
    # print("max iter: ", current_max_iter)

    return trans

def ICP_alignment(pcd_list, mode):
    '''
    Align different point cloud by given a list contains point cloud
    '''
    # preprocess
    pcd_down_list = []
    pcd_fpfh_list = []
    voxel_size = 0.1
    print("     Preprocessing point cloud......")
    s = time.time()
    for pcd in pcd_list:
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size=voxel_size)

        pcd_down_list.append(pcd_down)
        pcd_fpfh_list.append(pcd_fpfh)
    e = time.time()
    print("     Precessing pcd time: ", e - s)
    
    # ICP
    pred_cam_pos = []
    pred_cam_pos.append(np.eye(4))
    print("     ICP alignment......")
    s = time.time()
    for i in range(1, len(pcd_list)):
        source_down, target_down = pcd_down_list[i], pcd_down_list[i - 1]
        source_fpfh, target_fpfh = pcd_fpfh_list[i], pcd_fpfh_list[i - 1]

        # get initial transform
        init_trans = execute_global_registration(source_down, target_down,
                                                 source_fpfh, target_fpfh, voxel_size=voxel_size)
        if mode == "open3d":
            trans = local_icp_algorithm(source_down, target_down, init_trans, voxel_size=voxel_size)
        else:
            trans = my_local_icp_algorithm(source_down, target_down, init_trans, 
                                           voxel_size=voxel_size, max_iters=20, convergence_threshold=1e-6)
        
        pred_cam_pos.append(pred_cam_pos[i - 1] @ trans)
    e = time.time()
    print("     ICP alignment time: ", e - s)
    
    # align the original point cloud
    for i in range(len(pcd_list)):
        pcd_list[i].transform(pred_cam_pos[i])
    pred_cam_pos = np.array(pred_cam_pos)

    return pcd_list, pred_cam_pos

def reconstruct(args, pic_num = None):
    # TODO: Return results
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    depth_path = os.path.join(args.data_root, "depth")
    rgb_path = os.path.join(args.data_root, "rgb")
    
    if pic_num is None: 
        pic_num = len(os.listdir(depth_path))
    else: 
        pic_num = pic_num
    
    # get all point cloud 
    pcd_list = []
    z_threshold=1000
    print("Reconstruct point cloud......")
    s = time.time()
    for i in range(1, pic_num + 1):
        depth = cv2.imread(os.path.join(depth_path, f"{i}.png"))
        rgb = cv2.imread(os.path.join(rgb_path, f"{i}.png"))
        
        pcd = depth_image_to_point_cloud(rgb, depth, z_threshold)
        pcd_list.append(pcd)    
    e = time.time()
    print("reconstruct time: ", e - s)
    
    print("Precessing ICP......")
    s = time.time()
    result_pcd, pred_cam_pos = ICP_alignment(pcd_list, mode = args.version)
    e = time.time()
    print("Total ICP time: ", e - s)

    return result_pcd, pred_cam_pos

def cal_l2_dist(ground_truth_path, estimate_pose, num = None):
    if num is None:
        ground_truth = np.load(ground_truth_path)
    else:
        ground_truth = np.load(ground_truth_path)[:num, ...]

    ground_truth_pose = np.tile(np.eye(4), (ground_truth.shape[0], 1, 1))
    ground_truth_pose[:, 0:3, 0:3] = R.from_quat(ground_truth[:, 3:7]).as_matrix()
    ground_truth_pose[:, 0:3, 3] = ground_truth[:, 0:3]

    ground_truth_pose = np.tile(np.linalg.inv(ground_truth_pose[0, ...]), (ground_truth_pose.shape[0], 1, 1)) @ ground_truth_pose
    ground_truth_pose[:, 0, 2] = -ground_truth_pose[:, 0, 2]
    ground_truth_pose[:, 2, 0] = -ground_truth_pose[:, 2, 0]

    ground_truth_position = ground_truth_pose[:, 0:3, 3]
    ground_truth_position[:, 0] = -ground_truth_position[:, 0]
    ground_truth_position[:, 2] = -ground_truth_position[:, 2]
    ground_truth_position = ground_truth_position

    L2_dist = np.mean(np.linalg.norm(ground_truth_pose - estimate_pose))
    # print("Mean L2 distance: ", np.mean(np.linalg.norm(ground_truth_pose - estimate_pose)))
    return L2_dist

def draw_camera_trajectory(result_pcd, ground_truth_path, estimate_pose, num = None):
    if num is None:
        ground_truth = np.load(ground_truth_path)
    else:
        ground_truth = np.load(ground_truth_path)[:num, ...]

    ground_truth_pose = np.tile(np.eye(4), (ground_truth.shape[0], 1, 1))
    ground_truth_pose[:, 0:3, 0:3] = R.from_quat(ground_truth[:, 3:7]).as_matrix() # quaternion to rotation martix
    ground_truth_pose[:, 0:3, 3] = ground_truth[:, 0:3]

    ground_truth_pose = np.tile(np.linalg.inv(ground_truth_pose[0, ...]), (ground_truth_pose.shape[0], 1, 1)) @ ground_truth_pose
    ground_truth_pose[:, 0, 2] = -ground_truth_pose[:, 0, 2]
    ground_truth_pose[:, 2, 0] = -ground_truth_pose[:, 2, 0]

    ground_truth_position = ground_truth_pose[:, 0:3, 3]    
    ground_truth_position[:, 0] = -ground_truth_position[:, 0]
    ground_truth_position[:, 2] = -ground_truth_position[:, 2]
    ground_truth_position = ground_truth_position

    estimate_position = estimate_pose[:, 0:3, 3]

    # draw path
    line = [[i, i+1] for i in range(ground_truth_position.shape[0] - 1)]
    ground_truth_color = [[0, 0, 0] for _ in range(len(line))]
    estimate_color = [[1, 0, 0] for _ in range(len(line))]

    ground_truth_line = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(ground_truth_position),
        lines = o3d.utility.Vector2iVector(line)
    )
    ground_truth_line.colors = o3d.utility.Vector3dVector(ground_truth_color)

    estimate_line = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(estimate_position),
        lines = o3d.utility.Vector2iVector(line)
    )
    estimate_line.colors = o3d.utility.Vector3dVector(estimate_color)

    o3d.visualization.draw_geometries(result_pcd+[ground_truth_line, estimate_line])

class debuger:
    def __init__(self, args):
        """
        unit debug
        """
        self.args = args

    def show_single_pic_pc(self, pic_id = 1, z_threshold = 50):
        depth = cv2.imread(self.args.data_root + f"depth/{pic_id}.png")
        rgb = cv2.imread(self.args.data_root + f"rgb/{pic_id}.png")

        pcd = depth_image_to_point_cloud(rgb, depth, z_threshold=z_threshold)
        p = np.asarray(pcd.points)
        print(np.max(p, axis=0))

        trimesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([pcd, trimesh])
    
    def show_partial_ICP_result(self, num):
        result_pcd, pred_cam_pos = reconstruct(self.args, num) 

        # rmove roof
        for i in range(len(result_pcd)):
            points = np.asarray(result_pcd[i].points)
            rgbs = np.asarray(result_pcd[i].colors)
            choose = (points[:, 1] <= 0.8)
            points, rgbs = points[choose], rgbs[choose]
            result_pcd[i].points = o3d.utility.Vector3dVector(points[:, 0:3])
            result_pcd[i].colors = o3d.utility.Vector3dVector(rgbs[:, 0:3])

        trimesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries(result_pcd + [trimesh])

        ground_truth_path = self.args.data_root + "/GT_pose.npy"
        L2_dist = cal_l2_dist(ground_truth_path, pred_cam_pos, num)
        print("L2 Dist: ", L2_dist)
    
    def show_whole_result(self, num):
        result_pcd, pred_cam_pos = reconstruct(self.args, num) 

        # rmove roof
        for i in range(len(result_pcd)):
            points = np.asarray(result_pcd[i].points)
            rgbs = np.asarray(result_pcd[i].colors)
            choose = (points[:, 1] <= 0.8)
            points, rgbs = points[choose], rgbs[choose]
            result_pcd[i].points = o3d.utility.Vector3dVector(points[:, 0:3])
            result_pcd[i].colors = o3d.utility.Vector3dVector(rgbs[:, 0:3])

        # calculate L2 distance
        ground_truth_path = self.args.data_root + "/GT_pose.npy"
        L2_dist = cal_l2_dist(ground_truth_path, pred_cam_pos, num)

        # draw camera trajectory
        draw_camera_trajectory(result_pcd, ground_truth_path, pred_cam_pos, num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"    

    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''    
    # checker = debuger(args)
    # checker.show_single_pic_pc(1, 500)
    # checker.show_partial_ICP_result(num=80)
    # checker.show_whole_result(171)
    # exit()

    ts = time.time()
    result_pcd, pred_cam_pos = reconstruct(args)   

    # rmove roof
    for i in range(len(result_pcd)):
        points = np.asarray(result_pcd[i].points)
        rgbs = np.asarray(result_pcd[i].colors)
        choose = (points[:, 1] <= 0.8)
        points, rgbs = points[choose], rgbs[choose]
        result_pcd[i].points = o3d.utility.Vector3dVector(points[:, 0:3])
        result_pcd[i].colors = o3d.utility.Vector3dVector(rgbs[:, 0:3])            

    te = time.time()
    print("\nTotal spending time: ", te - ts)

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    # calculate L2 distance
    ground_truth_path = args.data_root + "/GT_pose.npy"
    L2_dist = cal_l2_dist(ground_truth_path, pred_cam_pos)
    print("Mean L2 distance: ", L2_dist)

    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    # draw camera trajectory
    draw_camera_trajectory(result_pcd, ground_truth_path, pred_cam_pos)
