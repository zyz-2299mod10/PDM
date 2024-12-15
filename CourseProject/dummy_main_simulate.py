from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
import cv2
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R

from simulate.simulate_isaac import environment, Mover
from util.read_urdf import get_urdf_info
from util.reconstruct import depth_image_to_point_cloud, depth_image_to_point_cloud_np
from util.other_isaacgym_fuction import ( 
    quat_mul_NotForTensor,
    H_2_Transform,
    pq_to_H
    )
from keypoint_detection.keypoint_detector import get_keypoint_and_segmentation, get_keypoint_pcd
from hole_estimation.ICP import icp
import random


''' 
Basic setting:
1. Custom parameters and asset/urdf root path
2. Initiallize robot and table configration
'''

# Add custom arguments
custom_parameters = [ 
    {"name": "--device", "type": str, "default": "cuda", "help": "[cuda, cpu]"},
    {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
    {"name": "--random_seed", "type": int, "default": 100, "help": "Numpy random seed"},
    {"name": "--object", "type": str, "default": "Arrow", "help": "[Arrow, Circle, Square]"}
]

asset_root = "/home/hcis/isaacgym/assets"
urdf_root = "/home/hcis/Perception/pdm-f24/PDM/PDM_urdf"

table_dims = gymapi.Vec3(0.8, 0.8, 0.4)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(table_pose.p.x - 0.65 * table_dims.x, table_pose.p.y, table_dims.z)

args = gymutil.parse_arguments(
            custom_parameters=custom_parameters,
    )   

# Simulate Objects
cap_name = f"{args.object}_cube_cap.urdf" 
bottle_name = f"{args.object}_cube_bottle.urdf" 

'''
Set environment
'''
Env = environment(custom_parameters=custom_parameters, robot_root=asset_root, urdf_root=urdf_root, 
                  robot_type='franka', robot_pose=franka_pose)
# table
Env.set_box_asset(dims=table_dims, fix_base=True, disable_gravity=True, pos=table_pose.p, collision=0, name='table')
# Env.set_box_asset(dims=gymapi.Vec3(5, 5, 0.02), fix_base=True, disable_gravity=True, pos=gymapi.Vec3(0.0, 0.0, 2), collision=0, name='roof')

# usb
usb_urdf_info = get_urdf_info(urdf_root, cap_name)
usb_aabb = usb_urdf_info.get_mesh_aabb_size()
usb_pose = gymapi.Transform()
usb_pose.p = gymapi.Vec3(table_pose.p.x + 0.08, table_pose.p.y + 0.1, table_dims.z + 0.5 * usb_aabb[2] + 0.05)
Env.set_mesh_asset(mesh_file=cap_name , fix_base=False, disable_gravity=False, name='peg', 
                   pos = usb_pose.p, collision=1, semantic_id=400, random_rot =[[[0,0,1],[-0.5, -0.5]]])

# socket
socket_urdf_info = get_urdf_info(urdf_root, bottle_name)
socket_aabb = socket_urdf_info.get_mesh_aabb_size()
socket_pose = gymapi.Transform()
socket_pose.p.x = usb_pose.p.x 
socket_pose.p.y = usb_pose.p.y 
socket_pose.p.z = table_dims.z + socket_aabb[2] + 0.025
Env.set_mesh_asset(mesh_file=bottle_name , fix_base=True, disable_gravity=False, name='socket', 
                   pos = socket_pose.p, collision=0, semantic_id=600, 
                   random_pos_range=[[-0.04, 0.04], [-0.08, -0.1], [0, 0]], random_rot=[[[0,0,1],[-0.5, -0.7]]])

# set camera
p = gymapi.Vec3(0.07, 0, 0.07)
look_down = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(-95))
Env.set_camera(cam_pos=p, cam_rot=look_down, mode="hand", id = 0)

p = gymapi.Vec3(0.9,  0.3,  0.7)
cam_target = gymapi.Vec3(0, -0.4, 0.4)
Env.set_camera(cam_pos=p, cam_look_target=cam_target, mode="side", id = 1)

# create environment(s)
Env.create_env()

for _ in range(50):
    Env.step()

'''
Execute action
'''
mover = Mover(Env)

# Grasp the peg and take image of the shape of the peg
grasp_pos, grasp_rot = mover.get_naive_grasping_pose('peg')

mover.IK_move(goal_pos=grasp_pos + torch.tensor([0, 0.065, 0.03], device=mover.env.device), goal_rot=grasp_rot, closed=False, T=500)

_peg_pcd_list = []
for env_idx in range(mover.env.num_envs):
    depth, rgb, segmentation, intrinsic, view_matrix, t = mover.env.get_camera_img(id = 0, env_idx = env_idx, store=False)
    cv2.imwrite("./peg.png", rgb)
    corners, shape_segmentation = get_keypoint_and_segmentation(rgb, mode="peg", debug=True)
    _peg_pcd = get_keypoint_pcd(corners, depth, intrinsic.cpu().numpy(), view_matrix, store=False, color=[1, 0, 0])
    _peg_pcd_list.append(_peg_pcd)

mover.IK_move(goal_pos=grasp_pos, goal_rot=grasp_rot, closed=False, T=500)
mover.IK_move(goal_pos=grasp_pos, goal_rot=grasp_rot, closed=True, T=200)

# calculate the relative peg pose in current camera frame
_peg_pcd_in_cam_list = []
for env_idx in range(mover.env.num_envs):
    _, _, _, _, view_matrix, _ = mover.env.get_camera_img(id = 0, env_idx = env_idx, store=False)
    _peg_pcd = _peg_pcd_list[env_idx]

    dummy = np.ones(_peg_pcd.shape[0])
    _peg_pcd_current = np.vstack((_peg_pcd.T, dummy[np.newaxis, :]))
    _peg_pcd_in_cam = np.linalg.inv(view_matrix) @ _peg_pcd_current
    _peg_pcd_in_cam = _peg_pcd_in_cam.T[:, :3]
    _peg_pcd_in_cam_list.append(_peg_pcd_in_cam)

mover.IK_move(goal_pos=grasp_pos + torch.tensor([0, 0, 0.075], device=mover.env.device), goal_rot=grasp_rot, closed=True, T=500)

# get peghole keypoints
_hole_pcd_list = []
for env_idx in range(mover.env.num_envs):
    depth, rgb, segmentation, intrinsic, view_matrix, t = mover.env.get_camera_img(id = 0, env_idx = env_idx, store=False)
    corners, shape_segmentation = get_keypoint_and_segmentation(rgb, mode="peghole", debug=True)
    _hole_pcd = get_keypoint_pcd(corners, depth, intrinsic.cpu().numpy(), view_matrix, store=False, color=[1, 0, 0]) # hole keypoints
    _hole_pcd_list.append(_hole_pcd)

# # CFVS coarse part
obj_goal_position = []
obj_goal_quaternion = []
for env_idx in range(mover.env.num_envs):
    pred_pose = mover.get_pridicted_hole_pose(camera_id=0, env_idx=env_idx, visualize=True)
    pred_pose[2, 3] = pred_pose[2, 3] + 0.065
    goal_pose_pq = H_2_Transform(pred_pose)

    obj_goal_position.append([goal_pose_pq.p.x , goal_pose_pq.p.y, goal_pose_pq.p.z])
    obj_goal_quaternion.append([goal_pose_pq.r.x, goal_pose_pq.r.y, goal_pose_pq.r.z, goal_pose_pq.r.w])

obj_goal_position = np.array(obj_goal_position)
obj_goal_quaternion = np.array(obj_goal_quaternion)
mover.IK_move_object_to_target_pose(goal_pos=torch.tensor(obj_goal_position, device=mover.env.device, dtype=torch.float32),
                                    goal_rot=torch.tensor(obj_goal_quaternion, device=mover.env.device, dtype=torch.float32),
                                    obj_name = 'peg', 
                                    T=500) 

# project peg pose to current camera pose (after coarse part)
transformation_list = []
for env_idx in range(mover.env.num_envs):
    _, _, _, _, view_matrix, _ = mover.env.get_camera_img(id = 0, env_idx = env_idx, store=False)
    _peg_pcd_in_cam = _peg_pcd_in_cam_list[env_idx]
    _hole_pcd = _hole_pcd_list[env_idx]
    _peg_pcd = _peg_pcd_list[env_idx]

    dummy = np.ones(_peg_pcd.shape[0])
    _peg_pcd_in_cam = np.vstack((_peg_pcd_in_cam.T, dummy[np.newaxis, :]))
    _peg_pcd_current = view_matrix @ _peg_pcd_in_cam
    _peg_pcd_current = _peg_pcd_current.T[:, :3]

    peg_pcd = o3d.geometry.PointCloud()
    peg_pcd.points = o3d.utility.Vector3dVector(_peg_pcd_current)
    hole_pcd = o3d.geometry.PointCloud()
    hole_pcd.points = o3d.utility.Vector3dVector(_hole_pcd)

    aligned_pcd = copy.deepcopy(peg_pcd)
    correspondences = np.array([[i, i] for i in range(len(_peg_pcd))])
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    transformation = estimation.compute_transformation(aligned_pcd, hole_pcd, o3d.utility.Vector2iVector(correspondences))
    aligned_pcd = aligned_pcd.transform(transformation)
    print("Transformation Matrix:\n", transformation)
    transformation_list.append(transformation)
    
    # finer = icp(peg_pcd, hole_pcd, 0.6)
    # finer.draw_registration_result(aligned_pcd, hole_pcd, f'peg_with_hole_pcd{env_idx}.ply', coordinate=True)

# Rotate->move along x,y -> move along z
goal_quaternion = []
trans_goal_position_xy = []
trans_goal_position_z = []
rb_states, dof_pos, _ = mover.env.step()
for env_idx in range(mover.env.num_envs):
    # ============ Rotate =====================
    h = rb_states[mover.env.obj_idxs['hand'][env_idx], :3]
    hr = rb_states[mover.env.obj_idxs['hand'][env_idx], 3:7]
    transformation = transformation_list[env_idx]

    rotation_matrix = torch.tensor(transformation[:3, :3], dtype=torch.float32, device=mover.env.device)
    translation_vector = torch.tensor(transformation[:3, 3], dtype=torch.float32, device=mover.env.device)

    hand_pose_matrix = pq_to_H(h, hr)
    rotated_pose_matrix = torch.eye(4, dtype=torch.float32, device=mover.env.device)
    rotated_pose_matrix[:3, :3] = rotation_matrix
    goal_rot_pose_matrix = rotated_pose_matrix @ hand_pose_matrix

    r = R.from_matrix(rotation_matrix.cpu().numpy())
    euler_angles = r.as_euler('xyz', degrees=True)
    if abs(euler_angles[0]) > 175: 
        x_180_matrix = R.from_euler('x', 180 + (180 - abs(euler_angles[0])), degrees=True).as_matrix()
        x_180_matrix = torch.tensor(x_180_matrix, dtype=torch.float32, device=mover.env.device)
        goal_rot_pose_matrix[:3, :3] = torch.mm(goal_rot_pose_matrix[:3, :3], x_180_matrix)
    goal_rot_pose_pq = H_2_Transform(goal_rot_pose_matrix)

    goal_quaternion.append([goal_rot_pose_pq.r.x, goal_rot_pose_pq.r.y, goal_rot_pose_pq.r.z, goal_rot_pose_pq.r.w])

    # ============ Move Along xy ===============
    translated_pose_matrix = torch.eye(4, dtype=torch.float32, device=mover.env.device)
    translated_pose_matrix[:3, 3] = torch.tensor([translation_vector[0], translation_vector[1], 0.0], dtype=torch.float32, device=mover.env.device)
    goal_trans_pose_matrix_xy = translated_pose_matrix @ goal_rot_pose_matrix
    goal_trans_pose_matrix_xy[1, 3] = goal_trans_pose_matrix_xy[1, 3] 
    goal_trans_pose_pq_xy = H_2_Transform(goal_trans_pose_matrix_xy)

    trans_goal_position_xy.append([goal_trans_pose_pq_xy.p.x, goal_trans_pose_pq_xy.p.y, h[2].item()])

    # ============ Move Along z ================
    translated_pose_matrix[:3, 3] = torch.tensor([0.0, 0.0, translation_vector[2]], dtype=torch.float32, device=mover.env.device)
    goal_trans_pose_matrix_z = translated_pose_matrix @ goal_trans_pose_matrix_xy
    goal_trans_pose_pq_z = H_2_Transform(goal_trans_pose_matrix_z)

    trans_goal_position_z.append([goal_trans_pose_pq_z.p.x, goal_trans_pose_pq_z.p.y, goal_trans_pose_pq_z.p.z])

goal_quaternion = np.array(goal_quaternion)
trans_goal_position_xy = np.array(trans_goal_position_xy)
trans_goal_position_z = np.array(trans_goal_position_z)

# print(rb_states[mover.env.obj_idxs['socket'], :3])
# print(rb_states[mover.env.obj_idxs['hand'], :3])
# print(trans_goal_position_xy)
# print(trans_goal_position_z)

# exec goal pose
mover.IK_move(goal_pos=rb_states[mover.env.obj_idxs['hand'], :3].to(dtype=torch.float32), 
              goal_rot=torch.tensor(goal_quaternion, dtype=torch.float32, device=mover.env.device),
              closed=True, 
              T=500)

mover.IK_move(goal_pos=torch.tensor(trans_goal_position_xy, dtype=torch.float32, device=mover.env.device),
              goal_rot=torch.tensor(goal_quaternion, dtype=torch.float32, device=mover.env.device),
              closed=True, 
              T=500)

mover.IK_move(goal_pos=torch.tensor(trans_goal_position_z, dtype=torch.float32, device=mover.env.device),
              goal_rot=torch.tensor(goal_quaternion, dtype=torch.float32, device=mover.env.device),
              closed=True, 
              T=1000)

success_case = 0
for env_idx in range(mover.env.num_envs):
    cpp = rb_states[mover.env.obj_idxs['peg'][env_idx], :3]
    chp = rb_states[mover.env.obj_idxs['socket'][env_idx], :3]
    dist = torch.norm(cpp - chp, dim = -1)
    if dist < 0.015:
        success_case+=1
print(f"success rate: {success_case / mover.env.num_envs}")

# mover.return_initial_pose()

while not Env.gym.query_viewer_has_closed(Env.viewer):
    Env.step()

Env.kill()