from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import yaml
import pickle
import os
from scipy.spatial.transform import Rotation as R
import cv2
from typing import List
import numpy as np
import copy

from util.read_urdf import get_urdf_info
from util.other_isaacgym_fuction import (
    orientation_error,
    H_2_Transform,
    euler_xyz_to_matrix,
    pq_to_H,
    quat_mul_NotForTensor,
    euler_rotation_error_to_quaternion
    )
from util.camera import compute_camera_intrinsics_matrix
from hole_estimation.predict_hole_pose import CoarseMover, FineMover

class environment:
    def __init__(self, custom_parameters, robot_root, robot_type, urdf_root, robot_pose) -> None: 
        '''
        Usage:
            Set all environments (self.create_env) after setting all objects (self.set_box/mesh_asset)
            Calling-out each object by the "name" you setted (robot hand called "hand")
        input:
            robot pose: gymapi.Transform
        '''
        self.robot_root = robot_root
        self.urdf_root = urdf_root
        self.robot_pose = robot_pose
        self.robot_type = robot_type
        
        self.gym = gymapi.acquire_gym()       

        self.args = gymutil.parse_arguments(
            custom_parameters=custom_parameters,
        )        
        np.random.seed(self.args.random_seed)

        self.num_envs = self.args.num_envs
        
        device = self.args.device
        if device == "cuda":
            self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'
        else: self.device = 'cpu'
        
        self.prepare_sim()
        self.set_robot_asset()
        
        # prepare asset
        self.assets = {}
        self.camera = {}
    
    def prepare_sim(self):
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.contact_collection = gymapi.ContactCollection(1)
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        
        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
    
    def create_viewer(self):
        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")       
        
        # point camera at middle env
        num_per_row = int(math.sqrt(self.num_envs))
        cam_pos = gymapi.Vec3(1,  0.4, 1)
        cam_target = gymapi.Vec3(0, -0.4, 0)
        middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        
    def set_mesh_asset(self, 
                       mesh_file:str, 
                       fix_base:bool, 
                       disable_gravity:bool,
                       pos:gymapi.Vec3, 
                       collision:int, 
                       name:str, 
                       rot:gymapi.Quat = None,
                       color:gymapi.Vec3 = None,
                       random_pos_range:List[List] = None, 
                       random_rot_range:List[List[List]] = None,
                       semantic_id:int = None,):
        '''
        rot: exact quaternion
        random_pos_range: position random offset. format: [[x], [y], [z]]
        random_rot_range: given rotation axis and rotation range (degress). format: [[[axis], [range]]] (n* 3* 2)
        '''
        if name == None: 
            raise Exception("Need to name object")

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base
        asset_options.disable_gravity = disable_gravity
        asset_options.use_mesh_materials = True
        _asset = self.gym.load_asset(self.sim, self.urdf_root, mesh_file, asset_options)
        
        if rot is None:
            rot = gymapi.Quat(0, 0, 0, 1)
        
        if random_pos_range is not None:
            pos.x += np.random.uniform(random_pos_range[0][0], random_pos_range[0][1])
            pos.y += np.random.uniform(random_pos_range[1][0], random_pos_range[1][1])
            pos.z += np.random.uniform(random_pos_range[2][0], random_pos_range[2][1])
        
        if random_rot_range is not None:
            rot = gymapi.Quat(0, 0, 0, 1)
            for rotation in random_rot_range:
                rot_current = gymapi.Quat.from_axis_angle(gymapi.Vec3(rotation[0][0],
                                                                  rotation[0][1],
                                                                  rotation[0][2]),
                                                                  np.random.uniform(rotation[1][0], rotation[1][1]) * math.pi)
                rot = quat_mul_NotForTensor(rot, rot_current)
        
        # urdf information
        urdf_info = get_urdf_info(self.urdf_root, mesh_file)
        urdf_aabb = urdf_info.get_mesh_aabb_size()
        urdf_collisionMesh_path = urdf_info.get_collision_pathName_scale()["filename"]
        urdf_scale = urdf_info.get_collision_pathName_scale()["scale"]
        dims = gymapi.Vec3(urdf_aabb[0], urdf_aabb[1], urdf_aabb[2])

        asset_info = {
            'dims': dims,
            "asset": _asset,
            "obj_pos": pos,
            "obj_rot": rot,
            "collision": collision,
            "color": color,
            "urdf_collisionMesh_path": urdf_collisionMesh_path,
            "scale": urdf_scale,
            "semantic_id": semantic_id,
        }
        self.assets[name] = asset_info
    
    def set_box_asset(self, 
                      dims:gymapi.Vec3, 
                      fix_base:bool, 
                      disable_gravity:bool, 
                      pos:gymapi.Vec3, 
                      collision:int, 
                      name:str, 
                      rot:gymapi.Quat = None, 
                      color:gymapi.Vec3 = None, 
                      random_pos_range:List[List] = None, 
                      random_rot_range:List[List] = None,
                      semantic_id:int = None,):
        '''
        rot: exact quaternion
        random_pos_range: position random offset. format: [[x], [y], [z]]
        random_rot_range: given rotation axis and rotation range (degress). format: [[axis], [range]]
        '''        
        if name == None: 
            raise Exception("Need to name object")
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base
        asset_options.disable_gravity = disable_gravity
        asset_options.use_mesh_materials = True
        _asset = self.gym.create_box(self.sim, dims.x, dims.y, dims.z, asset_options)
        
        if rot is None:
            rot = gymapi.Quat(0, 0, 0, 1) # X, Y, Z, W
        
        if random_pos_range is not None:
            pos.x += np.random.uniform(random_pos_range[0][0], random_pos_range[0][1])
            pos.y += np.random.uniform(random_pos_range[1][0], random_pos_range[1][1])
            pos.z += np.random.uniform(random_pos_range[2][0], random_pos_range[2][1])

        if random_rot_range is not None:
            rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(random_rot_range[0][0],
                                                          random_rot_range[0][1],
                                                          random_rot_range[0][2]),
                                              np.random.uniform(random_rot_range[1][0], random_rot_range[1][1]) * math.pi)

        asset_info = {
            "dims": dims,
            "asset": _asset,
            "obj_pos": pos,
            "obj_rot": rot,
            "collision": collision,
            "color": color,
            "semantic_id": semantic_id,
        }
        self.assets[name] = asset_info
    
    def set_camera(self,
                   cam_pos:gymapi.Vec3,
                   cam_rot:gymapi.Quat = None,
                   cam_look_target:gymapi.Vec3 = None,
                   mode:str = "hand",
                   id:int = 0):
        '''
        mode: hand (wrist) or side camera
            if mode is hand, ur cam_pos and cam_rot will be the relative transformation to the robot hand
        id: To index the camera (default = 0)
        '''

        # create camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = 300
        camera_props.height = 300
        camera_props.horizontal_fov = 90 # default
        camera_props.near_plane = 0.01
        camera_props.far_plane = 1.5
        camera_props.enable_tensors = True

        cam_trans = gymapi.Transform()
        cam_trans.p = cam_pos
        cam_trans.r = cam_rot if cam_rot is not None else gymapi.Quat(0, 0, 0, 1)
        cam_info = {
            "mode":mode,
            "cam_props": camera_props,
            "cam_target": cam_look_target,
            "pose": cam_trans,
        }
        self.camera[id] = cam_info

    def set_robot_asset(self):
        with open('./simulate/robot_type.yml', 'r') as f:
            robot_file = yaml.safe_load(f)[self.robot_type]

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        self.robot_asset = self.gym.load_asset(self.sim, self.robot_root, robot_file, asset_options)
    
    def get_robot_defaultstate_prop(self):
        '''
        Currently only support franka
        '''
        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.4 * (franka_upper_limits + franka_lower_limits)

        # use position drive for all dofs
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(50.0)
        franka_dof_props["damping"][:7].fill(40.0)
            
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(100.0)
        franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        default_dof_pos[7:] = franka_upper_limits[7:]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        if self.robot_type == 'franka':
            self.franka_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)
        
        return franka_dof_props, default_dof_state, default_dof_pos
    
    def create_env(self):
        '''
        create environments with all asset and robot
        '''
        # configure env grid
        num_envs = self.num_envs
        num_per_row = int(math.sqrt(num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % num_envs)
        
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        self.obj_idxs = {}
        for obj_name in self.assets:
            self.obj_idxs[obj_name] = []
        self.obj_idxs["hand"] = []

        self.envs = []
        self.init_pos_list = []
        self.init_rot_list = []
        franka_dof_props, default_dof_state, default_dof_pos = self.get_robot_defaultstate_prop()
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env) 
            
            # set obj
            for obj_name in self.assets:   
                obj = self.assets[obj_name]
                _pose = gymapi.Transform()
                
                _pose.p = obj["obj_pos"]
                _pose.r = obj["obj_rot"] 

                if obj["semantic_id"] is not None:
                    _handle = self.gym.create_actor(env, obj["asset"], _pose, obj_name, i, obj["collision"], segmentationId = obj["semantic_id"])
                else:
                    _handle = self.gym.create_actor(env, obj["asset"], _pose, obj_name, i, obj["collision"])

                if obj["color"] is not None:
                    self.gym.set_rigid_body_color(env, _handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj["color"])                
                
                _idx = self.gym.get_actor_rigid_body_index(env, _handle, 0, gymapi.DOMAIN_SIM)
                self.obj_idxs[obj_name].append(_idx)
            
            # add franka
            franka_handle = self.gym.create_actor(env, self.robot_asset, self.robot_pose, "franka", i, 2)
                    
            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.obj_idxs["hand"].append(hand_idx)

            for id in self.camera:
                cam_handle = self.gym.create_camera_sensor(env, self.camera[id]["cam_props"])
                
                if self.camera[id]["mode"] == "hand":
                    self.gym.attach_camera_to_body(cam_handle, env, hand_handle, self.camera[id]["pose"], gymapi.FOLLOW_TRANSFORM)
                elif self.camera[id]["cam_target"] is not None: # side camera
                    self.gym.set_camera_location(cam_handle, env, self.camera[id]["pose"].p, self.camera[id]["cam_target"])
                else:
                    self.gym.set_camera_transform(cam_handle, env, self.camera[id]["pose"])

                self.camera[id]["handle"] = cam_handle
        
        # create viewer
        self.create_viewer()

        # prepare sim
        self.gym.prepare_sim(self.sim)
    
    def get_camera_img(self,
                       id:int,
                       env_idx:int = 0,
                       store:bool = False):
        '''
        get the camera image (view matrix and camera transform also) from specific camera (id and env)
        if u don't give the env index, it will return first env camera image

        return:
            depth: nparray
            rgb: nparray
            segmentation: nparray
            intrinsic: tensor
            view_matrix: nparray 4*4
            camera_transform: gymapi.Transform
        '''
        if id not in self.camera.keys():
            raise Exception(f"can't find the camera id: [{id}]")

        self.gym.start_access_image_tensors(self.sim)
        # get depth image
        color_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], self.camera[id]["handle"], gymapi.IMAGE_COLOR)
        rgb = gymtorch.wrap_tensor(color_tensor).cpu().numpy()[..., 0:3]
        depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], self.camera[id]["handle"], gymapi.IMAGE_DEPTH)
        depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
        segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], self.camera[id]["handle"], gymapi.IMAGE_SEGMENTATION)
        segmentation = gymtorch.wrap_tensor(segmentation_tensor).cpu().numpy()

        # for hole pose estimate
        height, width, _ = rgb.shape
        intrinsic = compute_camera_intrinsics_matrix(height, width, 90)
        camera_transform = self.gym.get_camera_transform(self.sim, self.envs[env_idx], self.camera[id]["handle"])
        view_matrix = self.gym.get_camera_view_matrix(self.sim, self.envs[env_idx], self.camera[id]["handle"])

        if store:
            store_depth = (-1) * depth * 1000 / 255 # to mm
            # print(store_depth)
            cv2.imwrite(f"./{self.camera[id]['mode']}_{id}_rgb.png", rgb)
            cv2.imwrite(f"./{self.camera[id]['mode']}_{id}_depth.png", store_depth)
            cv2.imwrite(f"./{self.camera[id]['mode']}_{id}_segmentation.png", segmentation)
        
        self.gym.end_access_image_tensors(self.sim)
        return depth, rgb, segmentation, intrinsic, view_matrix, camera_transform
    
    def get_specific_object_segmentation(self,
                                         id:int,
                                         object:str,
                                         env_idx:int = 0,
                                         store:bool = False)-> np.array:
        '''
        get the semantic segmentation image from specific camera (id and env) and specific object (use the name u set)
        if u don't give the env index, it will return first env camera image
        '''
        if self.assets[object]["semantic_id"] is None:
            raise AssertionError(f"You don't set the semantic id of {object}, U should set it in set_box/mesh_asset")


        segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], self.camera[id]["handle"], gymapi.IMAGE_SEGMENTATION)
        segmentation = gymtorch.wrap_tensor(segmentation_tensor).cpu().numpy()        
        segmentation[(segmentation != self.assets[object]["semantic_id"])] = 0

        if store:
            cv2.imwrite(f"./{self.camera[id]['mode']}_{id}_{object}_segmentation.png", segmentation)

        return segmentation
    
    def get_state(self):
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # Rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states).to(self.device)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states).to(self.device)
        dof_pos = dof_states[:, 0].view(self.num_envs, 9, 1).to(self.device)
        dof_vel = dof_states[:, 1].view(self.num_envs, 9, 1).to(self.device)

        return rb_states, dof_pos, dof_vel

    def step(self, action=None):
        if action is not None:
            for idx, env_i in enumerate(self.envs):
                env_i.step(action[idx])

        return self._evolve_step()

    def _evolve_step(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)

        return self.get_state()

    def kill(self):
        self.gym.destroy_viewer(self.viewer)
        for env in self.envs:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)

class Mover():
    def __init__(self, env:environment):
        self.env = env
    
    def return_initial_pose(self):
        '''
        Make robot return to the  initial pose
        '''
        self.IK_move(goal_pos=torch.tensor(self.env.init_pos_list, device=self.env.device), 
                     goal_rot=torch.tensor(self.env.init_rot_list, device=self.env.device), 
                     closed=False, T = 200)

    def get_grasping_pose(self, name)->torch.Tensor:
        '''
        return position and rotation (quaternion) from pre-defined grasping pose
        '''
        with open("./grasping_pose/USBStick_2.pickle", "rb") as f:
            grasp = pickle.load(f)

        z_mat = euler_xyz_to_matrix(0, 0, np.pi/2)
        grasp = torch.tensor(grasp, dtype = torch.float32, device=self.env.device) @ z_mat.repeat(100, 1, 1).to(self.env.device) # 100 for sample 100 grasping

        grasping_id = 10
        t = H_2_Transform(grasp[grasping_id, ...])

        grasp_position = [t.p.x, t.p.y, t.p.z]
        grasp_pos = []
        rb_states, dof_pos, _ = self.env.step()
        for env in range(self.env.num_envs):
            self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)
            
            pos_tmp = rb_states[self.env.obj_idxs[name][env], :3].tolist()
            grasp_pos.append([grasp_position[0] + pos_tmp[0],
                            grasp_position[1] + pos_tmp[1],
                            grasp_position[2] + pos_tmp[2] - 0.02]) 

        grasp_pos = torch.tensor(grasp_pos).to(self.env.device)
        grasp_rot = torch.tensor([t.r.x, t.r.y, t.r.z, t.r.w]).repeat(self.env.num_envs, 1).to(self.env.device)
        return grasp_pos, grasp_rot
    
    def get_naive_grasping_pose(self, name):
        '''
        return position and rotation (quaternion)
        '''        
        grasp_pos = []
        rb_states, dof_pos, _ = self.env.step()
        for env in range(self.env.num_envs):
            self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)
            
            pos_tmp = rb_states[self.env.obj_idxs[name][env], :3].tolist()      
            grasp_pos.append([pos_tmp[0], pos_tmp[1], pos_tmp[2] + 0.1]) 
        grasp_pos = torch.tensor(grasp_pos).to(self.env.device)

        up_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi)
        up_rot = torch.tensor([[up_rot.x, up_rot.y, up_rot.z, up_rot.w]]).to(self.env.device)
        obj_rot = rb_states[self.env.obj_idxs[name], 3:7]
        grasp_rot = quat_mul(obj_rot, up_rot)            

        return grasp_pos, grasp_rot
    
    def get_pridicted_hole_pose(self, camera_id, visualize = False):
        '''
        STILL NEED TO IMPROVE !!! (about the rotation accuracy, also not sure the performance in real-world..)
        Get the predicted hole pose from specific camera (only work in 1 env)

        return:
            transform matrix (3*4)
        '''        
        if not self.env.camera:
            raise Exception("Need to set camera first !!")

        depth, rgb, segmentation, intrinsic, view_matrix, t = self.env.get_camera_img(id = camera_id, env_idx = 0)
        depth = -depth
        view_matrix[:3, :3] = view_matrix[:3, :3] @ R.from_euler("XYZ", np.array([np.pi, 0, 0])).as_matrix()
        view_matrix[:3, 3] = np.array([t.p.x, t.p.y, t.p.z]) 
        view_matrix[3, :3] = np.array([0, 0, 0])

        predictor = CoarseMover(model_path='/kpts/2024-11-03_08-22', model_name='pointnet2_kpts',
                               checkpoint_name='best_model_e_390.pth', use_cpu=False, out_channel=9)
        H = predictor.predict_kpts_pose(depth=depth, factor=1, K=intrinsic.cpu().numpy(), view_matrix=view_matrix, fine_grain=False, visualize=visualize)        
        H = torch.tensor(H, device = self.env.device).to(torch.float32)

        # pq = H_2_Transform(H)

        return H
    
    def get_predicted_refinement_pose(self, camera_id, visualize = False):
        if not self.env.camera:
            raise Exception("Need to set camera first !!")
        
        rb_states, dof_pos, dof_vel = self.env.step()        
        gripper_pos = rb_states[self.env.obj_idxs['hand'][0], :3].tolist()
        gripper_pos[-1] = gripper_pos[-1] - 0.1

        depth, rgb, segmentation, intrinsic, view_matrix, t = self.env.get_camera_img(id = camera_id, env_idx = 0)
        depth = -depth
        view_matrix[:3, :3] = view_matrix[:3, :3] @ R.from_euler("XYZ", np.array([np.pi, 0, 0])).as_matrix()
        view_matrix[:3, 3] = np.array([t.p.x, t.p.y, t.p.z]) 
        view_matrix[3, :3] = np.array([0, 0, 0])

        predictor = FineMover(model_path='offset/2024-11-04_13-42', model_name='pointnet2_offset',
                              checkpoint_name='best_model_e_41.pth', use_cpu=False, out_channel=9)
        
        delta_trans, delta_rot, delta_rot_euler = predictor.predict_refinement_pose(depth=depth, factor=1, K=intrinsic.cpu().numpy(), view_matrix=view_matrix,
                                                                                    gripper_pose = gripper_pos, visualize = visualize)

        return delta_trans, delta_rot_euler

    def control_ik(self, dpose): 
        damping = 0.05
        _jacobian = self.env.gym.acquire_jacobian_tensor(self.env.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        j_eef = jacobian[:, self.env.franka_link_dict["panda_hand"] - 1, :, :7].to(self.env.device)

        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2).to(self.env.device)
        lmbda = torch.eye(6, device=self.env.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(self.env.num_envs, 7)
        return u
    
    def IK_move(self,
                goal_pos:torch.Tensor,
                goal_rot:torch.Tensor,
                closed: bool,
                T = 200):
        '''
        Currently only support franka
        '''
        rb_states, dof_pos, dof_vel = self.env.step()  
        pose_action = torch.zeros_like(dof_pos).squeeze(-1).to(self.env.device)

        if closed:
            grip_acts = torch.Tensor([[0., 0.]] * self.env.num_envs)
        else:
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.env.num_envs)

        for _ in range(T):
            rb_states, dof_pos, _ = self.env.step() 
            hand_pos = rb_states[self.env.obj_idxs['hand'], :3]
            hand_rot = rb_states[self.env.obj_idxs['hand'], 3:7]

            # compute position and orientation error
            pos_err = goal_pos - hand_pos
            orn_err = orientation_error(goal_rot, hand_rot)

            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

            # Deploy control based on type
            pose_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
            pose_action[:, 7:9] = grip_acts

            self.env.gym.set_dof_position_target_tensor(self.env.sim, gymtorch.unwrap_tensor(pose_action))
    
    def IK_move_object_to_target_pose(self,
                                      goal_pos:torch.Tensor,
                                      goal_rot:torch.Tensor,
                                      obj_name = str,
                                      T = 200):
        '''
        Currently only support franka
        '''
        rb_states, dof_pos, dof_vel = self.env.step()  
        pose_action = torch.zeros_like(dof_pos).squeeze(-1).to(self.env.device)

        grip_acts = torch.Tensor([[0., 0.]] * self.env.num_envs)

        for _ in range(T):
            rb_states, dof_pos, _ = self.env.step() 
            obj_pos = rb_states[self.env.obj_idxs[obj_name], :3]
            obj_rot = rb_states[self.env.obj_idxs[obj_name], 3:7]

            # compute position and orientation error
            pos_err = goal_pos - obj_pos
            orn_err = orientation_error(goal_rot, obj_rot)

            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

            # Deploy control based on type
            pose_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
            pose_action[:, 7:9] = grip_acts

            self.env.gym.set_dof_position_target_tensor(self.env.sim, gymtorch.unwrap_tensor(pose_action))
    
    def IK_move_from_transform_error(self, delta_pos, delta_rot, T, closed = False):
            
        rb_states, dof_pos, dof_vel = self.env.step()  
        pose_action = torch.zeros_like(dof_pos).squeeze(-1).to(self.env.device)

        if closed:
            grip_acts = torch.Tensor([[0., 0.]] * self.env.num_envs)
        else:
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.env.num_envs)

        delta_rot = euler_rotation_error_to_quaternion(delta_rot[0].cpu().numpy())
        delta_rot = delta_rot[np.newaxis, :]
        delta_rot = torch.tensor(delta_rot).to(delta_pos)
        for _ in range(T):
            rb_states, dof_pos, _ = self.env.step() 
            dpose = torch.cat([delta_pos, delta_rot], -1).unsqueeze(-1)

            # Deploy control based on type
            pose_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
            pose_action[:, 7:9] = grip_acts

            self.env.gym.set_dof_position_target_tensor(self.env.sim, gymtorch.unwrap_tensor(pose_action))

