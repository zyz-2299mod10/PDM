import os

from copy import deepcopy
import json
import time
import cv2
import carla
from collections import deque
import math
import torch
import torch.nn.functional as F
import carla
import numpy as np
from PIL import Image

import torchvision
from nav_planner import RoutePlanner
from pyquaternion import Quaternion

from stp3.trainer import TrainingModule
from stp3.utils.network import NormalizeInverse
from stp3.utils.geometry import (
    mat2pose_vec,
)

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from config import GlobalConfig

from autopilot import AutoPilot

# for debug 
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


# SAVE_PATH = os.environ.get('SAVE_PATH', None)
# IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

def get_entry_point():
    return 'Stp3Agent'

class Stp3Agent(AutoPilot):
    # def setup(self, path_to_conf_file):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None, scenario_name=None):
        super().setup(path_to_conf_file, route_index, traffic_manager, scenario_name)

        (self.save_path / "debug_stp3").mkdir()
        
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.distance_history = np.zeros(2) # 0 for previous, 1 for current
        
        trainer = TrainingModule.load_from_checkpoint(path_to_conf_file, strict=True)
        print(f'Loaded weights from \n {path_to_conf_file}')
        trainer.eval()
        device = torch.device('cuda:0')
        trainer.to(device)
        self.model = trainer.model
        self.cfg = self.model.cfg

        # Generate new config for the case that it has new variables.
        self.config = GlobalConfig()
        
        # Filtering
        self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=self.residual_state_x)
        self.ukf = UKF(dim_x=4,
                    dim_z=4,
                    fx=self.bicycle_model_forward,
                    hx=self.measurement_function_hx,
                    dt=self.config.carla_frame_rate,
                    points=self.points,
                    x_mean_fn=self.state_mean,
                    z_mean_fn=self.measurement_mean,
                    residual_x=self.residual_state_x,
                    residual_z=self.residual_measurement_h)
        
        # State noise, same as measurement because we
        # initialize with the first measurement later
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
        # Used to set the filter state equal the first measurement
        self.filter_initialized = False
        # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
        self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))
        

        # command 
        self.bb_buffer = deque(maxlen=1)
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev_tmp = [1e5, 1e5]


        #Temporal LiDAR
        self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)
        self.lidar_last = None
        
        
        self.x_prev = 0
        self.y_prev = 0
        self.compass_prev = 0
        
        self.extrinsics, self.intrinsics = self.get_cam_para()
        
        
        self.normalise_image = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.denormalise_img = torchvision.transforms.Compose(
                        [NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            torchvision.transforms.ToPILImage(),]
                        )
        
        self.input_buffer = {'rgb': deque(maxlen=11), 
                             'lidar':deque(maxlen=11), 
                             'xs': deque(maxlen=11), 
                             'ys': deque(maxlen=11), 
                             'yaw': deque(maxlen=11),
                             'intrinsics': deque(maxlen=11),
                             'extrinsics': deque(maxlen=11),
                             }
        
                
    def _init(self):
        
        super()._init()
        
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        
        # for debug 
        self._vehicle = CarlaDataProvider.get_hero_actor()


    def sensors(self):
        
        result = super().sensors()
        
        result +=[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 800, 'height': 450, 'fov': 90,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 800, 'height': 450, 'fov': 90,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 800, 'height': 450, 'fov': 90,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 800, 'height': 450, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 800, 'height': 450, 'fov': 90,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 800, 'height': 450, 'fov': 90,
                    'id': 'CAM_BACK_RIGHT'
                },
                {
                    'type': 'sensor.lidar.ray_cast',
                    'x': 0.0, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'rotation_frequency': 10,
                    'points_per_second': 600000,
                    'id': 'LIDAR'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
            ]
        
        return result

    def tick(self, input_data):
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_BACK_LEFT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT', 'CAM_BACK']:
            camera = input_data[cam][1][:, :, :3]
            # Also add jpg artifacts at test time, because the training data was saved as jpg.
            _, compressed_image_i = cv2.imencode('.jpg', camera)
            camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            
            imgs[cam] = img
            
        lidar = self.lidar_to_ego_coordinate(input_data['LIDAR'])
            
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
  
        gps_pos = self.convert_gps_to_carla(gps)
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
        #  propess compass 
        compass = self.normalize_angle(compass - np.deg2rad(90.0))
        
      
        if not self.filter_initialized:
            
            self.gps_offset_x = gps_pos[0]
            self.gps_offset_y = gps_pos[1]
            
            self.ukf.x = np.array([0.0, 0.0, self.normalize_angle(compass), speed])
            self.filter_initialized = True
            
        gps_pos[0] -=  self.gps_offset_x
        gps_pos[1] -=  self.gps_offset_y
        

        self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)        
        self.ukf.update(np.array([gps_pos[0], gps_pos[1], self.normalize_angle(compass), speed]))
        
        
        
        filtered_state = self.ukf.x
        self.state_log.append(filtered_state)

        pos_carla = filtered_state[0:2]
        waypoint_route = self._route_planner.run_step( [gps_pos[0], gps_pos[1], 0] )#filtered_state[0:3])  
        
        if len(waypoint_route) > 2:
            target_point, far_command = waypoint_route[1]
        elif len(waypoint_route) > 1:
            target_point, far_command = waypoint_route[1]
        else:
            target_point, far_command = waypoint_route[0]   
            
        target_point = target_point[:2]
        
        if (target_point[:2] != self.target_point_prev_tmp).all():
            self.target_point_prev_tmp = target_point
            self.commands.append(far_command.value)

        
        ego_target_point = self.inverse_conversion_2d(target_point, pos_carla, compass)
        
        result = {
                'imgs': imgs,
                'lidar': lidar,
                'gps': gps,
                'pos': pos_carla,
                'speed': speed,
                'compass': compass,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near': self.commands[-2],
                'command_near_xy': ego_target_point
                }
        
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.tick(input_data)
            self.lidar_last = deepcopy(tick_data['lidar'])
            
        tick_data = self.tick(input_data)
        
        lidar_indices = []
        for i in range(self.config.lidar_seq_len):
            lidar_indices.append(i * self.config.data_save_freq)
        
        # Current position of the car
        ego_x = self.state_log[-1][0]
        ego_y = self.state_log[-1][1]
        ego_theta = self.state_log[-1][2]


        ego_x_last = self.state_log[-2][0]
        ego_y_last = self.state_log[-2][1]
        ego_theta_last = self.state_log[-2][2]
                
        # We only get half a LiDAR at every time step. Aligns the last half into the current coordinate frame.
        lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)
        lidar_current = deepcopy(tick_data['lidar'])
        lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)
        self.lidar_buffer.append(lidar_full)
        self.lidar_last = deepcopy(tick_data['lidar'])
        
        # prepare rgb  
        images = []
        
        images.append(self.normalise_image(np.array(
                self.scale_image(Image.fromarray(tick_data['imgs']['CAM_FRONT']), output_size=(224, 400)))).unsqueeze(0))
        images.append(self.normalise_image(np.array(
                self.scale_image(Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']), output_size=(224, 400)))).unsqueeze(0))
        images.append(self.normalise_image(np.array(
                self.scale_image(Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']), output_size=(224, 400)))).unsqueeze(0))
        images.append(self.normalise_image(np.array(
                self.scale_image(Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']), output_size=(224, 400)))).unsqueeze(0))
        images.append(self.normalise_image(np.array(
                self.scale_image(Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']), output_size=(224, 400)))).unsqueeze(0))
        images.append(self.normalise_image(np.array(
                self.scale_image(Image.fromarray(tick_data['imgs']['CAM_BACK']), output_size=(224, 400)))).unsqueeze(0))
        images = torch.cat(images, dim=0)
        self.input_buffer['rgb'].append(images) 
        
        # prepare lidar 
        lidar_feature = np.array(self.lidar_to_histogram_features(lidar_full, True))
        lidar_feature = torch.from_numpy(lidar_feature).float()        
        self.input_buffer['lidar'].append(lidar_feature) 
        
        self.input_buffer['intrinsics'].append(self.intrinsics) 
        self.input_buffer['extrinsics'].append(self.extrinsics) 
        
        self.input_buffer['xs'].append(deepcopy(tick_data['pos'][0]))
        self.input_buffer['ys'].append(deepcopy(tick_data['pos'][1]))
        self.input_buffer['yaw'].append(deepcopy(np.degrees(tick_data['compass']))) # in degrees
        
        if len(self.input_buffer['xs']) < 11:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            self.control = control

            return control
        
        
        
        ## prepare model inputs 
        index = 10
        images = []
        intrinsics = []
        extrinsics = []
        lidar = []
        affine_mats = []
                               
        current_ego_x = self.input_buffer['xs'][index] 
        current_ego_y = self.input_buffer['ys'][index]
        current_ego_yaw = self.input_buffer['yaw'][index]

        seq_x = []
        seq_y = []
        seq_theta = []
        
        
        for index_t in [0, 5, 10]:
            images.append(self.input_buffer['rgb'][index_t].unsqueeze(0))
            intrinsics.append(self.input_buffer['intrinsics'][index_t].unsqueeze(0))
            extrinsics.append(self.input_buffer['extrinsics'][index_t].unsqueeze(0))
            
            x_t = self.input_buffer['xs'][index_t]
            y_t = self.input_buffer['ys'][index_t]
            yaw_t = self.input_buffer['yaw'][index_t]
            
            affine_mat = self.create_affine_mat(x_t, y_t, yaw_t, current_ego_x, current_ego_y, current_ego_yaw)
            affine_mats.append(affine_mat.unsqueeze(0))
            
            lidar_tensor = self.input_buffer['lidar'][index_t].unsqueeze(0) 
            grid = torch.nn.functional.affine_grid(affine_mat.unsqueeze(0), 
                    size=lidar_tensor.shape, align_corners=False)
            
            lidar_tensor = torch.nn.functional.grid_sample(lidar_tensor, grid, mode='nearest', 
                padding_mode='zeros', align_corners=False)
            lidar.append(lidar_tensor)
            
            seq_x.append(x_t)
            seq_y.append(y_t)
            seq_theta.append(yaw_t)
        
        
        images = torch.cat(images, dim=0).unsqueeze(0).to('cuda', dtype=torch.float32) # torch.Size([1, 3, 6, 3, 224, 400])
        intrinsics = torch.cat(intrinsics, dim=0).unsqueeze(0).to('cuda', dtype=torch.float32) # torch.Size([1, 3, 6, 3, 3])
        extrinsics = torch.cat(extrinsics, dim=0).unsqueeze(0).to('cuda', dtype=torch.float32) # torch.Size([1, 3, 6, 4, 4]) 
        affine_mats = torch.cat(affine_mats, dim=0).unsqueeze(0).to('cuda', dtype=torch.float32) # torch.Size([1, 3, 2, 3])
        lidar = torch.cat(lidar, dim=0).unsqueeze(0).to('cuda', dtype=torch.float32) # torch.Size([1, 3, 2, 256, 256])
        future_egomotion = self.get_future_egomotion(seq_x, seq_y, seq_theta).unsqueeze(0).to('cuda', dtype=torch.float32) # torch.Size([1, 2, 6])
        
        
        # to pixel space 
        command_point_x = int(256 // 2 - tick_data['command_near_xy'][0] * 4.0)
        command_point_y = int(256 // 2 + tick_data['command_near_xy'][1] * 4.0)
            
        
        command = torch.from_numpy(np.array([tick_data['command_near']])).long().unsqueeze(0)
        target_point = torch.from_numpy(np.array([command_point_x, command_point_y])).float().unsqueeze(0)
    
        
        with torch.no_grad():
            output = self.model(
                    images, intrinsics, extrinsics, future_egomotion, affine_mats, 
                    None, lidar, False, None, target_point, command
                )
    
        
        ## vis    
        self.vis(output, images, lidar)
        
        def get_our_control(output):
            """ 
            Use the output of the model to control the vehicle.
            To ensure the vehicle successfully reaches the target position, steering control is handled by the autopilot.
            In this case, only our throttle and brake controls are used.

            Hint:
            The future-predicted segmentation information is included in the output. You can use this information to make control decisions.
            Refer to the vis function for guidance on extracting segmentation information from the output.

            Args:
                model's output(dict): The output of the model.

            Returns:
                carla.VehicleControl: The control for the vehicle.
                - throttle (float) : A scalar value to control the vehicle throttle [0.0, 1.0]. Default is 0.0.
                - steer (float) : A scalar value to control the vehicle steering [-1.0, 1.0]. Default is 0.0.
                - brake (float) : A scalar value to control the vehicle brake [0.0, 1.0]. Default is 0.0.
            """
            steer = 0.0
            throttle = 0.0
            brake = 0.0
            # TODO_4: Implement the basic control logic

            # print(f"segmentation map: {output['segmentation']}") # (1, 7, 2, 256, 256) 
            segmentation_prob = output['segmentation'].argmax(dim=2).detach().cpu().numpy()
            # print(f"segmentation prob shape: {segmentation_prob.shape}")  # (1, 7, 256, 256) 

            mid_point_x = 128
            epsilon_x = 45
            mid_point_y = 128
            epsilon_y = 9
            obstacle_search_region = segmentation_prob[0,
                                                       2,
                                                       mid_point_x - epsilon_x: mid_point_x + epsilon_x,
                                                       mid_point_y - epsilon_y: mid_point_y + epsilon_y]
            
            # Compute absolute obstacle positions in the segmentation map
            obstacle_positions = np.array(np.where(obstacle_search_region > 0.2))  
            obstacle_positions[0] += (mid_point_x - epsilon_x)  # Adjust row index to absolute map
            obstacle_positions[1] += (mid_point_y - epsilon_y)  # Adjust column index to absolute map

            # Define vehicle's position in the BEV map
            vehicle_position = np.array([mid_point_x, mid_point_y])  

            # Compute distances to obstacles
            if len(obstacle_positions[0]) > 0:  # If obstacles are present
                distances = np.linalg.norm(obstacle_positions - vehicle_position[:, None], axis=0)
                min_distance = distances.min()
            else:
                min_distance = None
            
            print(f"min distance: {min_distance}")

            # Define control logic
            distance_theshold = 40
            self.distance_history[0] = self.distance_history[1]
            self.distance_history[1] = min_distance
            if min_distance is not None:
                # Stop if the obstacle is very close
                if (min_distance < distance_theshold) and (self.distance_history[0] > self.distance_history[1]): 
                    # Gradual brake increase based on distance (using a sigmoid-like function) 
                    print("Collision warning, stop")
                    max_brake = 1
                    throttle = 0
                    brake = max_brake / (1 + np.exp(2 * (min_distance - distance_theshold)))  
                else:
                    # it means the speed of the car front of my agent faster than mine or the car behind my agent is too closed (they will pause in a distance) 
                    throttle = 0.8
                    brake = 0 
            else:
                # Default control if no obstacle is near
                throttle = 0.7
                brake = 0

            print(f"current throttle: {throttle}, current brake: {brake}")     

            return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

        # Get the control from the autopilot
        control_by_autopilot = super().run_step(input_data, timestamp, plant=plant)
        # Achieve our control using the output of the model
        our_control = get_our_control(output)
        # Combine the control from the autopilot and the control from the model 
        control = carla.VehicleControl(steer=control_by_autopilot.steer, throttle=our_control.throttle, brake=our_control.brake)
            
        # CARLA will not let the car drive in the initial frames.
        # We set the action to brake so that the filter does not get confused.
        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control
        

        return control


    def vis(self, output, images, lidar):
        for key in output:
            # print(key)
            if  output[key] is not None and torch.is_tensor(output[key]):
                output[key]= output[key].detach().clone()
                    
        colors = {
            "black": (0, 0, 0),
            "silver": (192, 192, 192),
            "gray": (128, 128, 128),
            "white": (255, 255, 255),
            "maroon": (128, 0, 0),
            "red": (255, 0, 0),
            "purple": (128, 0, 128), # vehicle current t
            
            "fuchsia": (255, 0, 255),
            "green": (0, 128, 0),
            "lime": (0, 255, 0),
            "olive": (128, 128, 0),
            "yellow": (255, 255, 0),
            "navy": (0, 0, 128),
            "blue": (0, 0, 255),
            "teal": (0, 128, 128),
            "aqua": (0, 255, 255),
            "pink": (255, 182, 193),
            "orange": (255, 165, 0),
            "brown": (150, 105, 25),
            
            "purple_0": (210, 144, 248),
            "purple_1": (196, 110, 246),
            "purple_2": ( 44, 3, 67),
            "purple_3": (169, 42, 242),
            "purple_4": (154, 13, 236),
            "purple_5": (132, 11, 202),
            "purple_6": (110, 9, 168),
            
            "teal_0": (170, 221, 221),
            "teal_1": (149, 202, 207),
            "teal_2": (22, 47, 49),
            "teal_3":  (124, 190, 196),
            "teal_4": (99, 178, 185),
            "teal_5": (77, 164, 172),
            "teal_6": (66, 141, 148),

            
        }
        
        
        nonzero_indices = lambda arr: arr != 0
        b = 0
        t = 2
                    
        # draw current static objects ( driviable area, ) 
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            
            area_seg = output['area'].argmax(dim=2).detach().cpu().numpy()
            # current t 
            total_plot = np.zeros((*area_seg[b, t].shape, 3))
            output['area'] = F.softmax(output['area'], dim=2)   
            area_value_tensor = output['area'][:, :, 1:, :, :]
            area_value_tensor = area_value_tensor.squeeze(2).detach().cpu().numpy()
            
            total_plot[area_value_tensor[b, t] > 0.3] = colors['white']
            
            # draw area t-2
            area_plot_2 = np.zeros((*area_seg[b, t].shape, 3))
            area_plot_2[area_value_tensor[b, 0] > 0.3] = colors['white']
            
            # draw area t-1
            area_plot_1 = np.zeros((*area_seg[b, t].shape, 3))
            area_plot_1[area_value_tensor[b, 1] > 0.3] = colors['white']
                
                        
            if self.cfg.SEMANTIC_SEG.OBSTACLE.ENABLED:
                obs_seg = output['obstacle'].argmax(dim=2).detach().cpu().numpy()
                total_plot[nonzero_indices(obs_seg[b, t])] = colors['brown']
                    
        # vehicle segmentation 
        semantic_seg =  F.softmax(output['segmentation'], dim=2)[:, :, 1:, :, :].squeeze(2).detach().cpu().numpy()
            
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            if output['pedestrian'].shape[2] > 1:
                pedestrian_seg = F.softmax(output['pedestrian'], dim=2)[:, :, 1:, :, :].squeeze(2).detach().cpu().numpy()
            else:
                pedestrian_seg = output['pedestrian'].detach().squeeze(2).cpu().numpy()
            
        _, total_of_time, _, _ = semantic_seg.shape
        
        
        for t in range(total_of_time):
            total_plot[semantic_seg[b, t] > 0.4 ] = colors[f'purple_{t}']
            total_plot[pedestrian_seg[b, t] > 0.4 ] = colors[f'teal_{t}']

        
        # draw ego 
        total_plot[118:139, 124:132] = colors["blue"]
        
        
       # draw stop 
        t = 2
        if self.cfg.SEMANTIC_SEG.STOP.ENABLED:
            if output['stop'].shape[2] > 1:
                stop_seg = F.softmax(output['stop'], dim=2)[:, :, 1:, :, :].squeeze(2).detach().cpu().numpy()
            else:
                stop_seg = output['stop'].detach().squeeze(2).cpu().numpy()              
            total_plot[stop_seg[b, t] > 0.4 ] = colors[f'orange']
            
            
        
        # model inputs             
        t = 2
        image_front = np.array(self.denormalise_img(images[0,t,0].cpu()).transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
        image_left = np.array(self.denormalise_img(images[0,t,1].cpu()).transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
        image_rear_left = np.array(self.denormalise_img(images[0,t,2].cpu()).transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
        image_right = np.array(self.denormalise_img(images[0,t,3].cpu()).transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
        image_rear_right = np.array(self.denormalise_img(images[0,t,4].cpu()).transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
        image_rear = np.array(self.denormalise_img(images[0,t,5].cpu()).transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
        image_front_merge = cv2.hconcat([image_left, image_front, image_right]) 
        image_rear_merge = cv2.hconcat([image_rear_right, image_rear, image_rear_left]) 
    
        Draw_lidar_mask = np.zeros((256, 256, 3)) 
        lidar_up = np.array(lidar[0,t,0].cpu())
        lidar_down = np.array(lidar[0,t,1].cpu())
        Draw_lidar_mask[nonzero_indices(lidar_up)] = colors['olive']
        Draw_lidar_mask[nonzero_indices(lidar_down)] = colors['teal']
        
        image_front_merge = cv2.hconcat([image_left, image_front, image_right]) 
        image_rear_merge = cv2.hconcat([image_rear_right, image_rear, image_rear_left]) 
        
        image_rgb = cv2.vconcat([image_front_merge, image_rear_merge])

        image_rgb = cv2.resize(image_rgb, (512, 256), interpolation=cv2.INTER_AREA)
        
        output_mask = np.zeros((512, 768, 3))
        
        output_mask[:256, :256] = Draw_lidar_mask
        output_mask[:256, 256:768] = image_rgb
        
        output_mask[256:512, 512:] = total_plot
        output_mask[256:512, 0:256] = area_plot_1
        output_mask[256:512, 256:512] = area_plot_2
        
        
        bgr = output_mask[..., ::-1]
        
        
        if (self.step % self.config.data_save_freq == 0) :
            frame = self.step // self.config.data_save_freq
            
            cv2.imwrite(str(self.save_path / 'debug_stp3' / (f'{frame:04}.jpg')), bgr)
                
        
        # self.save_path / "debug_stp3"
        # video_out.append(bgr)

    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self, results=None):
        out = cv2.VideoWriter(str(self.save_path / "video.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 
                        5,  (768, 512)) 
        file_list = sorted(os.listdir(str(self.save_path / "debug_stp3")))
        for file_name in file_list:
            tmp_img = cv2.imread(str(self.save_path / "debug_stp3" / file_name))
            out.write(tmp_img)
        out.release()
        
        pass
    
    
    
    # Filter Functions
    def bicycle_model_forward(self, x, dt, steer, throttle, brake):
        # Kinematic bicycle model.
        # Numbers are the tuned parameters from World on Rails
        front_wb = -0.090769015
        rear_wb = 1.4178275

        steer_gain = 0.36848336
        brake_accel = -4.952399
        throt_accel = 0.5633837

        locs_0 = x[0]
        locs_1 = x[1]
        yaw = x[2]
        speed = x[3]

        if brake:
            accel = brake_accel
        else:
            accel = throt_accel * throttle

        wheel = steer_gain * steer

        beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
        next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
        next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
        next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
        next_speed = speed + accel * dt
        next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

        next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

        return next_state_x
    
    def measurement_function_hx(self, vehicle_state):
        '''
            For now we use the same internal state as the measurement state
            :param vehicle_state: VehicleState vehicle state variable containing
                                an internal state of the vehicle from the filter
            :return: np array: describes the vehicle state as numpy array.
                            0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
            '''
        return vehicle_state
    
    def normalize_angle(self, x):
        x = x % (2 * np.pi)  # force in range [0, 2 pi)
        if x > np.pi:  # move to [-pi, pi)
            x -= 2 * np.pi
        return x
    
    def residual_state_x(self, a, b):
        y = a - b
        y[2] = self.normalize_angle(y[2])
        return y
           
    def measurement_mean(self, state, wm):
        '''
        We use the arctan of the average of sin and cos of the angle to
        calculate the average of orientations.
        :param state: array of states to be averaged. First index is the
        timestep.
        '''
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
        sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
        x[0] = np.sum(np.dot(state[:, 0], wm))
        x[1] = np.sum(np.dot(state[:, 1], wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(state[:, 3], wm))

        return x

    def residual_state_x(self, a, b):
        y = a - b
        y[2] = self.normalize_angle(y[2])
        return y

    def residual_measurement_h(self, a, b):
        y = a - b
        y[2] = self.normalize_angle(y[2])
        return y
    
    def state_mean(self, state, wm):
        '''
            We use the arctan of the average of sin and cos of the angle to calculate
            the average of orientations.
            :param state: array of states to be averaged. First index is the timestep.
            :param wm:
            :return:
            '''
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
        sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
        x[0] = np.sum(np.dot(state[:, 0], wm))
        x[1] = np.sum(np.dot(state[:, 1], wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(state[:, 3], wm))

        return x

    def inverse_conversion_2d(self, point, translation, yaw):
        """
        Performs a forward coordinate conversion on a 2D point
        :param point: Point to be converted
        :param translation: 2D translation vector of the new coordinate system
        :param yaw: yaw in radian of the new coordinate system
        :return: Converted point
        """
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], 
                                    [np.sin(yaw), np.cos(yaw)]])

        converted_point = rotation_matrix.T @ (point - translation)
        return converted_point
    
    def convert_gps_to_carla(self, gps):
        
        self.mean = np.array([0.0, 0.0])
        self.scale = np.array([111324.60662786, 111319.490945])
        
        """
        Converts GPS signal into the CARLA coordinate frame
        :param gps: gps from gnss sensor
        :return: gps as numpy array in CARLA coordinates
        """
        gps = (gps - self.mean) * self.scale
        # GPS uses a different coordinate system than CARLA.
        # This converts from GPS -> CARLA (90° rotation)
        gps = np.array([gps[1], -gps[0]])
        return gps
        
    def get_cam_to_ego(self, dof):
        """
        Calculate the 4x4 homogeneous transformation matrix from the camera coordinate frame 
        to the ego vehicle coordinate frame.

        Parameters:
            dof : A list containing 6 degrees of freedom in the format [x, y, z, roll, pitch, yaw].
                - x, y, z: Translation of the camera relative to the ego vehicle (meters).
                - roll, pitch, yaw: Rotation angles of the camera (degrees). 
                Only yaw (rotation about the z-axis) is considered.

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix that transforms points from 
                        the camera coordinate frame to the ego vehicle coordinate frame.
        """
        yaw = -(dof[5] + 180) * np.pi / 180
        rotation = Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)])
        translation = np.array(dof[:3])[:, None]
        translation[0] = -translation[0]
        cam_to_ego = np.vstack([
            np.hstack((rotation.rotation_matrix,translation)),
            np.array([0,0,0,1])
        ])
        return cam_to_ego
    
    def generate_intrinsic(self, width, height, fov):
        f = width / (2 * np.tan(fov * np.pi/ 360))
        Cu = width / 2
        Cv = height / 2
        intrinsic = torch.Tensor([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ])
        final_height, final_width = (224, 400)
        intrinsic = self.update_intrinsics(
            intrinsic, 0, 0,
            scale_width=final_width/width,
            scale_height=final_height/height
        )
        P = torch.Tensor([[0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]])
        intrinsic = intrinsic.matmul(P)
        return intrinsic
    
    def update_intrinsics(self, intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
        """
        Parameters
        ----------
            intrinsics: torch.Tensor (3, 3)
            top_crop: float
            left_crop: float
            scale_width: float
            scale_height: float
        """
        updated_intrinsics = intrinsics.clone()
        # Adjust intrinsics scale due to resizing
        updated_intrinsics[0, 0] *= scale_width
        updated_intrinsics[0, 2] *= scale_width
        updated_intrinsics[1, 1] *= scale_height
        updated_intrinsics[1, 2] *= scale_height

        # Adjust principal point due to cropping
        updated_intrinsics[0, 2] -= left_crop
        updated_intrinsics[1, 2] -= top_crop

        return updated_intrinsics
      
    def get_cam_para(self):
        cam_front = [0.8, 0.0, 1.6, 0.0, 0.0, 0.0] # x,y,z,roll,pitch, yaw
        cam_front_left = [0.27, -0.55, 1.6, 0.0, 0.0, -55.0]
        cam_front_right = [0.27, 0.55, 1.6, 0.0, 0.0, 55.0]
        cam_rear = [-2.0, 0.0, 1.6, 0.0, 0.0, 180.0]
        cam_rear_left = [-0.32, -0.55, 1.6, 0.0, 0.0, -110.0]
        cam_rear_right = [-0.32, 0.55, 1.6, 0.0, 0.0, 110.0]

        front_to_ego = torch.from_numpy(self.get_cam_to_ego(cam_front)).float().unsqueeze(0)
        front_left_to_ego = torch.from_numpy(self.get_cam_to_ego(cam_front_left)).float().unsqueeze(0)
        front_right_to_ego = torch.from_numpy(self.get_cam_to_ego(cam_front_right)).float().unsqueeze(0)
        rear_to_ego = torch.from_numpy(self.get_cam_to_ego(cam_rear)).float().unsqueeze(0)
        rear_left_to_ego = torch.from_numpy(self.get_cam_to_ego(cam_rear_left)).float().unsqueeze(0)
        rear_right_to_ego = torch.from_numpy(self.get_cam_to_ego(cam_rear_right)).float().unsqueeze(0)
        
        extrinsic = torch.cat([front_to_ego, front_left_to_ego, rear_left_to_ego, 
                front_right_to_ego, rear_right_to_ego, rear_to_ego], dim=0)

        intrinsic = self.generate_intrinsic(width=800, height=450, fov=90)
        intrinsic_rear = self.generate_intrinsic(width=800, height=450, fov=110)
        
        intrinsic = intrinsic.unsqueeze(0).expand(5,3,3)
        intrinsic = torch.cat([intrinsic, intrinsic_rear.unsqueeze(0)], dim=0)
        return extrinsic, intrinsic

    # lidar preprocess 
    def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
        pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
        rot_diff = self.normalize_angle(orientation_target - orientation)

        # Rotate difference vector from global to local coordinate system.
        rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                    [np.sin(orientation_target),
                                    np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
        pos_diff = rotation_matrix.T @ pos_diff
        
        rotation_matrix = np.array([[np.cos(rot_diff), -np.sin(rot_diff), 0.0], [np.sin(rot_diff), np.cos(rot_diff), 0.0], [0.0, 0.0, 1.0]])
        aligned_lidar = (rotation_matrix.T @ (lidar - pos_diff).T).T

        return aligned_lidar
    
    def lidar_to_ego_coordinate(self, lidar):
        """
        Converts the LiDAR points given by the simulator into the ego agents
        coordinate system
        :param config: GlobalConfig, used to read out lidar orientation and location
        :param lidar: the LiDAR point cloud as provided in the input of run_step
        :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
        coordinate system.
        """
        yaw = np.deg2rad(-90)
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

        # lidar pos 
        translation = np.array([0.0, 0.0, 2.5])

        # The double transpose is a trick to compute all the points together.
        ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

        return ego_lidar        

    def lidar_to_histogram_features(self, lidar, use_ground_plane=False):
        """
        Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
        :param lidar: (N,3) numpy, LiDAR point cloud
        :param use_ground_plane, whether to use the ground plane
        :return: (2, H, W) numpy, LiDAR as sparse image
        """

        max_height_lidar = 100.0
        lidar_split_height = 0.2

        def splat_points(point_cloud):
        
            # Max and minimum LiDAR ranges used for voxelization
            min_x = -32
            max_x = 32
            min_y = -32
            max_y = 32

            pixels_per_meter = 4.0
            
            # Max number of LiDAR points per pixel in voxelized LiDAR
            hist_max_per_pixel = 5
            # Height at which the LiDAR points are split into the 2 channels.
            # Is relative to lidar_pos[2]

            # 256 x 256 grid
            xbins = np.linspace(min_x, max_x,
                                (max_x - min_x) * int(pixels_per_meter) + 1)
            ybins = np.linspace(min_y, max_y,
                                (max_y - min_y) * int(pixels_per_meter) + 1)
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > hist_max_per_pixel] = hist_max_per_pixel
            overhead_splat = hist / hist_max_per_pixel
            # The transpose here is an efficient axis swap.
            # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
            # (x height channel, y width channel)
            return overhead_splat.T

        # Remove points above the vehicle
        lidar = lidar[lidar[..., 2] < max_height_lidar]
        
        above = lidar[lidar[..., 2] > lidar_split_height]
        above_features = splat_points(above)
        
        below = lidar[lidar[..., 2] <= 0.05 ]
        below_features = splat_points(below)
        
        above_features = np.rot90(above_features)
        below_features = np.rot90(below_features)
        
        if use_ground_plane:
            features = np.stack([above_features, below_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return features
    
    # preprocess images 
    def scale_image(self, image, output_size):
        """
        Scale and crop a PIL image, returning a channels-first numpy array.
        """
        image = image.resize((output_size[1], output_size[0]))
        image = np.asarray(image)
        image = image[:, ::-1]
        return image
    
    
    def create_affine_mat(self, x1, y1, theta1, x2, y2, theta2):
        """
        Create an affine transformation matrix to map the BEV representation from one ego vehicle pose to another.
        Please refer to the Documentation of Carla's coordinate system (https://github.com/autonomousvision/carla_garage/blob/main/docs/coordinate_systems.md)
        The Compass coordinate system is different from the World's coordinate system.

        Hint:
        You can follow OpenCV's or PyTorch's Affine Transformation process and return proper shape of your affine matrix.
        The wrap_features function in stp3.py will use this matrix to warp the BEV representation.        
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
        scale = 4.0  # pixel to m
    
        # related pose
        init_theta_rad = np.deg2rad(theta1)
        target_theta_rad = np.deg2rad(theta2)
        delta_theta = target_theta_rad - init_theta_rad
        rotation_matrix = np.array([
        	[np.cos(-init_theta_rad), -np.sin(-init_theta_rad)],
        	[np.sin(-init_theta_rad), np.cos(-init_theta_rad)]
        ])
        
        relative_pos = np.array([x2 - x1, y2 - y1])
        transformed_pos = rotation_matrix @ relative_pos
        dx_pixel = transformed_pos[1] * scale + bev_dim_x // 2
        dy_pixel = -transformed_pos[0] * scale + bev_dim_y // 2

	    # Normalize  
        normalized_dx = dx_pixel / (bev_dim_x / 2) - 1
        normalized_dy = dy_pixel / (bev_dim_y / 2) - 1

	    # Construct transformation matrix
        affine_matrix = np.array([
		    [np.cos(delta_theta), -np.sin(delta_theta), normalized_dx],
		    [np.sin(delta_theta), np.cos(delta_theta), normalized_dy]
		    ], dtype=np.float32)
        
        matrix =  torch.tensor(affine_matrix, dtype=torch.float32)
	
        return matrix
    
    def get_future_egomotion(self, seq_x, seq_y, seq_theta):
        """
        Computes the future egomotion transformations for a sequence of ego vehicle poses.
        This function calculates the relative motion between consecutive poses of the ego vehicle.
        For each pair of consecutive ego vehicle poses (position and yaw angle), it computes the
        rigid-body transformation matrix, which is then converted into a pose vector representing
        the motion between the two poses.
        The resulting egomotion is expressed in the coordinate system of the previous pose (ego coordinate system), not the world coordinate system.
        Hint: 
        You can use the mat2pose_vec(matrix: torch.Tensor) function to convert the transformation matrix to pose vector.
        
        Parameters:
            seq_x (list or np.ndarray): Sequence of x-coordinates of the ego vehicle in world coordinates.
            seq_y (list or np.ndarray): Sequence of y-coordinates of the ego vehicle in world coordinates.
            seq_theta (list or np.ndarray): Sequence of yaw angles (in degrees) of the ego vehicle.

        Returns:
            torch.Tensor: A tensor of shape (N-1, 6), where N is the length of the input sequences.
                        Each row represents the egomotion between two consecutive poses in the format:
                        [dx, dy, dz, roll, pitch, yaw].
        """
        future_egomotions = []
        # TODO_1: Implement the future egomotion calculation
        for i in range(len(seq_x) - 1):
            theta_rad_t0 = np.radians(seq_theta[i])
            t0 = np.eye(4, dtype=np.float32)
            t0[:2, :2] = [
                    [np.cos(theta_rad_t0), -np.sin(theta_rad_t0)],
                    [np.sin(theta_rad_t0), np.cos(theta_rad_t0)]
                    ]
            t0[0, 3] = seq_x[i]
            t0[1, 3] = seq_y[i]

            inv_t0 = np.block([
                [t0[:3, :3].T, -t0[:3, :3].T @ t0[:3, 3:4]],
                [np.zeros((1, 3)), np.ones((1, 1))]
            ])            
            
            # calculate relative motion
            theta_rad_t1 = np.radians(seq_theta[i + 1])
            t1 = np.eye(4, dtype=np.float32)
            t1[:2, :2] = [
                [np.cos(theta_rad_t1), -np.sin(theta_rad_t1)],
                [np.sin(theta_rad_t1), np.cos(theta_rad_t1)]
                ]
            t1[0, 3] = seq_x[i + 1]
            t1[1, 3] = seq_y[i + 1]
            
            relative_motion = np.dot(inv_t0, t1)
            relative_motion[3, :3] = 0
            relative_motion[3, 3] = 1
            
            # Convert the relative motion to pose vector
            pos_vec = mat2pose_vec(torch.tensor(relative_motion, dtype=torch.float32))
            future_egomotions.append(pos_vec.unsqueeze(0))
            
        return torch.cat(future_egomotions, dim=0)
        
