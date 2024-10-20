import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import json
import math
import natsort    
import shutil


class indoor_simulator:
    def __init__(self, sim_setting, unit_rotate = 1.0, unit_forward_len = 0.02):
        self.unit_rotate = unit_rotate
        self.unit_forward_len = unit_forward_len

        semantic_id_path = "replica_v1/apartment_0/habitat/info_semantic.json"
        with open(semantic_id_path, "r") as f:
            annotations = json.load(f)
        self.id_to_label = []
        self.id_to_label = np.where(np.array(annotations["id_to_label"]) < 0, 0, annotations["id_to_label"])

        self.sim_settings = sim_setting

        # Initiallize sim
        self.cfg = self.make_simple_cfg(self.sim_settings)
        self.sim = habitat_sim.Simulator(self.cfg)
        
        # Initiallize agent
        self.agent = self.sim.initialize_agent(self.sim_settings["default_agent"])
        self.agent_state = habitat_sim.AgentState()
        self.action_names = list(self.cfg.agents[self.sim_settings["default_agent"]].action_space.keys())
        print("Discrete action space: ", self.action_names)
        self.agent_state.position = np.array([0.0, 0.0, 0.0])  # agent in world space

        
    def transform_rgb_bgr(self, image):
        return image[:, :, [2, 1, 0]]

    def transform_depth(self, image):
        depth_img = (image / 10 * 255).astype(np.uint8)
        return depth_img

    def transform_semantic(self, semantic_obs):
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
        return semantic_img

    def make_simple_cfg(self, settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]
        

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0, settings["sensor_height"], 0.0]
        rgb_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        #depth snesor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        #semantic snesor
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=self.unit_forward_len) # unit: m
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.unit_rotate) # unit: degree
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.unit_rotate)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def navigateAndSee(self, action, frame):
        if action not in self.action_names:
            return
            
        observations = self.sim.step(action)        
        if frame % 5 == 0:
            objects_id = self.id_to_label[observations["semantic_sensor"]]
            target_mask = np.where(objects_id == self.target_semantic_id)          
            
            img = self.transform_rgb_bgr(observations["color_sensor"])
            mask_color = np.full(img.shape, (0, 0, 255), dtype=np.uint8)
            if target_mask[0].size > 0:
                blend_img = cv2.addWeighted(img, 0.5, mask_color, 0.5, 0)
                img[target_mask] = blend_img[target_mask]
                
            cv2.imwrite(f"./tmp_result_folder/RGB_{frame}.png", img)
        
    def calculate_rotation(self, dist):
        sensor_state = self.agent.get_state().sensor_states['color_sensor']

        current_rotation = math.atan2(
            2.0 * (sensor_state.rotation.w * sensor_state.rotation.y + 
                sensor_state.rotation.x * sensor_state.rotation.z), 
            1.0 - 2.0 * (sensor_state.rotation.y**2 + sensor_state.rotation.z**2)
        ) # yaw = atan2(2(wy+xz),1−2(y^2 + z^2))
        
        desired_rotation = -math.atan2(dist[0], -dist[1])

        # normalize to [-pi, pi]
        rot_diff = (desired_rotation - current_rotation + math.pi) % (2 * math.pi) - math.pi
        rotation_angle = np.degrees(rot_diff)
        
        # right or left
        RightOrLeft = "turn_left" if rot_diff > 0 else "turn_right"
        return RightOrLeft, abs(rotation_angle)
    
    def rotate(self, RightOrLeft, rotation_angle):
        while(rotation_angle > 0):
            self.navigateAndSee(RightOrLeft, frame=self.frame)
            rotation_angle -= self.unit_rotate
            self.frame += 1

    def forward(self, dist_len):
        while(dist_len > 0):
            self.navigateAndSee("move_forward", frame=self.frame)
            dist_len -= self.unit_forward_len
            self.frame += 1

    def ExecuteTrajectory(self, trajectory, target_point_world, target_object, target_semantic_id):
        self.frame = 0
        self.target_object = target_object
        self.target_semantic_id = target_semantic_id

        self.agent_state.position = np.array([trajectory[0][0], 0.0, trajectory[0][1]]) # starting point
        self.agent.set_state(self.agent_state)

        for i, target_point in enumerate(trajectory):
            print(f"Go to No.{i} node")

            sensor_state = self.agent.get_state().sensor_states['color_sensor']
            dist = np.array([target_point[0] - sensor_state.position[0],
                             target_point[1] - sensor_state.position[2]])
            dist_len = np.linalg.norm(dist)
            dist /= dist_len            
            RightOrLeft, rotation_angle = self.calculate_rotation(dist)    
            
            self.rotate(RightOrLeft, rotation_angle)
            self.forward(dist_len)

        print(f"Trajectort execute done, turning to target...")
        sensor_state = self.agent.get_state().sensor_states['color_sensor']
        dist = np.array([target_point_world[0] - sensor_state.position[0], 
                         target_point_world[1] - sensor_state.position[2]]) 
        dist_len = np.linalg.norm(dist)
        dist /= dist_len
        RightOrLeft, rotation_angle = self.calculate_rotation(dist)
        self.rotate(RightOrLeft, rotation_angle)

    def save_navigation(self, object=None):
        '''
        Save navigation result into GIF
        '''
        root_dir = "./tmp_result_folder"
        img_list = []
        img_name_list = []
        for i in natsort.natsorted(os.listdir(root_dir)):
            if i.startswith("RGB"):
                img_path = os.path.join(root_dir, i)
                img = Image.open(img_path)           
                img_list.append(img)      
                img_name_list.append(img_path)      
        # save
        print("Save GIF")
        img_list[0].save(os.path.join('../result', f"{object}.gif"), save_all=True, append_images=img_list[1:], duration=100, loop=0, quality=8)

        # remove tmp result folder
        shutil.rmtree(root_dir)
        print("All process done")


