import os
import numpy as np
import gzip
import ujson
from tqdm import tqdm
import math
import multiprocessing
from math import sqrt
from stp3.utils.geometry import calculate_birds_eye_view_parameters

obj2idx = {
    "ROAD": 0,
    "ROUTE": 1,
    "ROAD_LINE": 2,
    "VEHICLES": 3,
    "Emercency_vehicle": 4,
    "PEDESTRIANS": 5,
    "R_LIGHT_STOP": 6,
    "G_LIGHT_STOP": 7,
    "Y_LIGHT_STOP": 8,
    "STOP_SIGN": 9,
    "EGO_AGENT": 10,
    "OBSTACLES": 11,
    "VISIBLE_MASK": 12,
    "STOP_LINE": 13
}

keys = ['front', 'front_left', 'front_right', 
        'rear', 'rear_left', 'rear_right',
        'front_depth', 'front_left_depth', 'front_right_depth',
        'rear_depth', 'rear_left_depth', 'rear_right_depth',
        'x', 'y', 'command_point', 'is_stop_moment',
        'theta', 'steer', 'throttle', 'brake', 'command', 'velocity',
        'hdmap', 'iamap', 'have_vehicle', 'have_pedestrian', 'have_stop_area', 'have_obstacle',
        'pedestrian_heatmap', 'stop_area_heatmap']

def compute_affine(src_ref, tar_ref, src_yaw, tar_yaw):
    rotation_matrix = np.array([[np.cos(np.radians(tar_yaw)), -np.sin(np.radians(tar_yaw))], 
                                [np.sin(np.radians(tar_yaw)), np.cos(np.radians(tar_yaw))]])
    T = rotation_matrix.T @ (src_ref - tar_ref)
    yaw_diff = np.radians(src_yaw - tar_yaw)
    R = np.array([
        [np.cos(yaw_diff), -np.sin(yaw_diff)],
        [np.sin(yaw_diff),  np.cos(yaw_diff)]
    ])
    return R, T

def process_preloading_pathfiles(episode_name, receptive_field, 
        sequence_length, bev_dimension, is_train, factor=4):
    bev_dim = max(bev_dimension)
    results = {k: [] for k in keys}
    num_seq = len(os.listdir(episode_name + "/rgb_front/")) - sequence_length
    scene = episode_name.split('/')[-1]
    for seq in range(num_seq):
        fronts, front_lefts, front_rights, rears, rear_lefts, rear_rights = [], [], [], [], [], []
        fr_depths, fr_le_depths, fr_ri_depths, re_depths, re_le_depths, re_ri_depths = [], [], [], [], [], []
        xs, ys, thetas = [], [], []
        ped_poss, stop_poss = [], []
        ped_extents, stop_extents = [], []
        ped_heatmaps, stop_heatmaps = [], []
        is_stop_moments = []
        hdmap = []
        for i in range(receptive_field): # 0,3 
            filename = f"{str(seq+1+i).zfill(4)}.jpg"
            fronts.append(episode_name + "/rgb_front/" + filename)
            front_lefts.append(episode_name + "/rgb_front_left/" + filename)
            front_rights.append(episode_name + "/rgb_front_right/" + filename)
            rears.append(episode_name + "/rgb_back/" + filename)
            rear_lefts.append(episode_name + "/rgb_back_left/" + filename)
            rear_rights.append(episode_name + "/rgb_back_right/" + filename)
            filename = f"{str(seq+1+i).zfill(4)}.png"
            fr_depths.append(episode_name + "/depth_front/" + filename)
            fr_le_depths.append(episode_name + "/depth_front_left/" + filename)
            fr_ri_depths.append(episode_name + "/depth_front_right/" + filename)
            re_depths.append(episode_name + "/depth_back/" + filename)
            re_le_depths.append(episode_name + "/depth_back_left/" + filename)
            re_ri_depths.append(episode_name + "/depth_back_right/" + filename)
            
            
            hdmap.append(episode_name + f"/bev_mask_merge_{bev_dim}/" + f"{str(seq+1+i).zfill(4)}.npz")
            
            # position
            with gzip.open(episode_name + f"/measurements/{str(seq+1+i).zfill(4)}.json.gz","r") as read_file:
                data = ujson.load(read_file)
            
            with gzip.open(episode_name + f"/boxes/{str(seq+1+i).zfill(4)}.json.gz","r") as read_file:
                bbox_i = ujson.load(read_file)
                ped_pos, stop_pos = [], []
                ped_extent, stop_extent = [], []
                is_RY_light = is_stop = is_junction = 0
                for box in bbox_i:
                    if type(box) != list:       
                        object_class = box['class']
                        if "ego_car" == object_class:  
                            ego_matrix = np.array(box['matrix'])
                            ego_yaw = extract_yaw_from_matrix(ego_matrix)
                            ego_yaw = np.degrees(ego_yaw)
                        elif "traffic_light" == object_class and \
                            box['state'] != "Green" and box['id'] == data['correct_traffic_light_id']:
                            stop_pos.append(np.array([box['position'][0], box['position'][1]]))
                            stop_extent.append(np.array([box['extent'][0], box['extent'][1]]))
                            is_RY_light = 1
                        elif "walker" == object_class:
                            ped_pos.append(np.array([box['position'][0], box['position'][1]]))
                            ped_extent.append(np.array([box['extent'][0], box['extent'][1]]))
                        elif 'stop' in object_class and box['id'] == data['correct_stop_sign_id']:
                            is_stop = 1
                if data['junction']:
                    is_junction = 1
                        
                is_stop_moments.append(np.array([is_RY_light, is_stop, is_junction]))
                ped_poss.append(ped_pos) 
                stop_poss.append(stop_pos)
                ped_extents.append(ped_extent) 
                stop_extents.append(stop_extent)

            xs.append(data['pos_global'][0])
            ys.append(data['pos_global'][1])
            thetas.append(ego_yaw)

        command_point_x = int(bev_dim // 2 - data['target_point'][0] * factor)
        command_point_y = int(bev_dim // 2 + data['target_point'][1] * factor)
        results['command_point'].append(np.array([command_point_x, command_point_y]))
        results['steer'].append(data['steer'])
        results['throttle'].append(data['throttle'])
        results['brake'].append(data['brake'])
        results['command'].append(data['command'])
        results['velocity'].append(data['speed'])
        
        for i in range(receptive_field, sequence_length): # ( 3, 7)
            with gzip.open(episode_name + f"/measurements/{str(seq+1+i).zfill(4)}.json.gz","r") as read_file:
                data = ujson.load(read_file)
            xs.append(data['pos_global'][0])
            ys.append(data['pos_global'][1])
            
            with gzip.open(episode_name + f"/boxes/{str(seq+1+i).zfill(4)}.json.gz","r") as read_file:
                bbox_i = ujson.load(read_file)
                ped_pos, stop_pos = [], []
                ped_extent, stop_extent = [], []
                is_RY_light = is_stop = is_junction = 0
                for box in bbox_i:
                    if type(box) != list:       
                        object_class = box['class']
                        if "ego_car" == object_class:  
                            ego_matrix = np.array(box['matrix'])
                            ego_yaw = extract_yaw_from_matrix(ego_matrix)
                            ego_yaw = np.degrees(ego_yaw)
                        elif "traffic_light" == object_class and \
                            box['state'] != "Green" and box['id'] == data['correct_traffic_light_id']:
                            stop_pos.append(np.array([box['position'][0], box['position'][1]]))
                            stop_extent.append(np.array([box['extent'][0], box['extent'][1]]))
                            is_RY_light = 1
                        elif "walker" == object_class:
                            ped_pos.append(np.array([box['position'][0], box['position'][1]]))
                            ped_extent.append(np.array([box['extent'][0], box['extent'][1]]))
                        elif 'stop' in object_class and box['id'] == data['correct_stop_sign_id']:
                            is_stop = 1
                if data['junction']:
                    is_junction = 1

                is_stop_moments.append(np.array([is_RY_light, is_stop, is_junction]))
                ped_poss.append(ped_pos) 
                stop_poss.append(stop_pos)
                ped_extents.append(ped_extent) 
                stop_extents.append(stop_extent)

            if np.isnan(ego_yaw):
                thetas.append(0)
            else:
                thetas.append(ego_yaw)    
            hdmap.append(episode_name + f"/bev_mask_merge_{bev_dim}/" + f"{str(seq+1+i).zfill(4)}.npz")

        tar_ref = np.array([xs[receptive_field - 1], ys[receptive_field - 1]])
        tar_yaw = thetas[receptive_field - 1]
        for i in range(sequence_length):
            src_ref = np.array([xs[i], ys[i]])
            src_yaw = thetas[i]
            R, T = compute_affine(src_ref, tar_ref, src_yaw, tar_yaw)
            ped_heatmap = []
            for ped_pos, ped_ext in zip(ped_poss[i], ped_extents[i]):
                ped_pos_trans = R.dot(ped_pos) + T
                ped_pos_trans[0] = int(bev_dim // 2 - ped_pos_trans[0] * factor)
                ped_pos_trans[1] = int(bev_dim // 2 + ped_pos_trans[1] * factor)
                
                center = ped_pos_trans / factor
                center_int = center.astype(int)
                radius = gaussian_radius(ped_ext * 2, 0.1)
                ped_heatmap.append({'radius': radius, 'center': center_int, 
                    'offset': center - center_int})
            ped_heatmaps.append(ped_heatmap)

            stop_heatmap = []
            for stop_pos, stop_ext in zip(stop_poss[i], stop_extents[i]):
                stop_pos_trans = R.dot(stop_pos) + T
                stop_pos_trans[0] = int(bev_dim // 2 - stop_pos_trans[0] * factor)
                stop_pos_trans[1] = int(bev_dim // 2 + stop_pos_trans[1] * factor)
                
                center = stop_pos_trans / factor
                center_int = center.astype(int)
                radius = gaussian_radius(stop_ext * 2, 0.1)
                stop_heatmap.append({'radius': radius, 'center': center_int, 
                    'offset': center - center_int})
            stop_heatmaps.append(stop_heatmap)
            
        have_vehicle = False
        have_pedestrian = False
        have_stop_area = False
        have_obstacle = False
        for hd_path in hdmap[:2]:
            vehicle, pedestrian, stop_area, obstacle = get_labels(hd_path)
            if vehicle.sum() != 0:
                have_vehicle = True
            if pedestrian.sum() != 0:
                have_pedestrian = True
            if obstacle.sum() != 0:
                have_obstacle = True
            if stop_area.sum() != 0:
                have_stop_area = True
        results['have_vehicle'].append(have_vehicle)
        results['have_pedestrian'].append(have_pedestrian)
        results['have_stop_area'].append(have_stop_area)
        results['have_obstacle'].append(have_obstacle)
        results['front'].append(fronts)
        results['front_left'].append(front_lefts)
        results['front_right'].append(front_rights)
        results['rear'].append(rears)
        results['rear_left'].append(rear_lefts)
        results['rear_right'].append(rear_rights)
        results['front_depth'].append(fr_depths)
        results['front_left_depth'].append(fr_le_depths)
        results['front_right_depth'].append(fr_ri_depths)
        results['rear_depth'].append(re_depths)
        results['rear_left_depth'].append(re_le_depths)
        results['rear_right_depth'].append(re_ri_depths)
        results['x'].append(xs)
        results['y'].append(ys)
        results['theta'].append(thetas)
        results['hdmap'].append(hdmap)
        results['pedestrian_heatmap'].append(ped_heatmaps)
        results['stop_area_heatmap'].append(stop_heatmaps)
        results['is_stop_moment'].append(is_stop_moments)
        results['iamap'].append(episode_name + f"/bev_IA_{bev_dim}/" + f"{str(seq+receptive_field).zfill(4)}.npz")
    return results, is_train

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

def get_hdmap(path):
    img = np.load(path, allow_pickle=True)['arr_0']
    
    _, w, h = img.shape
    
    lane = np.zeros(shape=(w, h), dtype=np.uint8)
    road_area = np.zeros(shape=(w, h), dtype=np.uint8)
    
    ## get HD Map - Lane
    lane_idx = obj2idx["ROAD_LINE"]
    lane[img[lane_idx, :, :] != 0] = 1
        
    ## get HD map - Drivable area 
    road_idx = obj2idx["ROAD"]
    road_area[img[road_idx, :, :] != 0] = 1
    
    # down, right is the positive
    lane = lane[::-1,::-1]
    road_area = road_area[::-1,::-1]
    return lane, road_area

def get_labels(path):
    img = np.load(path, allow_pickle=True)['arr_0']
    
    _, w, h = img.shape
    
    pedestrian = np.zeros(shape=(w, h), dtype=np.uint8)
    vehicle = np.zeros(shape=(w, h), dtype=np.uint8)
    stop_area = np.zeros(shape=(w, h), dtype=np.uint8)
    obstacle = np.zeros(shape=(w, h), dtype=np.uint8)
    
    ped_idx = obj2idx["PEDESTRIANS"]
    pedestrian[img[ped_idx, :, :] != 0] = 1
        
    veh_idx = obj2idx["VEHICLES"]
    vehicle[img[veh_idx, :, :] != 0] = 1

    ## get HD map - Drivable area 
    # light_idx = obj2idx["R_LIGHT_STOP"]
    # stop_area[img[light_idx, :, :] != 0] = 1
    # light_idx = obj2idx["Y_LIGHT_STOP"]
    # stop_area[img[light_idx, :, :] != 0] = 1
    stop_idx = obj2idx["STOP_LINE"]
    stop_area[img[stop_idx, :, :] != 0] = 1

    obs_idx = obj2idx["OBSTACLES"]
    obstacle[img[obs_idx, :, :] != 0] = 1
    return vehicle.copy(), pedestrian.copy(), stop_area.copy(), obstacle.copy()

def get_iamap(ia_map_path):
    img = np.load(ia_map_path, allow_pickle=True)['arr_0']
    _, w, h = img.shape

    iamap = np.zeros(shape=img.shape, dtype=np.uint8)
    iamap[img != 0] = 1
    iamap = iamap[:, ::-1,::-1]
    return iamap.copy()

def get_command_point(command_point, bev_dimension):
    x_tp = (command_point[0] * 4  + bev_dimension[0] // 2)
    y_tp = (command_point[1] * 4  + bev_dimension[1] // 2)    
    x_tp, y_tp = np.clip(x_tp, 0, bev_dimension[0] - 1), np.clip(x_tp, 0, bev_dimension[1] - 1)
    point = np.array([x_tp, y_tp])
    
    # rotate to top
    rotate_yaw = 90 / 180 * np.pi
    rotation_matrix = np.array([[np.cos(rotate_yaw), -np.sin(rotate_yaw)], 
                                [np.sin(rotate_yaw), np.cos(rotate_yaw)]])
    point = np.array([[point[0] - bev_dimension[0] // 2, point[1] - bev_dimension[1] // 2]])    
    converted_point = (rotation_matrix.T @ (point).T)
    x_tp = converted_point[0] + bev_dimension[0] // 2
    y_tp = converted_point[1] + bev_dimension[1] // 2
    x_tp, y_tp = -x_tp + bev_dimension[0], -y_tp + bev_dimension[1]
    return [y_tp, x_tp]


def gaussian_radius(det_size, min_overlap):
    """Generate 2D gaussian radius.

    This function is modified from the `official github repo
    <https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/
    utils.py#L65>`_.

    Given ``min_overlap``, radius could computed by a quadratic equation
    according to Vieta's formulas.

    There are 3 cases for computing gaussian radius, details are following:

    - Explanation of figure: ``lt`` and ``br`` indicates the left-top and
        bottom-right corner of ground truth box. ``x`` indicates the
        generated corner at the limited position when ``radius=r``.

    - Case1: one corner is inside the gt box and the other is outside.

    .. code:: text

        |<   width   >|

        lt-+----------+         -
        |  |          |         ^
        +--x----------+--+
        |  |          |  |
        |  |          |  |    height
        |  | overlap  |  |
        |  |          |  |
        |  |          |  |      v
        +--+---------br--+      -
            |          |  |
            +----------+--x

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{(w-r)*(h-r)}{w*h+(w+h)r-r^2} \ge {iou} \quad\Rightarrow\quad
        {r^2-(w+h)r+\cfrac{1-iou}{1+iou}*w*h} \ge 0 \\
        {a} = 1,\quad{b} = {-(w+h)},\quad{c} = {\cfrac{1-iou}{1+iou}*w*h}
        {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

    - Case2: both two corners are inside the gt box.

    .. code:: text

        |<   width   >|

        lt-+----------+         -
        |  |          |         ^
        +--x-------+  |
        |  |       |  |
        |  |overlap|  |       height
        |  |       |  |
        |  +-------x--+
        |          |  |         v
        +----------+-br         -

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{(w-2*r)*(h-2*r)}{w*h} \ge {iou} \quad\Rightarrow\quad
        {4r^2-2(w+h)r+(1-iou)*w*h} \ge 0 \\
        {a} = 4,\quad {b} = {-2(w+h)},\quad {c} = {(1-iou)*w*h}
        {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

    - Case3: both two corners are outside the gt box.

    .. code:: text

            |<   width   >|

        x--+----------------+
        |  |                |
        +-lt-------------+  |   -
        |  |             |  |   ^
        |  |             |  |
        |  |   overlap   |  | height
        |  |             |  |
        |  |             |  |   v
        |  +------------br--+   -
        |                |  |
        +----------------+--x

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{w*h}{(w+2*r)*(h+2*r)} \ge {iou} \quad\Rightarrow\quad
        {4*iou*r^2+2*iou*(w+h)r+(iou-1)*w*h} \le 0 \\
        {a} = {4*iou},\quad {b} = {2*iou*(w+h)},\quad {c} = {(iou-1)*w*h} \\
        {r} \le \cfrac{-b+\sqrt{b^2-4*a*c}}{2*a}

    Args:
        det_size (list[int]): Shape of object.
        min_overlap (float): Min IoU with ground truth for boxes generated by
            keypoints inside the gaussian kernel.

    Returns:
        radius (int): Radius of gaussian kernel.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def process_preloading(cfg):
    root_dir = cfg.DATASET.DATAROOT
    sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES # 3 + 4 
    receptive_field = cfg.TIME_RECEPTIVE_FIELD # 3 
    bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
    )
    factor = cfg.DETECTION_FACTOR
    bev_dimension = bev_dimension.tolist()
    # val_towns = ['Town10HD']
    train_results = {k: [] for k in keys}
    valid_results = {k: [] for k in keys}
    def log_result(inputs):
        results, is_train = inputs
        for k in keys:
            if is_train:
                train_results[k].extend(results[k])
            else:
                valid_results[k].extend(results[k])
        pbar.update()

    pool = multiprocessing.Pool(processes=128)
    scenes = [s for s in os.listdir(root_dir) if not '.npy' in s]
    pbar = tqdm(total=len(scenes), desc="Preloading Route Configurations")

    for idx, scene in enumerate(scenes):
        path = os.path.join(root_dir, scene)
        is_train = True
        # for town in val_towns:
        #     if town in path:
        #         is_train = False
        #         break
        if (idx + 1) % 10 == 0:
            is_train = False
        pool.apply_async(process_preloading_pathfiles, 
            args = (path, receptive_field, sequence_length, bev_dimension, is_train, factor), 
            callback = log_result)
    
    pool.close()
    pool.join()
    pbar.close()

    pedestrian_sets = []
    stop_area_sets = []
    obstacle_sets = []
    for i, have_ped in enumerate(train_results['have_pedestrian']):
        if have_ped:
            pedestrian_sets.append(i)
    for i, have_stop in enumerate(train_results['have_stop_area']):
        if have_stop:
            stop_area_sets.append(i)
    for i, have_obs in enumerate(train_results['have_obstacle']):
        if have_obs:
            obstacle_sets.append(i)
    train_results['pedestrian_sets'] = pedestrian_sets
    train_results['stop_area_sets'] = stop_area_sets
    train_results['obstacle_sets'] = obstacle_sets
    np.save(f'{root_dir}/{cfg.TAG}_train_preload.npy', train_results)
    np.save(f'{root_dir}/{cfg.TAG}_valid_preload.npy', valid_results)