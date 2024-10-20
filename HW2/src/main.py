import cv2
import numpy as np
from util.semantic_map_construction import SemanticMap
from util.semantic_map_construction import SemanticMap
from util.planning import RRT, RRT_star
import ast
import pandas as pd
from util.simulator import indoor_simulator

def click_event(event, x, y, flags, param):
    global start, img
    if event == cv2.EVENT_LBUTTONDOWN:
        start = [x, y]
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow('image', img)

def Choose_start_and_end():
    # save the image that remove the roof and floor
    GetMap = SemanticMap()
    GetMap.save_semantic_map()

    df = pd.read_excel("color_coding_semantic_segmentation_classes.xlsx")
    df['Color_Code (R,G,B)'] = df['Color_Code (R,G,B)'].apply(lambda x: ast.literal_eval(x.replace('0250', '250'))) # turn the string to integer
    name2color = dict(zip(df['Name'], df['Color_Code (R,G,B)']))

    # input target point
    print("Choose a target, OPTION: [rack, cushion, sofa, stair, cooktop]")
    while True:
        target_object = input('Target object: ')
        if target_object not in name2color:
            print('Object not found.')
            continue
        target_color = name2color[target_object]
        target_color = target_color[::-1] # RGB to BGR for cv2
        break

    # click start point
    print("Choose a start spot, Press 'ANY KEY' after done")
    global start, img
    img = cv2.imread('remove_both.png')
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    indices = np.where(np.all(img == target_color, axis=-1))
    pixel_coordinates = np.column_stack((indices[0], indices[1]))
    target = np.mean(pixel_coordinates, axis=0).astype(int)

    print("start point: ", start)
    print(f"{target_object}: ", target)

    return start, target, target_color, target_object
        
if __name__ == "__main__":
    sim_setting = {
            "scene": "replica_v1/apartment_0/habitat/mesh_semantic.ply",  # Scene path
            "default_agent": 0,  # Index of the default agent
            "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
            "width": 512,  # Spatial resolution of the observations
            "height": 512,
            "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
        }
    Simulator = indoor_simulator(sim_setting)

    print("\n\n")
    start, target, target_color, target_object = Choose_start_and_end()
    GetMap = SemanticMap()
    trans = GetMap.get_2D_to_3D_transform()

    # RRT part
    Planner = RRT(start, target, max_iter=2000, step_size=50, map="./remove_both.png", 
                  target_color=target_color, target_object=target_object, finding_epsilon=20, collision_epsilon=5)
    # Planner = RRT_star(start, target, max_iter=2000, step_size=50, map="./remove_both.png", 
    #                    target_color=target_color, target_object=target_object, finding_epsilon=20, collision_epsilon=5, rewire_radius=20)
    path = Planner.FindPath()

    if path is None:
        exit()

    path = np.array(path)
    path = path[:, [1, 0]] # (x, y) to (y, x): pixel
    target_semantic_id = GetMap.get_semantic_id(target_object)

    # pixel to world
    trajectory = np.matmul(np.hstack((path, np.ones((path.shape[0], 1)))), trans.T) # x, z
    trajectory = np.array(list(reversed(trajectory)))
    target_point_world = np.matmul(np.append(target, 1), trans.T) # x, z
    print("3D target point: ", target_point_world)
    print("3D path: \n", trajectory)
    print()

    Simulator.ExecuteTrajectory(trajectory, target_point_world, target_object, target_semantic_id)
    Simulator.save_navigation(object=target_object)

    