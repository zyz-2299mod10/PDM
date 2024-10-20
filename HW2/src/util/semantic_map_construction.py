import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import os

class SemanticMap():
    def __init__(self):
        self.pcd_coor = np.load('semantic_3d_pointcloud/point.npy')
        self.pcd_color_01 = np.load('semantic_3d_pointcloud/color01.npy')

        # print(np.min(self.pcd_coor, axis = 0)) # [-0.07882993 -0.04527081 -0.12619986]
        # print(np.max(self.pcd_coor, axis = 0)) # [0.15922015 0.06031018 0.25276738]

        # remove roof and floor
        other = (self.pcd_coor[:, 1] <= -0.002) & (self.pcd_coor[:, 1] >= -0.03)
        self.remove_both_pcd = self.pcd_coor[other]

        # color
        self.other_color = self.pcd_color_01[other]

    def save_semantic_map(self):
        '''
        Save the semantic map that remove the roof and floor    
        '''        
        # remove roof and floor
        plt.figure()
        plt.scatter(self.remove_both_pcd[:, 0], self.remove_both_pcd[:, 2], s=0.3, c=self.other_color)
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('remove_both.png', dpi=200, bbox_inches='tight')

    def Find_corresponden_point(self, x, y, z):
        '''
        Mapping the world coordinate into pixel coordinate
        '''
        color = np.vstack((self.other_color, np.array([0, 0, 0], dtype=np.float32)))
        coordinate = np.vstack((self.remove_both_pcd, np.array([x, y, z], dtype=np.float32)))

        plt.figure()
        plt.scatter(coordinate[:, 0], coordinate[:, 2], s=0.3, c=color)
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('tt.png', dpi=200, bbox_inches='tight')
        
        tt = cv2.imread('tt.png')
        mapping_color = [0, 0, 0]
        correspond_index = np.column_stack(np.where(np.all(tt == mapping_color, axis=-1)))
        
        os.remove('tt.png')

        return np.mean(correspond_index, axis=0)

    def get_2D_to_3D_transform(self):
        '''
        Return the transformation matrix from pixel coordinate to world coordinate
        '''
        points_3d = np.array([
            [0.05, 0, -0.05],
            [0.05, 0, 0.05],
            [-0.03, 0, 0.08],
            [0.04, 0, -0.07]
        ], dtype=np.float32)
        
        points_2d = np.array([
            self.Find_corresponden_point(*p) for p in points_3d
        ], dtype=np.float32)
        
        dest = points_3d[:, [0, 2]] 
        source = points_2d  
        
        transformation_matrix, mask = cv2.findHomography(source, dest)
        transformation_matrix *= 10000 / 255.0  

        return transformation_matrix

    def get_semantic_id(slef, target):
        df = pd.read_excel('color_coding_semantic_segmentation_classes.xlsx')

        # get the label of target object
        target_label = df[df['Name'] == target]['Unnamed: 0'].values[0].astype(int)
        return target_label


