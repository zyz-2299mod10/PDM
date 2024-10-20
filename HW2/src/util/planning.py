import numpy as np
import cv2
import os
import math
import time

class TreeNode:
    def __init__(self, x, y, parent):
        '''
        x: x position
        y: y position
        parent: TreeNode
        '''
        self.x = x
        self.y = y
        self.parent = parent

class RRT:
    def __init__(self, start, target, max_iter, step_size, map, target_color, target_object, finding_epsilon, collision_epsilon):
        self.map = cv2.imread(map)
        self.max_iter = max_iter
        self.step_size = step_size
        self.target_color = target_color
        self.finding_epsilon = finding_epsilon
        self.target_object = target_object
        self.collision_epsilon = collision_epsilon

        # node
        self.start = TreeNode(start[0], start[1], None)
        self.target = TreeNode(target[1], target[0], None) # To consistence with the x, y coordinate (Cuz we store it in cv2 data structure)

        # tree
        self.tree = []
        self.tree.append(self.start)
        self.path = []
        self.smooth_path = []
    
    def get_tree_size(self,):
        return len(self.tree)

    def remove_node(self, index):
        self.tree.pop(index)

    def add_node(self, x, y, insert_index):
        new_node = TreeNode(x, y, None)
        self.tree.insert(insert_index, new_node)

    def isFree(self, insert_index):
        '''
        Check if the newest inserted node is free, otherwise, remove it from tree
        '''
        x = self.tree[insert_index].x
        y = self.tree[insert_index].y

        if (self.map[y, x] != [255, 255, 255]).any():
            self.remove_node(insert_index)
            return False
        return True

    def distance(self, n1, n2):
        (x1, y1) = (float(self.tree[n1].x), float(self.tree[n1].y))
        (x2, y2) = (float(self.tree[n2].x), float(self.tree[n2].y))

        return ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)

    def near(self, index):
        '''
        Return the index of nearest node from given tree node
        '''
        max_dist = self.distance(0, index)
        nnear = 0
        for i in range(index):
            cur_dist = self.distance(i, index)
            if(cur_dist < max_dist):
                max_dist = cur_dist
                nnear = i            
        return nnear

    def step(self, nnear, nrand):
        '''
        Modify the newest node to avaliable position
        '''
        dist = self.distance(nnear, nrand)
        if (dist > self.step_size):
            (xnear, ynear) = (self.tree[nnear].x, self.tree[nnear].y)
            (xrand, yrand) = (self.tree[nrand].x, self.tree[nrand].y)

            (px, py)=(xrand-xnear, yrand-ynear)
            theta = math.atan2(py, px)
            x = xnear + self.step_size*math.cos(theta)
            y = ynear + self.step_size*math.sin(theta)
            self.remove_node(nrand)
            self.add_node(int(x), int(y), nrand)

    def connect(self, nnear, nrand):
        nearest_point = np.array([self.tree[nnear].x, self.tree[nnear].y])
        newest_point = np.array([self.tree[nrand].x, self.tree[nrand].y])
        line = np.linspace(nearest_point, newest_point, 50, endpoint=True)
        line = line.astype(int)

        for point in line:
            finding_region = self.map[point[1] - self.collision_epsilon : point[1] + self.collision_epsilon,
                                      point[0] - self.collision_epsilon : point[0] + self.collision_epsilon]
            if (finding_region != [255, 255, 255]).any():
                self.remove_node(nrand)
                return        
            
        self.tree[nrand].parent = self.tree[nnear]

    def expand(self):
        x = np.random.randint(0, self.map.shape[1])
        y = np.random.randint(0, self.map.shape[0])
        insert_index = self.get_tree_size()
        
        self.add_node(x, y, insert_index)
        if (self.isFree(insert_index)):
            nearest_index = self.near(insert_index)
            self.step(nearest_index, insert_index)
            self.connect(nearest_index, insert_index)

    def bias(self):
        '''
        To find the goal faster
        '''
        n = self.get_tree_size()
        self.add_node(n, self.target.x, self.target.y)

        nnear = self.near(n)
        self.step(nnear, n)
        self.connect(nnear, n)
    
    def finded_target(self, ):
        lastest_index = self.get_tree_size() - 1
        lastest_node = self.tree[lastest_index]
        finding_region = self.map[lastest_node.y - self.finding_epsilon : lastest_node.y + self.finding_epsilon,
                                  lastest_node.x - self.finding_epsilon : lastest_node.x + self.finding_epsilon]

        is_target_color_region = np.all(finding_region == self.target_color, axis=-1)
        if np.any(is_target_color_region):
            return True
        return False

    def path_to_goal(self,):
        current_node = self.tree[-1]        
        while current_node is not None:
            (x, y) = (current_node.x, current_node.y)
            self.path.append([x, y])
            current_node = current_node.parent
    
    def smooth(self):
        '''
        smooth the finding path
        '''
        s = 0 # s for start point
        self.smooth_path.append(self.path[s])
        for i in range(1, len(self.path)-1):
            current_point = np.array([self.path[s][0], self.path[s][1]])
            potential_point = np.array([self.path[i][0], self.path[i][1]])
            line = np.linspace(current_point, potential_point, 100, endpoint=True)
            line = line.astype(int)
            for point in line:
                finding_region = self.map[point[1] - self.collision_epsilon : point[1] + self.collision_epsilon,
                                          point[0] - self.collision_epsilon : point[0] + self.collision_epsilon]
                
                if (finding_region != [255, 255, 255]).any():
                    self.smooth_path.append(self.path[i - 1])
                    s = i - 1
                    break

        self.smooth_path.append(self.path[-1])
            
    def FindPath(self,):
        s = time.time()
        for i in range(0, self.max_iter):
            if i%10 != 0: self.expand()
            else: self.bias()

            if self.finded_target():
                print ("Found ! ")
                self.path_to_goal()
                self.smooth()
                break
        e = time.time()
        print("Total planning time: ", e - s)

        if self.path == []:
            print("Can't find the path to target, try again")
        else:            
            if not os.path.exists("./tmp_result_folder"):
                os.mkdir("./tmp_result_folder")    
            self.visualize_path()
            return self.smooth_path
                
    def visualize_path(self,):
        for node in self.tree:
            if node.parent is not None:
                parent = node.parent
                cv2.line(self.map, (node.x, node.y), (parent.x, parent.y), (0, 0, 0), 1) 
            cv2.circle(self.map, (node.x, node.y), 3, (0, 0, 0), thickness=-1)  

        for i in range(0, len(self.path) - 1):
            cv2.line(self.map, (self.path[i][0], self.path[i][1]), (self.path[i+1][0], self.path[i+1][1]), (0, 0, 255), 5)  # path
            cv2.circle(self.map, (self.path[i][0], self.path[i][1]), 6, (0, 0, 255), thickness=-1)  
        
        for i in range(0, len(self.smooth_path) - 1):
            cv2.line(self.map, (self.smooth_path[i][0], self.smooth_path[i][1]), (self.smooth_path[i+1][0], self.smooth_path[i+1][1]), (127,255, 0), 5)  # smooth path
            cv2.circle(self.map, (self.smooth_path[i][0], self.smooth_path[i][1]), 6, (127,255, 0), thickness=-1)  

        cv2.circle(self.map, (self.path[-1][0], self.path[-1][1]), 8, (0, 255, 0), thickness=-1)  # finding point 
        cv2.circle(self.map, (self.path[0][0], self.path[0][1]), 8, (255, 0, 0), thickness=-1)  # starting point
        
        # cv2.imwrite(f'./tmp_result_folder/path_{self.target_object}.png', self.map)
        cv2.imwrite(f'./path_{self.target_object}.png', self.map)

        print("Press 'ANY KEY' to continue")
        cv2.imshow('path_image', self.map)    
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class RRT_star(RRT):
    def __init__(self, start, target, max_iter, step_size, map, target_color, target_object, finding_epsilon, collision_epsilon, rewire_radius):
        super().__init__(start, target, max_iter, step_size, map, target_color, target_object, finding_epsilon, collision_epsilon)
        self.rewire_radius = rewire_radius  

    def near_nodes(self, nrand):
        """
        Return a list of nodes that are within the rewire radius from the random node.
        """
        near_nodes = []
        for i in range(self.get_tree_size()):
            dist = self.distance(i, nrand)
            if dist < self.rewire_radius:
                near_nodes.append(i)
        return near_nodes

    def rewire(self, nrand):
        """
        Rewire the tree to achieve shorter paths by connecting near nodes to the new random node.
        """
        near_nodes = self.near_nodes(nrand)
        for i in near_nodes:
            if i == nrand:
                continue
            dist = self.distance(nrand, i)
            potential_new_cost = self.distance(0, nrand) + dist
            existing_cost = self.distance(0, i)
            if potential_new_cost < existing_cost:
                self.tree[i].parent = self.tree[nrand]

    def connect(self, nnear, nrand):
        """
        Modify to allow rewiring after adding the node.
        """
        super().connect(nnear, nrand)  
        if nrand in self.tree:
            self.rewire(nrand)  

    def expand(self):
        """
        Overriding the expand method to use the rewiring strategy.
        """
        x = np.random.randint(0, self.map.shape[1])
        y = np.random.randint(0, self.map.shape[0])
        insert_index = self.get_tree_size()
        
        self.add_node(x, y, insert_index)
        if self.isFree(insert_index):
            nearest_index = self.near(insert_index)
            self.step(nearest_index, insert_index)
            self.connect(nearest_index, insert_index)

    def bias(self):
        """
        Overriding the bias method to add rewiring after biasing.
        """
        n = self.get_tree_size()
        self.add_node(self.target.x, self.target.y, n)

        nnear = self.near(n)
        self.step(nnear, n)
        self.connect(nnear, n)
    
    def FindPath(self):
        """
        Find the path using the RRT* algorithm.
        """
        s = time.time()
        for i in range(self.max_iter):
            if i % 10 != 0:
                self.expand()
            else:
                self.bias()

            if self.finded_target():
                print("Found!")
                self.path_to_goal()
                self.smooth()
                break
        e = time.time()
        print("Total planning time: ", e - s)

        if not self.path:
            print("Can't find the path to target, try again")
        else:
            if not os.path.exists("./tmp_result_folder"):
                os.mkdir("./tmp_result_folder")
            self.visualize_path()
            return self.smooth_path
    
    def visualize_path(self,):
        """
        Overriding the visulized method of RRT*
        """
        for node in self.tree:
            if node.parent is not None:
                parent = node.parent
                cv2.line(self.map, (node.x, node.y), (parent.x, parent.y), (0, 0, 0), 1) 
            cv2.circle(self.map, (node.x, node.y), 3, (0, 0, 0), thickness=-1)  

        for i in range(0, len(self.path) - 1):
            cv2.line(self.map, (self.path[i][0], self.path[i][1]), (self.path[i+1][0], self.path[i+1][1]), (0, 0, 255), 5)  # path
            cv2.circle(self.map, (self.path[i][0], self.path[i][1]), 6, (0, 0, 255), thickness=-1)  
        
        for i in range(0, len(self.smooth_path) - 1):
            cv2.line(self.map, (self.smooth_path[i][0], self.smooth_path[i][1]), (self.smooth_path[i+1][0], self.smooth_path[i+1][1]), (127,255, 0), 5)  # smooth path
            cv2.circle(self.map, (self.smooth_path[i][0], self.smooth_path[i][1]), 6, (127,255, 0), thickness=-1)  

        cv2.circle(self.map, (self.path[-1][0], self.path[-1][1]), 8, (0, 255, 0), thickness=-1)  # finding point 
        cv2.circle(self.map, (self.path[0][0], self.path[0][1]), 8, (255, 0, 0), thickness=-1)  # starting point
        
        # cv2.imwrite(f'./tmp_result_folder/path_{self.target_object}.png', self.map)
        cv2.imwrite(f'./path_{self.target_object}_star.png', self.map)

        print("Press 'ANY KEY' to continue")
        cv2.imshow('path_image', self.map)    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
