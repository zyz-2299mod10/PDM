# PDM HW1 BEV projection and 3D Scene Reconstruction

# BEV projection
Modify the pic_id in the top of the bev.py to choose the picture id  
```
python bev.py
```

# 3D reconstruction

First collect the data. 1 is first floor, 2 is second floor.
```
python load.py -f {1, 2} 
```

Then, reconstruct the point cloud by ICP
```
python reconstruction.py -f {1, 2} -v {my_icp, open3d}
```

