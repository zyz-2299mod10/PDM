import torch
import numpy as np
import open3d as o3d


def depth_image_to_point_cloud(depth, K, depth_scale=1., device='cuda:0', rgb = None, z_threshold = 1000):
    v, u = np.mgrid[0:depth.shape[0], depth.shape[1]-1:-1:-1]
    u = torch.tensor(u.astype(np.float32)).to(device)
    v = torch.tensor(v.astype(np.float32)).to(device)
    Z = depth / depth_scale
    X = (u - K[0, 2]) * Z / K[0, 0]  # (u-cx) * Z / fx
    Y = (v - K[1, 2]) * Z / K[1, 1]  # (v-cy) * Z / fy

    img_stack = torch.dstack((X, Y, Z))

    X = torch.ravel(X)
    Y = torch.ravel(Y)
    Z = torch.ravel(Z)
    
    # remove points which is too far
    valid = Z < z_threshold
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    dummy = torch.ones_like(Z).to(device)
    position = torch.vstack((X, Y, Z, dummy))

    if rgb is not None:
        R = torch.ravel(rgb[:, :, 0])[valid] / 255.
        G = torch.ravel(rgb[:, :, 1])[valid] / 255.
        B = torch.ravel(rgb[    :, :, 2])[valid] / 255.
        colors = torch.vstack((R, G, B)).transpose(0, 1)
        return position, colors

    return position

def depth_image_to_point_cloud_np(depth, K, depth_scale=1., rgb=None, z_threshold=1000):
    v, u = np.mgrid[0:depth.shape[0], depth.shape[1]-1:-1:-1]
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    
    Z = depth / depth_scale
    X = (u - K[0, 2]) * Z / K[0, 0]  # (u-cx) * Z / fx
    Y = (v - K[1, 2]) * Z / K[1, 1]  # (v-cy) * Z / fy

    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()
    
    valid = Z < z_threshold
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    dummy = np.ones_like(Z)
    position = np.vstack((X, Y, Z, dummy))

    if rgb is not None:
        R = rgb[:, :, 0].ravel()[valid] / 255. 
        G = rgb[:, :, 1].ravel()[valid] / 255.
        B = rgb[:, :, 2].ravel()[valid] / 255.
        colors = np.vstack((R, G, B)).T  
        return position, colors

    return position