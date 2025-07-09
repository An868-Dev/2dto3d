# voxel_utils.py

import numpy as np
import mcubes
import torch


def normalize_voxel(voxel):
    """
    Normalize voxel to [0, 1] range (if needed)
    """
    voxel = voxel.astype(np.float32)
    if voxel.max() > 1.0:
        voxel /= voxel.max()
    return voxel


def tensor_to_numpy(voxel_tensor):
    """
    Convert torch tensor (1, D, H, W) or (D, H, W) to numpy array
    """
    if isinstance(voxel_tensor, torch.Tensor):
        voxel_np = voxel_tensor.detach().cpu().numpy()
    else:
        voxel_np = voxel_tensor
    return voxel_np.squeeze()


def voxel_to_mesh(voxel_np, threshold=0.5):
    """
    Convert voxel (numpy) to mesh using marching cubes
    Returns: vertices, faces
    """
    voxel_np = normalize_voxel(voxel_np)
    verts, faces = mcubes.marching_cubes(voxel_np, threshold)

    # Normalize coordinates to [-0.5, 0.5]
    verts = (verts / voxel_np.shape[0]) - 0.5
    return verts, faces


def save_obj(filename, verts, faces):
    """
    Save mesh to .obj file manually (no trimesh required)
    """
    with open(filename, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            # OBJ format uses 1-based indexing
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    print(f"✅ Saved OBJ to {filename}")

def save_voxel_as_obj(voxel, path, threshold=0.5):
    """
    Save voxel grid (3D numpy array) to .obj file.
    Each active voxel will be converted to a vertex point.

    Args:
        voxel: 3D numpy array (shape: D x H x W)
        path: output .obj file path
        threshold: minimum voxel value to be considered "filled"
    """
    with open(path, 'w') as f:
        cube_size = 1.0 / voxel.shape[0]  # normalize to unit cube

        for x in range(voxel.shape[0]):
            for y in range(voxel.shape[1]):
                for z in range(voxel.shape[2]):
                    if voxel[x, y, z] > threshold:
                        f.write(f"v {x * cube_size} {y * cube_size} {z * cube_size}\n")