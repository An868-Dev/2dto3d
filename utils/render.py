# render.py
import os
import numpy as np
import mcubes
import trimesh


def voxel_to_mesh(voxel, threshold=0.5):
    """
    Convert voxel grid (numpy array) to mesh using Marching Cubes.
    """
    if isinstance(voxel, np.ndarray):
        voxel = voxel.squeeze()  # remove channel if exists
    else:
        raise ValueError("voxel must be a numpy ndarray")

    # Apply Marching Cubes
    vertices, triangles = mcubes.marching_cubes(voxel, threshold)

    # Normalize vertices to [-0.5, 0.5] range
    vertices = (vertices / voxel.shape[0]) - 0.5

    return vertices, triangles


def save_mesh(vertices, triangles, out_path):
    """
    Save mesh to .obj using trimesh.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.export(out_path)
    print(f"✅ Saved mesh to {out_path}")


def render_voxel_file(voxel_path, save_path, threshold=0.5):
    """
    Convert voxel .npy file to .obj mesh
    """
    voxel = np.load(voxel_path).astype(np.float32)
    verts, faces = voxel_to_mesh(voxel, threshold)
    save_mesh(verts, faces, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel", type=str, required=True, help="Path to voxel .npy file")
    parser.add_argument("--out", type=str, default="output.obj", help="Output .obj path")
    parser.add_argument("--th", type=float, default=0.5, help="Threshold for marching cubes")
    args = parser.parse_args()

    render_voxel_file(args.voxel, args.out, args.th)
