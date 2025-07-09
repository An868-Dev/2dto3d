import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import trimesh


def mesh_to_voxel(obj_path, resolution=32):
    """
    Convert mesh to voxel grid using trimesh.
    """
    mesh = trimesh.load(obj_path, force='mesh')

    # Normalize to unit cube
    mesh.apply_translation(-mesh.bounding_box.centroid)
    scale = max(mesh.bounding_box.extents)
    mesh.apply_scale(1.0 / scale)

    # Voxelization
    voxel = mesh.voxelized(pitch=1.0 / resolution).matrix.astype(np.float32)

    # Pad to exact shape (resolution x resolution x resolution)
    padded = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    s = voxel.shape
    offset = [(resolution - s[i]) // 2 for i in range(3)]
    padded[
    offset[0]:offset[0] + s[0],
    offset[1]:offset[1] + s[1],
    offset[2]:offset[2] + s[2]
    ] = voxel

    return padded


def convert_all(pix3d_root, voxel_out_dir, json_path, resolution=32):
    os.makedirs(voxel_out_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        ann = json.load(f)

    converted = 0
    for item in tqdm(ann):
        if not (item.get("model") and item.get("bbox_valid") and item.get("cad_index")):
            continue

        obj_name = item["model"]
        model_id = item["cad_index"]
        obj_path = os.path.join(pix3d_root, "model", obj_name)

        if not os.path.exists(obj_path):
            continue

        out_path = os.path.join(voxel_out_dir, f"{model_id}.npy")
        if os.path.exists(out_path):
            continue  # skip if already done

        try:
            voxel = mesh_to_voxel(obj_path, resolution)
            np.save(out_path, voxel)
            converted += 1
        except Exception as e:
            print(f"[!] Error with {obj_name}: {e}")

    print(f"✅ Done: {converted} models converted to voxel.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pix3d_root", type=str, default="ytre/data/pix3d", help="Path to Pix3D dataset root")
    parser.add_argument("--json_path", type=str, default="ytre/data/pix3d/pix3d.json", help="Path to pix3d.json file")
    parser.add_argument("--voxel_out", type=str, default="ytre/data/voxels", help="Where to save .npy voxel GT files")
    parser.add_argument("--res", type=int, default=32, help="Voxel resolution")

    args = parser.parse_args()

    convert_all(args.pix3d_root, args.voxel_out, args.json_path, resolution=args.res)