import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as T

class Pix3DVoxelDataset(Dataset):
    def __init__(self, root_dir, json_file, split='train', transform=None):
        """
        Args:
            root_dir (str): Thư mục gốc, ví dụ: 'data'
            json_file (str): Đường dẫn tới file pix3d.json
            split (str): 'train' hoặc 'test'
            transform (callable, optional): Biến đổi ảnh đầu vào
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Load annotation list
        with open(json_file, 'r') as f:
            ann = json.load(f)

        self.samples = []
        for item in ann:
            if item.get("split", "train") != split:
                continue

            img_path = os.path.join(root_dir, item["img"])  # e.g., data/images/0001.png
            voxel_path = os.path.join(root_dir, item["voxel"])  # e.g., data/voxels/IKEA_*.npy
            mask_path = os.path.join(root_dir, item["mask"]) if "mask" in item else None

            if not (os.path.exists(img_path) and os.path.exists(voxel_path)):
                continue

            self.samples.append({
                "img": img_path,
                "mask": mask_path,
                "voxel": voxel_path
            })

        print(f"[Dataset] Loaded {len(self.samples)} samples for split '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["img"]).convert("RGB")

        # Apply mask if available
        if sample["mask"] and os.path.exists(sample["mask"]):
            mask = Image.open(sample["mask"]).convert("L").resize(img.size)
            img = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)

        if self.transform:
            img = self.transform(img)

        # Load voxel grid (.npy)
        voxel = np.load(sample["voxel"], allow_pickle=True).astype(np.float32) # shape: (32, 32, 32)
        voxel = torch.from_numpy(voxel).unsqueeze(0)  # shape: (1, 32, 32, 32)
        print(sample["voxel"])
        return img, voxel
