# predict.py
import os
import torch
import numpy as np
from PIL import Image
from model import Pix2Vox
from utils.render import voxel_to_mesh, save_mesh
from torchvision import transforms
import argparse

def load_model(checkpoint_path, device):
    model = Pix2Vox().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

def predict_voxel(model, img_tensor, device, threshold=0.5):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        voxel = (pred > threshold).float().cpu().numpy()[0, 0]
        return voxel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image (e.g., .png)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pix2vox.pth", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="reconstructed.obj", help="Output .obj file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Voxel threshold for binarization")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {DEVICE}")

    model = load_model(args.checkpoint, DEVICE)
    img_tensor = preprocess_image(args.image)
    voxel = predict_voxel(model, img_tensor, DEVICE, args.threshold)

    verts, faces = voxel_to_mesh(voxel, threshold=args.threshold)
    save_mesh(verts, faces, args.output)
