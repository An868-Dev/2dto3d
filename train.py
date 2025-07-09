import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import Pix2Vox  # nếu file là model.py
from utils.dataset import Pix3DVoxelDataset

# === Config ===
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")

# === Data transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Dataset ===
dataset = Pix3DVoxelDataset(
    root_dir='data',
    json_file='data/pix3d.json',
    split='train',
    transform=transform
)

if len(dataset) == 0:
    raise ValueError("Dataset is empty. Vui lòng kiểm tra lại pix3d.json, ảnh hoặc voxel.")

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True  # ⚡ tăng tốc copy từ RAM -> VRAM
)

# === Model ===
model = Pix2Vox().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Mixed precision training (optional) ===
scaler = torch.cuda.amp.GradScaler()  # AMP for faster training

# === Training ===
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, voxels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        voxels = voxels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # === Mixed Precision ===
        with torch.cuda.amp.autocast():  # AMP mode
            preds = model(imgs)

            # Reshape nếu output bị flatten (thường là [B * 1, 128, 128, 128])
            if preds.shape[0] != voxels.shape[0]:
                preds = preds.view(voxels.shape)

            # Resize nếu chưa đúng resolution
            if preds.shape[-1] != voxels.shape[-1]:
                preds = torch.nn.functional.interpolate(
                    preds, size=voxels.shape[-3:], mode='trilinear', align_corners=True
                )

            loss = criterion(preds, voxels)

        if torch.isnan(loss):
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f}")

# === Save checkpoint ===
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/pix2vox.pth')
print("✅ Saved to checkpoints/pix2vox.pth")
