# pix2vox.py

import torch
import torch.nn as nn
import torchvision.models as models


class Pix2Vox(nn.Module):
    def __init__(self, voxel_resolution=32):
        super(Pix2Vox, self).__init__()
        self.res = voxel_resolution

        # Encoder: dùng ResNet18 bỏ lớp FC
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-2]  # bỏ avgpool + fc
        )

        # Decoder: FC + deconv → voxel (1, 32, 32, 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4x4
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8x8
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16x16
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),     # 32x32x32
            nn.Sigmoid()
        )

        # projection fc: 2D → 3D latent vector
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512 * 4 * 4 * 4),  # latent volume
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feat2d = self.encoder(x)  # (B, 512, 7, 7)

        latent = self.fc(feat2d)  # (B, 512*4*4*4)
        latent = latent.view(-1, 512, 4, 4, 4)  # (B, 512, 4, 4, 4)

        voxel = self.decoder(latent)  # (B, 1, 32, 32, 32)
        return voxel
