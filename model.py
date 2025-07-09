import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)  # 128x128 -> 64x64
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv5 = nn.Conv2d(512, 400, 4, stride=2, padding=1)  # 8x8 -> 4x4

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(1, 32, 4, stride=2, padding=1)  # 16x16x16
        self.deconv2 = nn.ConvTranspose3d(32, 64, 4, stride=2, padding=1)  # 32x32x32
        self.deconv3 = nn.ConvTranspose3d(64, 1, 1)  # output 32x32x32

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 1, 25, 28, 28)  # vì 1×25×28×28 = 19600 per sample → 19600×8 = 156800
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)  # ❗️Không dùng sigmoid ở đây
        return x


class Pix2Vox(nn.Module):
    def __init__(self):
        super(Pix2Vox, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        feat = self.encoder(x)
        print(f"[DEBUG] Encoder output shape: {feat.shape}")
        out = self.decoder(feat)
        return out
