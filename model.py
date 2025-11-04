import torch.nn as nn

latent_dim = 100 # ノイズzの次元
img_size = 784   # 28x28x1の画像サイズ

# Generator (G)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, img_size),
            nn.Tanh() # 出力を [-1, 1] に正規化
        )
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, 28, 28)

# Discriminator (D)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid() # 確率 [0, 1] を出力
        )
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity