import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from model import Generator, Discriminator, latent_dim, img_size

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ハイパーパラメータ
batch_size = 64
num_epochs = 200
lr = 0.0002
beta1 = 0.5

# モデルの初期化
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 損失関数とオプティマイザー
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# MNISTデータセットの読み込み
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [-1, 1]に正規化
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

# 損失の記録用
G_losses = []
D_losses = []

# 生成画像保存用ディレクトリ
os.makedirs('generated_images', exist_ok=True)

print("訓練開始...")

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size_actual = real_images.size(0)
        real_images = real_images.to(device)
        
        # ラベルの設定
        real_labels = torch.ones(batch_size_actual, 1).to(device)
        fake_labels = torch.zeros(batch_size_actual, 1).to(device)
        
        # ================================
        # Discriminatorの訓練
        # ================================
        optimizer_D.zero_grad()
        
        # 本物の画像での損失
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)
        
        # 偽の画像での損失
        noise = torch.randn(batch_size_actual, latent_dim).to(device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        
        # Discriminatorの総損失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # ================================
        # Generatorの訓練
        # ================================
        optimizer_G.zero_grad()
        
        # Generatorは本物と判定されるように訓練
        noise = torch.randn(batch_size_actual, latent_dim).to(device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        # 損失の記録
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        
        # 進捗表示
        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                  f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
    
    # 各エポック終了時に生成画像を保存
    if epoch % 10 == 0:
        with torch.no_grad():
            # 固定ノイズで生成画像を作成
            fixed_noise = torch.randn(16, latent_dim).to(device)
            generated_images = generator(fixed_noise)
            
            # 画像を保存
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i in range(16):
                row, col = i // 4, i % 4
                img = generated_images[i].cpu().squeeze().numpy()
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'generated_images/epoch_{epoch}.png')
            plt.close()
            
            print(f'生成画像を保存しました: generated_images/epoch_{epoch}.png')

# 訓練完了
print("訓練完了!")

# 最終的な生成画像を保存
with torch.no_grad():
    fixed_noise = torch.randn(64, latent_dim).to(device)
    final_images = generator(fixed_noise)
    
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for i in range(64):
        row, col = i // 8, i % 8
        img = final_images[i].cpu().squeeze().numpy()
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_images/final_generated.png')
    plt.close()

# 損失の可視化
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig('training_losses.png')
plt.show()

# モデルの保存
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
print("モデルを保存しました: generator.pth, discriminator.pth")
