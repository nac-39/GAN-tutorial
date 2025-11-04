# GAN for MNIST

## 概要

このプロジェクトは、PyTorchを使用してMNISTデータセットに対してGAN（Generative Adversarial Network）を実装したものです。

- **Generator（生成器）**: ランダムノイズから28x28ピクセルの数字画像を生成します
- **Discriminator（識別器）**: 入力画像が本物か偽物かを判別します

両方のネットワークを対戦的に訓練することで、本物の数字画像に近い画像を生成できるようになります。

## セットアップ

### 必要な環境

- Python 3.7以上
- PyTorch（CUDA対応GPUがある場合は推奨）

### インストール手順

1. リポジトリをクローンまたはダウンロードします

2. 依存関係をインストールします：

```bash
pip install -r requirements.txt
```
or 
```bash
uv sync
```

必要なパッケージ：
- `torch>=1.9.0`
- `torchvision>=0.10.0`
- `matplotlib>=3.3.0`
- `numpy>=1.21.0`

3. 訓練を実行します：

```bash
python train.py
```
or
```bash
uv run train.py
```

訓練中はMNISTデータセットが自動的にダウンロードされます（初回のみ）。

## 結果の確認方法

### 生成画像の確認

訓練中は、各エポック（10エポックごと）に生成画像が保存されます：

- `generated_images/epoch_X.png`: 各エポックでの生成画像（4x4グリッド、16枚）
- `generated_images/final_generated.png`: 訓練完了後の最終生成画像（8x8グリッド、64枚）

これらのPNGファイルを開いて、生成された数字画像を確認できます。

### 損失グラフの確認

訓練完了後、`training_losses.png`ファイルが生成されます。このファイルには以下が表示されます：

- Generator Loss: 生成器の損失の推移
- Discriminator Loss: 識別器の損失の推移

損失の推移を確認することで、訓練の進行状況を把握できます。

### モデルの保存

訓練完了後、以下のモデルファイルが保存されます：

- `generator.pth`: 訓練済みのGeneratorモデル
- `discriminator.pth`: 訓練済みのDiscriminatorモデル

これらのファイルを使用して、後で生成画像を作成することもできます。
