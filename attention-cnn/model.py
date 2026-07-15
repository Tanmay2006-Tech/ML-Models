import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.manual_seed(42)

# CNN with a self-attention block bolted on. Self attention over the
# spatial positions of a feature map, same idea as ViT but dropped into
# a normal conv stack instead of replacing it.


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else None

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


class SelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # starts as identity, learns to use attention

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q = self.q(x).view(b, self.num_heads, self.head_dim, n)
        k = self.k(x).view(b, self.num_heads, self.head_dim, n)
        v = self.v(x).view(b, self.num_heads, self.head_dim, n)

        scores = torch.einsum("bhdn,bhdm->bhnm", q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, h, w)
        out = self.out(out)
        return x + self.gamma * out


class AttentionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = ConvBlock(3, 32)
        self.block1 = ConvBlock(32, 64, pool=True)
        self.attn = SelfAttention2D(64, num_heads=4)
        self.block2 = ConvBlock(64, 128, pool=True)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.attn(x)
        x = self.block2(x)
        return self.head(x)


def load_data():
    digits = load_digits()
    images = digits.images / 16.0  # 8x8, scale to [0,1]
    images = np.repeat(images[:, None, :, :], 3, axis=1)  # fake RGB
    images = torch.tensor(images, dtype=torch.float32)
    images = F.interpolate(images, size=(32, 32), mode="nearest")
    labels = torch.tensor(digits.target, dtype=torch.long)
    return images, labels


X, y = load_data()
n_train = int(0.8 * len(X))
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

model = AttentionCNN()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epoch_losses = []
for epoch in range(15):
    perm = torch.randperm(len(X_train))
    running_loss = 0.0
    n_batches = 0
    for i in range(0, len(X_train), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = criterion(model(X_train[idx]), y_train[idx])
        loss.backward()
        opt.step()
        running_loss += loss.item()
        n_batches += 1
    epoch_losses.append(running_loss / n_batches)

model.eval()
with torch.no_grad():
    acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()

print("test accuracy:", round(acc, 4))

plt.figure(figsize=(6, 4))
plt.plot(epoch_losses)
plt.title("attention-cnn - training loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig("results.png", dpi=120)
