import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

data = load_digits()
X = torch.tensor(data.data / 16.0, dtype=torch.float32)

n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]


class Autoencoder(nn.Module):
    def __init__(self, in_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, in_dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


model = Autoencoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for epoch in range(60):
    opt.zero_grad()
    recon = model(X_train)
    loss = F.mse_loss(recon, X_train)
    loss.backward()
    opt.step()
    losses.append(loss.item())

with torch.no_grad():
    test_recon = model(X_test)
    test_mse = F.mse_loss(test_recon, X_test).item()

print("test reconstruction mse:", round(test_mse, 4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses)
axes[0].set_title("training loss")
axes[0].set_xlabel("epoch")

n_show = 6
with torch.no_grad():
    sample = X_test[:n_show]
    recon = model(sample)
comparison = torch.cat([sample, recon]).view(2 * n_show, 8, 8)
grid = comparison.view(2, n_show, 8, 8).permute(0, 2, 1, 3).reshape(16, 8 * n_show)
axes[1].imshow(grid, cmap="gray")
axes[1].set_title("top: original, bottom: reconstructed")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("results.png", dpi=120)
