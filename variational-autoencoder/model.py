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


class VAE(nn.Module):
    def __init__(self, in_dim=64, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, in_dim))

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


model = VAE()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for epoch in range(60):
    opt.zero_grad()
    recon, mu, logvar = model(X_train)
    loss = vae_loss(recon, X_train, mu, logvar) / X_train.size(0)
    loss.backward()
    opt.step()
    losses.append(loss.item())

with torch.no_grad():
    recon, mu, logvar = model(X_test)
    test_loss = (vae_loss(recon, X_test, mu, logvar) / X_test.size(0)).item()

print("test loss (recon + kl):", round(test_loss, 4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses)
axes[0].set_title("training loss")
axes[0].set_xlabel("epoch")

n_show = 6
with torch.no_grad():
    sample = X_test[:n_show]
    recon, _, _ = model(sample)
comparison = torch.cat([sample, recon]).view(2 * n_show, 8, 8)
grid = comparison.view(2, n_show, 8, 8).permute(0, 2, 1, 3).reshape(16, 8 * n_show)
axes[1].imshow(grid, cmap="gray")
axes[1].set_title("top: original, bottom: reconstructed")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("results.png", dpi=120)
