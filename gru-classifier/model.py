import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(42)


def make_sequences(n=800, seq_len=30):
    t = torch.linspace(0, 4 * np.pi, seq_len)
    sequences, labels = [], []
    for i in range(n):
        if i % 2 == 0:
            phase = torch.rand(1).item() * 2 * np.pi
            seq = torch.sin(t + phase) + 0.1 * torch.randn(seq_len)
            labels.append(1)
        else:
            seq = torch.randn(seq_len)
            labels.append(0)
        sequences.append(seq)
    X = torch.stack(sequences).unsqueeze(-1)
    y = torch.tensor(labels)
    return X, y


X, y = make_sequences()
n_train = int(0.8 * len(X))
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]


class GRUClassifier(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


model = GRUClassifier()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

losses = []
for epoch in range(60):
    opt.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    opt.step()
    losses.append(loss.item())

with torch.no_grad():
    acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()

print("test accuracy:", round(acc, 4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses)
axes[0].set_title("training loss")
axes[0].set_xlabel("epoch")

with torch.no_grad():
    sample_idx = 0
    axes[1].plot(X_test[sample_idx].squeeze().numpy())
    label = "sine" if y_test[sample_idx].item() == 1 else "noise"
    axes[1].set_title(f"example test sequence (true label: {label})")

plt.tight_layout()
plt.savefig("results.png", dpi=120)
