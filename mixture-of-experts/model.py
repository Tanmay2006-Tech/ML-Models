import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

torch.manual_seed(42)

# Sparse MoE layer - a router picks the top-k experts per input, only those
# experts run, and a load-balancing loss stops training from collapsing
# onto one or two experts (the classic MoE failure mode).


class Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    def __init__(self, in_dim, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        return top_probs, top_idx, probs


class SparseMoE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(in_dim, hidden_dim, out_dim) for _ in range(num_experts)])
        self.router = Router(in_dim, num_experts, top_k)
        self.out_dim = out_dim

    def forward(self, x):
        b = x.shape[0]
        top_probs, top_idx, full_probs = self.router(x)
        output = torch.zeros(b, self.out_dim, device=x.device)

        for e in range(self.num_experts):
            match = top_idx == e
            if not match.any():
                continue
            rows, slots = match.nonzero(as_tuple=True)
            out = self.experts[e](x[rows])
            weights = top_probs[rows, slots].unsqueeze(-1)
            output.index_add_(0, rows, out * weights)

        return output, full_probs, top_idx


def load_balancing_loss(full_probs, top_idx, num_experts):
    top1 = top_idx[:, 0]
    one_hot = F.one_hot(top1, num_classes=num_experts).float()
    fraction_routed = one_hot.mean(dim=0)
    mean_prob = full_probs.mean(dim=0)
    return num_experts * torch.sum(fraction_routed * mean_prob)


def make_data(n=4000, in_dim=10, n_regions=4):
    X = torch.randn(n, in_dim)
    weights = [torch.randn(in_dim) for _ in range(n_regions)]
    biases = [torch.randn(1).item() for _ in range(n_regions)]
    quantiles = torch.quantile(X[:, 0], torch.linspace(0, 1, n_regions + 1))
    region_ids = torch.bucketize(X[:, 0].contiguous(), quantiles[1:-1].contiguous())

    y = torch.zeros(n, 1)
    for r in range(n_regions):
        mask = region_ids == r
        y[mask, 0] = X[mask] @ weights[r] + biases[r]
    y += 0.05 * torch.randn(n, 1)
    return X, y


X, y = make_data()
n_train = int(0.8 * len(X))
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

moe = SparseMoE(in_dim=10, hidden_dim=32, out_dim=1, num_experts=8, top_k=2)
opt = torch.optim.AdamW(moe.parameters(), lr=1e-3)

losses = []
for epoch in range(60):
    opt.zero_grad()
    preds, probs, top_idx = moe(X_train)
    task_loss = F.mse_loss(preds, y_train)
    aux_loss = load_balancing_loss(probs, top_idx, 8)
    total = task_loss + 0.01 * aux_loss
    total.backward()
    opt.step()
    losses.append(task_loss.item())

with torch.no_grad():
    preds, probs, top_idx = moe(X_test)
    test_mse = F.mse_loss(preds, y_test).item()
    usage = Counter(top_idx[:, 0].tolist())
    n_experts_used = len(usage)

print("test mse:", round(test_mse, 4))
print("experts used (of 8):", n_experts_used)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses)
axes[0].set_title("training loss")
axes[0].set_xlabel("epoch")

counts = [usage.get(e, 0) for e in range(8)]
axes[1].bar(range(8), counts)
axes[1].set_title("expert usage on test set (top-1 choice)")
axes[1].set_xlabel("expert id")
axes[1].set_ylabel("count")

plt.tight_layout()
plt.savefig("results.png", dpi=120)
