import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

data = load_digits()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)

model = TSNE(n_components=2, random_state=42, perplexity=30)
projection = model.fit_transform(X)

print("input shape:", X.shape)
print("output shape:", projection.shape)

plt.figure(figsize=(6, 5))
scatter = plt.scatter(projection[:, 0], projection[:, 1], c=y, cmap="tab10", s=8)
plt.legend(*scatter.legend_elements(), title="digit", loc="best", fontsize=8)
plt.title("tsne")
plt.tight_layout()
plt.savefig("results.png", dpi=120)
