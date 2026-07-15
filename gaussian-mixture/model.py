import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

X, y_true = make_blobs(n_samples=600, centers=4, n_features=2, cluster_std=1.1, random_state=42)

model = GaussianMixture(n_components=4, random_state=42)
labels = model.fit_predict(X)

ari = adjusted_rand_score(y_true, labels)
n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
sil = silhouette_score(X, labels) if n_clusters_found > 1 else float("nan")

print("adjusted_rand_index:", round(ari, 4))
print("silhouette:", round(sil, 4))

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15)
plt.title("gaussian mixture")
plt.tight_layout()
plt.savefig("results.png", dpi=120)
