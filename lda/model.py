import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = load_digits()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)

model = LinearDiscriminantAnalysis(n_components=2)
projection = model.fit_transform(X, y)

print("input shape:", X.shape)
print("output shape:", projection.shape)

plt.figure(figsize=(6, 5))
scatter = plt.scatter(projection[:, 0], projection[:, 1], c=y, cmap="tab10", s=8)
plt.legend(*scatter.legend_elements(), title="digit", loc="best", fontsize=8)
plt.title("lda")
plt.tight_layout()
plt.savefig("results.png", dpi=120)
