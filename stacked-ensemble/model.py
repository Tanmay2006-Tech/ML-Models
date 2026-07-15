import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Manual stacking: get out-of-fold predictions from each base model,
# then fit a Ridge meta-learner on those predictions. Written by hand
# instead of using StackingRegressor so the mechanics are visible.

data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

base_models = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
    "svr": SVR(kernel="rbf", C=10, epsilon=0.1),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
names = list(base_models.keys())
oof_preds = np.zeros((len(X_train), len(names)))
test_preds = np.zeros((len(X_test), len(names)))

for i, name in enumerate(names):
    model = base_models[name]
    fold_test_preds = np.zeros((len(X_test), kf.n_splits))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        m = type(model)(**model.get_params())
        m.fit(X_train[tr_idx], y_train[tr_idx])
        oof_preds[val_idx, i] = m.predict(X_train[val_idx])
        fold_test_preds[:, fold] = m.predict(X_test)

    test_preds[:, i] = fold_test_preds.mean(axis=1)

meta = Ridge(alpha=1.0)
meta.fit(oof_preds, y_train)
final_preds = meta.predict(test_preds)

rmse = np.sqrt(mean_squared_error(y_test, final_preds))
r2 = r2_score(y_test, final_preds)

print("stacked ensemble rmse:", round(rmse, 4))
print("stacked ensemble r2:", round(r2, 4))
for name, weight in zip(names, meta.coef_):
    print(f"meta-learner weight ({name}):", round(weight, 4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(y_test, final_preds, alpha=0.6)
lims = [min(y_test.min(), final_preds.min()), max(y_test.max(), final_preds.max())]
axes[0].plot(lims, lims, "r--")
axes[0].set_xlabel("actual")
axes[0].set_ylabel("predicted")
axes[0].set_title("stacked ensemble - predicted vs actual")

axes[1].bar(names, meta.coef_)
axes[1].set_title("meta-learner weights per base model")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("results.png", dpi=120)
