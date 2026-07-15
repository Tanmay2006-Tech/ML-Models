# ML Models

A collection of ML/AI models I built while learning and practicing. Each folder is one model: a `model.py` you can just run, a `results.txt` with the printed metrics, and a `results.png` with a plot (confusion matrix, predicted vs actual, cluster scatter, training curve, whatever fits the model).

## How to run

```bash
pip install torch scikit-learn numpy matplotlib
cd some-model-folder
python model.py
```

Everything uses small datasets that come bundled with scikit-learn or are generated on the fly, so there's nothing to download.

## The models I actually built from scratch

Most of the folders below are the standard sklearn algorithms, which is normal and not meant to be impressive on its own. These four are different — I implemented the core mechanism myself instead of calling a pre-built class:

- `attention-cnn/` — a CNN with a self-attention block I wrote by hand (query/key/value, scaled dot product, the whole thing)
- `transformer-nlp/` — a transformer encoder built from the actual pieces (multi-head attention, positional encoding) instead of `nn.Transformer`
- `mixture-of-experts/` — a sparse MoE layer with top-k routing and a load-balancing loss, the routing approach used in a lot of large models
- `stacked-ensemble/` — a stacking ensemble where I wrote the out-of-fold prediction logic myself instead of using `StackingRegressor`

## Everything else

Classification: `logistic-regression`, `knn-classifier`, `naive-bayes`, `decision-tree-classifier`, `random-forest-classifier`, `gradient-boosting-classifier`, `svm-classifier`, `mlp-classifier`

Regression: `linear-regression`, `ridge-regression`, `lasso-regression`, `elasticnet-regression`, `knn-regressor`, `decision-tree-regressor`, `random-forest-regressor`, `svr`

Clustering: `kmeans`, `dbscan`, `agglomerative-clustering`, `gaussian-mixture`

Dimensionality reduction: `pca`, `truncated-svd`, `lda`, `tsne`

Deep learning: `autoencoder`, `variational-autoencoder`, `lstm-classifier`, `gru-classifier`

32 models total.

## Notes

- `attention-cnn` is slow on CPU (a few minutes) because of the attention block, everything else runs in seconds.
- Results in `results.txt` are from my machine, yours might differ slightly depending on hardware and random seeds.
- If you want to try a real Kaggle dataset instead of the bundled ones, swap out the `load_...` call near the top of any `model.py` for `pd.read_csv("your_file.csv")`.
