# ML Models

32 ML/AI models, one per folder. Each folder has `model.py` (runnable, self-contained), `results.txt` (metrics from the last run), and `results.png` (a plot). This README explains what each model actually is and how it's built, not just the name.

## How to run

```bash
pip install torch scikit-learn numpy matplotlib
cd some-model-folder
python model.py
```

No downloads needed — everything uses datasets bundled with scikit-learn or generated on the fly.

---

## Built from scratch

These four are implemented at the mechanism level — I wrote the actual math, not just called a library class. This is the part I'd point to if someone wants to check I understand what's going on under the hood rather than just calling `.fit()`.

### `attention-cnn/` — CNN with self-attention

A convolutional network with a self-attention block dropped in the middle of it, the same mechanism ViTs use, applied inside a normal conv stack instead of replacing it.

- **Stem**: 3x3 conv → batchnorm → ReLU
- **ConvBlock + maxpool**: standard downsampling block
- **SelfAttention2D** (custom): the feature map is flattened to a sequence of spatial positions, projected into query/key/value with 1x1 convs, scaled dot-product attention is computed across all positions (4 heads), then projected back and added to the input through a learnable gate that starts at zero (so the block starts out as identity and gradually learns to use attention)
- **ConvBlock + maxpool**: second downsampling block
- **Head**: global average pool → linear → class logits

Trained on digit images (8x8, upscaled to 32x32, from `sklearn.datasets.load_digits`) with cross-entropy loss and Adam.

### `transformer-nlp/` — Transformer encoder for text classification

A transformer encoder built from the individual pieces, no `nn.Transformer`, no pretrained weights.

- **Embedding + positional encoding**: token embeddings, sinusoidal position encoding added on top (the fixed sin/cos scheme from the original transformer paper, not learned position embeddings)
- **N encoder blocks**, each: multi-head self-attention (manual Q/K/V linear projections, split into heads, scaled dot-product, masked so padding tokens don't get attended to) → residual + layer norm → feed-forward (linear → GELU → linear) → residual + layer norm
- **Pooling + head**: mean-pool over non-padding tokens, linear classifier on top

Trained on a small hand-labeled sentiment dataset (50 movie-review-style sentences, embedded directly in the script).

### `mixture-of-experts/` — Sparse Mixture-of-Experts

The routing architecture behind most large-scale language models today, at small scale.

- **8 experts**, each a small feed-forward network (linear → GELU → linear)
- **Router**: a linear layer + softmax over experts, followed by top-k selection (k=2) — so each input only activates 2 of the 8 experts, not all of them
- **Sparse dispatch**: inputs are grouped by which expert they were routed to, each expert only processes the inputs assigned to it, outputs are recombined weighted by the router's probabilities
- **Load-balancing loss**: an auxiliary loss term that penalizes the router for sending everything to the same one or two experts. Without this term, MoE training reliably collapses onto a small subset of experts — this is the actual reason MoE models need this loss term, not an optional extra

Trained on synthetic regression data built from 4 distinct linear functions selected by input region, so there's a real reason for different experts to specialize.

### `stacked-ensemble/` — Manual stacking ensemble

A stacking ensemble where the out-of-fold prediction logic is written by hand instead of using `sklearn.ensemble.StackingRegressor`.

- **4 base models**: Linear Regression, Random Forest, Gradient Boosting, SVR
- **Out-of-fold predictions**: 5-fold cross-validation, each base model is trained on 4 folds and predicts on the held-out fold — this avoids the meta-learner training on predictions the base models have already seen, which is the part that's easy to get wrong if you're not careful
- **Meta-learner**: a Ridge regression trained on the base models' out-of-fold predictions, learning how much to trust each one

---

## Classification (8 models)

All trained on the breast cancer dataset (`sklearn.datasets.load_breast_cancer`, 30 features, binary diagnosis).

| Model | Folder | What it is |
|---|---|---|
| Logistic Regression | `logistic-regression/` | Linear model, fits a weighted sum of features passed through a sigmoid to estimate class probability |
| K-Nearest Neighbors | `knn-classifier/` | No training step — classifies a point by majority vote among its 7 nearest neighbors in feature space |
| Gaussian Naive Bayes | `naive-bayes/` | Assumes features are conditionally independent given the class and Gaussian-distributed, applies Bayes' rule |
| Decision Tree | `decision-tree-classifier/` | Recursively splits the feature space on the single most informative threshold at each node (max depth 5) |
| Random Forest | `random-forest-classifier/` | 200 decision trees, each trained on a bootstrapped sample with random feature subsets, predictions averaged |
| Gradient Boosting | `gradient-boosting-classifier/` | 150 shallow trees trained sequentially, each one fitting the residual errors of the ones before it |
| Support Vector Machine | `svm-classifier/` | Finds the max-margin separating boundary, using an RBF kernel to handle non-linear separation |
| Multi-Layer Perceptron | `mlp-classifier/` | Feed-forward neural net, two hidden layers (64, 32 units), trained with backprop |

## Regression (8 models)

All trained on the diabetes dataset (`sklearn.datasets.load_diabetes`, predicting disease progression from 10 features).

| Model | Folder | What it is |
|---|---|---|
| Linear Regression | `linear-regression/` | Ordinary least squares — fits a straight-line combination of features to the target |
| Ridge | `ridge-regression/` | Linear regression with an L2 penalty on the coefficients, shrinks weights to reduce overfitting |
| Lasso | `lasso-regression/` | Linear regression with an L1 penalty, can push coefficients to exactly zero (feature selection) |
| ElasticNet | `elasticnet-regression/` | Combines L1 and L2 penalties (Ridge + Lasso) |
| K-Nearest Neighbors | `knn-regressor/` | Predicts by averaging the target value of the 7 nearest neighbors |
| Decision Tree | `decision-tree-regressor/` | Same splitting logic as the classifier, but predicts the mean target value in each leaf |
| Random Forest | `random-forest-regressor/` | 200 regression trees, predictions averaged |
| Support Vector Regression | `svr/` | SVM adapted for regression — fits a function that stays within an epsilon margin of the actual targets, RBF kernel |

## Clustering (4 models)

All run on synthetic 2D blob data (`sklearn.datasets.make_blobs`, 4 true clusters) so the results are visually checkable.

| Model | Folder | What it is |
|---|---|---|
| K-Means | `kmeans/` | Iteratively assigns points to the nearest of 4 centroids, then recomputes centroids, until convergence |
| DBSCAN | `dbscan/` | Density-based — groups points that are closely packed together, marks sparse points as noise, doesn't need a fixed number of clusters |
| Agglomerative Clustering | `agglomerative-clustering/` | Bottom-up hierarchical clustering — starts with every point as its own cluster and merges the closest pairs |
| Gaussian Mixture | `gaussian-mixture/` | Models the data as a mixture of 4 Gaussian distributions, fit with expectation-maximization |

## Dimensionality Reduction (4 models)

All projecting 64-dimensional handwritten digit features (`sklearn.datasets.load_digits`) down to 2D.

| Model | Folder | What it is |
|---|---|---|
| PCA | `pca/` | Finds the directions of maximum variance in the data and projects onto the top 2 |
| Truncated SVD | `truncated-svd/` | Same idea as PCA but works directly on the data matrix via SVD, doesn't require centering — useful for sparse data |
| Linear Discriminant Analysis | `lda/` | Supervised — finds the projection that best separates the known classes, not just the one with the most variance |
| t-SNE | `tsne/` | Non-linear, preserves local neighborhood structure so similar points stay close together in the 2D projection |

## Deep Learning (4 models)

| Model | Folder | What it is |
|---|---|---|
| Autoencoder | `autoencoder/` | Encoder compresses 64-dim digit features to a 16-dim latent code, decoder reconstructs the original — trained to minimize reconstruction error |
| Variational Autoencoder | `variational-autoencoder/` | Like the autoencoder, but the encoder outputs a mean and variance instead of a fixed code, a sample is drawn via the reparameterization trick, and a KL-divergence term keeps the latent space well-structured |
| LSTM Classifier | `lstm-classifier/` | Single-layer LSTM reads a sequence step by step, final hidden state feeds a linear classifier — trained to tell sine waves apart from random noise sequences |
| GRU Classifier | `gru-classifier/` | Same task and setup as the LSTM, using a GRU cell instead (fewer gates, usually faster to train, similar performance) |

---

## Notes

- `attention-cnn` is the slowest one to run (a few minutes on CPU) because of the attention block — everything else finishes in seconds.
- `results.txt` and `results.png` in each folder are from my own run — re-running will give slightly different numbers depending on hardware and random seeds, though I fixed seeds where it mattered.
- To swap in a real dataset (e.g. from Kaggle) instead of the bundled ones, replace the `load_...` / `make_...` call near the top of any `model.py` with `pd.read_csv("your_file.csv")` and adjust the feature/target columns.
