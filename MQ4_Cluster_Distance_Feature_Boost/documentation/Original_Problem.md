# Original Problem Statement

Use cluster-distance features to boost a binary perceptron baseline.

Dataset: Use scikit-learn's make_blobs generator (Colab built-in; docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).

```python
from sklearn.datasets import make_blobs
X, cluster_ids = make_blobs(
    n_samples=900,
    centers=3,
    cluster_std=[1.0, 1.2, 1.4],
    random_state=12,
)
y = (cluster_ids == 0).astype(int)
```

Tasks:

1. Standardise X and fit K-Means with k=3.
2. Derive distance-to-centroid features with model.transform.
3. Train two classifiers on a 75/25 split: the baseline perceptron on original features and an enhanced perceptron on original plus distance features.
4. Compare accuracy, precision, recall, and ROC AUC averaged over 5 random splits.

Deliverables: metric table with averaged scores, 200 word explanation of whether distance features helped and why. Success criteria: enhanced model improves at least one key metric by >= 5 percentage points and the explanation references cluster geometry and boundary shifts.
