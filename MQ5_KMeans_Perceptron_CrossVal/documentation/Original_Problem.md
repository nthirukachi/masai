# Original Problem Statement

Combine K-Means features with a perceptron and evaluate using stratified cross-validation.

Dataset: Use scikit-learn's load_wine loader (Colab built-in; docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html).

```python
from sklearn.datasets import load_wine
```

Tasks:

1. Standardise features and create a binary label where class 0 is positive.
2. Fit K-Means with k=4 on each training fold only.
3. Augment features with one-hot cluster membership and distances to each centroid.
4. Train a baseline perceptron on original features and an enhanced perceptron on augmented features.
5. Run stratified 5-fold cross-validation capturing accuracy, F1, and average precision.
6. Write an executive summary recommending whether to keep the clustering augmentation for production.

Deliverables: cross-validation metric table, comparison plots, 400-450 word executive summary.

Success criteria: enhanced pipeline improves at least two metrics or provides evidence-based reasons if it does not, and the summary references statistical significance and operational impact.
