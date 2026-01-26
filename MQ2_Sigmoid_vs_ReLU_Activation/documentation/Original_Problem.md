# Original Problem Statement

Compare sigmoid versus ReLU activations in a shallow neural network on make_moons data.

Dataset: Use scikit-learn's make_moons generator (Colab built-in; docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=800, noise=0.25, random_state=21)
```

Tasks:

1. Split data 70/30 and standardise features.
2. Train two MLPClassifier models with hidden_layer_sizes=(20, 20) using activation 'logistic' and 'relu'.
3. Record loss_curve_, final accuracy, and a confusion matrix for each run.
4. Explain how activation choice affected convergence speed and decision boundaries.

Deliverables: combined loss plot, metric table, confusion matrices, 200-250 word comparison. Success criteria: training finishes within 300 iterations, loss plot compares both runs, commentary links gradient behaviour to observed metrics.
