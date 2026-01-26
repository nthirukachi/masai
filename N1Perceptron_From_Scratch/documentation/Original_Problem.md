# Original Problem Statement

Build a perceptron from scratch on a clearly separable dataset and analyse its learning dynamics.

Dataset: Use scikit-learn's make_classification (Colab built-in; docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).

```python
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=600,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    class_sep=1.6,
    random_state=7,
)
```

Tasks:

1. Implement the perceptron training loop using NumPy.
2. Train for at least 40 epochs with shuffling each epoch.
3. Track accuracy per epoch and plot the final decision boundary.
4. Count how many weight updates occurred.

Deliverables: accuracy plot, boundary visualisation, 150-200 word commentary. 

Success criteria: 
- Test accuracy >= 0.95 on a 20 percent holdout
- Plot showing the separating line
- Commentary referencing update count and learning rate impact
