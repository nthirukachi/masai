@[/full-teaching-project] Build an adaptive perceptron for a data stream with mild concept drift.

Dataset: Use scikit-learn's make_classification (Colab built-in; docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).

import numpy as np
from sklearn.datasets import make_classification

def drifting_stream(seed=99):
    rng = np.random.default_rng(seed)
    batches = []
    shifts = [(0.0, 0.0), (0.8, -0.6), (1.2, 0.9)]
    for drift_x, drift_y in shifts:
        X, y = make_classification(
            n_samples=500,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            class_sep=1.2,
            random_state=rng.integers(1000),
        )
        X[:, 0] += drift_x
        X[:, 1] += drift_y
        batches.append((X, y))
    return batches

Tasks:

    Stream batches sequentially and evaluate accuracy on a 200 sample validation buffer after each batch.
    Implement a perceptron that decays the learning rate by 10 percent every five epochs and resets weights only if accuracy drops below 70 percent.
    Log sliding window accuracy with window size 50 and the number of weight resets.
    Analyse how the adaptive schedule copes with drift and where it struggles.

Deliverables: accuracy timeline with reset markers, table of learning rates per epoch, 350-400 word analysis. Success criteria: validation accuracy stays above 0.80 on the final batch, narrative connects drift magnitude to accuracy dips, and any architecture tweaks are justified.
