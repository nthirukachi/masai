# Original Problem Statement

Evaluate K-Means cluster quality with inertia and silhouette analysis on the iris dataset.

Dataset: Use scikit-learn's load_iris loader (Colab built-in; docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).

from sklearn.datasets import load_iris

Tasks:

    Standardise all four features.
    Run K-Means for k from 2 to 6 with init 'k-means++' and n_init 'auto'.
    Capture inertia and average silhouette score for each k.
    Produce an elbow plot and a silhouette plot for the chosen k.
    Justify the final choice of k in fewer than 200 words using metrics and domain intuition.

Deliverables: metrics table, elbow plot, silhouette plot, written justification. Success criteria: metrics table contains no missing values, plots include annotations, argument clearly states how the chosen k balances cohesion and separation.
