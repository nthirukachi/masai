# Interview Preparation: Robust Clustering Pipeline

## 1. High-Level Project Summary
- **Problem**: Segment 5000 B2B accounts into meaningful groups without labels.
- **Solution**: Built a pipeline with imputation/scaling, compared KMeans/MiniBatch/GMM using Silhouette/Inertia, and verified stability with ARI.
- **Outcome**: Identified 5 stable clusters for targeted marketing.

## 2. Core Concepts – Interview & Exam View

### K-Means
- **What**: Partitions data into K spherical clusters.
- **Why**: Fast, scalable baseline.
- **When**: Numeric data, roughly equal cluster sizes.
- **Trap**: Fails on non-spherical shapes (use DBSCAN there).

### Gaussian Mixture Models (GMM)
- **What**: Probabilistic model (soft clustering) assuming Gaussian distribution.
- **Why**: Allows elliptical clusters and mixed membership.
- **When**: Clusters might overlap or have different variances.

### Pipeline
- **What**: Chains preprocessing steps (fill missing -> scale -> model).
- **Why**: Prevents data leakage and ensures reproducibility (same steps for training/prediction).

## 3. Frequently Asked Interview Questions

### Q1: How did you select K?
**Answer**: I used the **Silhouette Score** because it balances cohesion and separation. I also checked the **Elbow Plot** (Inertia) for confirmation.

### Q2: Why did you scale the data?
**Answer**: Distance metrics like Euclidean distance are sensitive to scale. A feature with range 0-10000 (Revenue) would dominate 0-5 (Logins). `StandardScaler` puts them on equal footing.

### Q3: How do you know the clusters are real and not random?
**Answer**: I performed **Stability Analysis**. I ran the model 5 times with different seeds and calculated the **Adjusted Rand Index (ARI)**. A high ARI (>0.9) proved the clusters were consistent.

## 4. Comparisons (Quick Revision)

| Feature | KMeans | GMM |
| :--- | :--- | :--- |
| **Shape** | Spherical (circles) | Elliptical (ovals) |
| **Assignment** | Hard (belongs to 1) | Soft (probability) |
| **Speed** | Very Fast | Slower |
| **Params** | K | Components |

| Metric | Silhouette | Inertia | ARI |
| :--- | :--- | :--- | :--- |
| **Range** | -1 to 1 | 0 to $\infty$ | -1 to 1 |
| **Goal** | Maximize | Minimize | Maximize |
| **Requires Ground Truth?** | No | No | Yes (or 2 runs) |

## 5. Parameter & Argument Questions

### `random_state`
- **Why**: To ensure results are the same every time we run code.
- **If removed**: K-Means results might change slightly each run due to random initialization.

### `n_init` (in KMeans)
- **Why**: Number of times to run with different centroids.
- **Default (10)**: Prevents getting stuck in local optima.

## 6. Common Mistakes & Traps
- **Mistake**: Forgetting to scale data before K-Means.
- **Mistake**: Using Inertia to compare K-Means vs DBSCAN (Inertia is only valid for K-Means/spherical).
- **Trap**: "Does a high Silhouette score guarantee business value?" -> **No**, it only guarantees mathematical separation. Business context is needed.

## 7. Output Interpretation
- **Scenario**: Stability is 0.4.
- **Meaning**: The clusters are unstable/random.
- **Action**: Don't use them. Try capturing better features or different constraints.
