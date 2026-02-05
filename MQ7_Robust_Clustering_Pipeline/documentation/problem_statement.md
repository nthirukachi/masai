# Robust Clustering Evaluation Pipeline

## 🧩 Problem Statement

### Definition
A B2B SaaS company needs to segment its account base to identify usage patterns, churn risks, and upsell opportunities. The marketing and product teams want to group their ~5000 accounts based on behavioral metrics (e.g., login frequency, feature usage, support tickets) but lack a pre-labeled dataset. They need a **strictly reproducible** and **robust** way to find these groups (clusters) and verify that these groups are stable, not just random noise.

### Why the Problem Exists
- **Data Complexity**: SaaS usage data is often high-dimensional and contains missing values (e.g., new accounts haven't used features yet).
- **Unsupervised Nature**: There is no "correct" answer (labels). We don't know if there are 3 types of customers or 10.
- **Stability Risk**: A clustering algorithm might give different results every time it runs (metric instability), leading to poor business decisions.

### Real-World Relevance
- **Churn Prevention**: Identifying "at-risk" clusters allows for targeted intervention.
- **Tiered Pricing**: Grouping "power users" can help design Enterprise tiers.
- **Product Roadmap**: Understanding which features are used together helps in designing new modules.

---

## 🪜 Steps to Solve the Problem

1.  **Synthetic Data Generation**: Since no real data is provided, we simulate a realistic B2B dataset with 5000 accounts and 12 numeric features, introducing missing values to mimic real-world "dirty" data.
2.  **Preprocessing Pipeline**: We build a `scikit-learn` Pipeline to handle missing values (Imputation) and scale features (StandardScaler) to ensure fair distance calculations.
3.  **Model Selection & Comparison**: We test three algorithms: `KMeans`, `MiniBatchKMeans`, and `GaussianMixture` (GMM) across different cluster counts (K=3, 4, 5, 6).
4.  **Metric Evaluation**: We evaluate quality using:
    - **Inertia**: Compactness of clusters.
    - **Silhouette Score**: Separation distance between clusters.
    - **Calinski-Harabasz**: Ratio of dispersion.
5.  **Stability Analysis**: We take the best model and run it 5 times with different random seeds, measuring the **Adjusted Rand Index (ARI)** to ensure the clusters don't drastically change.
6.  **Business Interpretation**: We interpret the final clusters to give meaningful names (e.g., "Power Users", "Dormant Accounts").

---

## 🎯 Expected Output

### Final Deliverables
1.  **Metric Comparison Tables (CSV)**: A clear table showing Inertia, Silhouette, and CH scores for all models and K values.
2.  **Visualizations (PNG)**:
    - Line plots comparing the 3 metrics across K.
    - Stability boxplot showing ARI variance.
3.  **Cluster Summary (CSV)**: A profile of each cluster showing average logins, feature usage, etc.
4.  **Recommendation Slide**: A strategic decision on whether to proceed with segmentation.

### Sample Output Explanation
If the **Silhouette Score** is high (> 0.5) and **Stability (ARI)** is near 1.0, it means the segments are distinct and reliable. If Stability is low (< 0.5), it means the clusters are random and should not be used for business strategy.
