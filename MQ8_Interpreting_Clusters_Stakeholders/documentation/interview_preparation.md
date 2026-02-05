# Interview Preparation

## 1. High-Level Project Summary
- **Problem:** Unsupervised clusters (K-Means) lack business labels, making them hard to act upon.
- **Solution:** We used K-Means to segment customers, Inverse Transformed the centroids to original units ($), and named the clusters based on their spending profiles.
- **Outcome:** Identified 5 distinct segments (e.g., "Target Group", "Sensible Shoppers") to guide marketing strategy.

## 2. Core Concepts – Interview & Exam View
### K-Means Clustering
- **What:** A partition-based clustering algorithm.
- **Why:** To group similar data points without labels.
- **When:** When you have numeric data and know K.
- **When NOT:** When clusters are non-globular (circles) or have varying densities (use DBSCAN).

### Inverse Transformation
- **What:** Converting scaled data (Z-scores) back to original units.
- **Why:** Stakeholders don't speak "Standard Deviations"; they speak Dollars.
- **Code:** `scaler.inverse_transform(centroids)`

## 3. Frequently Asked Interview Questions
### Q1: How did you determine what distinct clusters mean?
**A:** I examined the cluster centroids. I inverse-transformed them to see the average Income and Spending Score. For example, a cluster with High Income ($88k) and High Score (82) represents our "Target Customers".

### Q2: Why did you scale the data before clustering?
**A:** K-Means uses Euclidean distance. Without scaling, Income (range 15-137k) would dominate Spending Score (1-100), effectively reducing the model to just "Income Clustering". Scaling gives them equal weight.

### Q3: What if the clusters overlap in the visualization?
**A:** 2D PCA plots are projections. Overlap in 2D doesn't mean overlap in high-dimensional space. However, if they severely overlap, K-Means might not be the best model, or the features aren't discriminative enough.

## 4. Comparisons
| Feature | K-Means | Hierarchical | DBSCAN |
| :--- | :--- | :--- | :--- |
| **Shape** | Spherical | Any | Arbitrary |
| **Parameters** | K (Clusters) | Distance Metric | Epsilon, MinPts |
| **Scalability** | Fast ($O(n)$) | Slow ($O(n^2)$) | Moderate |
| **Noise** | Sensitive to Outliers | Sensitive | Handles Noise |

## 5. Common Mistakes
- **Forgetting to Scale:** Leads to biased clusters.
- **Forgetting to Inverse Transform:** attempting to interpret "0.5 Income" is confusing.
- **Over-interpretation:** Assigning a complex story to a cluster that is just "Average people".

## 6. One-Page Quick Revision
- **Goal:** Interpret clusters for business.
- **Key Step:** `scaler.inverse_transform(kmeans.cluster_centers_)`
- **Visualization:** PCA to reduce dimensions to 2D.
- **Key Insight:** Look for extremes (High/High, Low/Low) to find niche groups.
