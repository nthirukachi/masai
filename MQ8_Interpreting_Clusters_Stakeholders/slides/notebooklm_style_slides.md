# Interpreting Clusters for Stakeholders
## MQ8 - Marketing Intelligence Project

---
<!-- slide -->

## 1. Problem Statement
- **The Issue:** K-Means clusters (e.g., "Cluster 2") are meaningless labels to business teams.
- **The Gap:** Stakeholders see "Standard Deviation = 1.2" but do not understand what that means for their budget.
- **The Goal:** Translate mathematical clusters into **human-readable business segments** (e.g., "High Spend, Low Income").

---
<!-- slide -->

## 2. Real-World Use Case
- **Retail:** Sending luxury coupons to "High Spenders" vs. discount codes to "Frugal Shoppers".
- **Banking:** Identifying "High Risk" credit card users who spend more than they earn.
- **Healthcare:** Grouping patients by "Risk Level" based on multiple vitals.

---
<!-- slide -->

## 3. Input Data
- **Dataset:** Mall Customer Data.
- **Key Features:**
    - **Annual Income ($k):** How much they earn.
    - **Spending Score (1-100):** How much they spend.
- **Scale:** Income ranges from 15-137k. Score ranges from 1-100. (Requires Scaling!)

---
<!-- slide -->

## 4. Concepts Used
1.  **Scaling:** Normalizing data so Income doesn't dominate Score.
2.  **K-Means:** Finding the groups.
3.  **Inverse Transformation:** The **magic step** to convert "Z-Scores" back to "$ Dollars".
4.  **Profiling:** Calculating averages per group.

---
<!-- slide -->

## 5. Solution Flow
1.  **Load** Data.
2.  **Scale** (StandardScaler).
3.  **Cluster** (K-Means, K=5).
4.  **Inverse Transform Centroids**.
5.  **Profile** (Create Table).
6.  **Visualize** (PCA).

---
<!-- slide -->

## 6. Code Logic - The Key Lines
- `scaler.fit_transform(X)`:Prepares data.
- `kmeans.fit(X_scaled)`: Finds patterns.
- `scaler.inverse_transform(kmeans.cluster_centers_)`: **Translates patterns to English/Dollars.**

---
<!-- slide -->

## 7. Execution Output (Profile Table)
| Cluster | Income ($) | Score (1-100) | Name |
| :--- | :--- | :--- | :--- |
| **0** | ~88k | ~17 | **Savers** |
| **1** | ~55k | ~50 | **Standard** |
| **2** | ~86k | ~82 | **Target (VIP)** |
| **3** | ~25k | ~79 | **Careless** |
| **4** | ~26k | ~20 | **Sensible** |

---
<!-- slide -->

## 8. Visual Observations
- **Distinct Groups:** The 5 clusters form clear "blobs" in the PCA plot.
- **Separation:** The "Target" and "Careless" groups are very distinct from the "Standard" middle.
- **Business Insight:** We have clearly separable customer bases.

---
<!-- slide -->

## 9. Interview Key Takeaways
- **Q:** Why inverse transform?
- **A:** To interpret centroids in original units.
- **Q:** How do you name clusters?
- **A:** By looking at the profile (Mean) of each feature.
- **Q:** What if clusters overlap?
- **A:** It suggests the features aren't distinctive enough.

---
<!-- slide -->

## 10. Conclusion
- We successfully turned raw data into **5 Actionable Personas**.
- The business can now run **targeted campaigns** (e.g., "VIP Sale" for Cluster 2).
- **Next Steps:** A/B test marketing messages on each group.
