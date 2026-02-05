### 🧩 Problem Statement
**Problem:**  
Unsupervised learning models (like K-Means) group data into clusters, but they do not provide labels or explanations for *why* data points are grouped together. Business stakeholders cannot act on "Cluster 0" or "Cluster 1" without knowing what those clusters represent (e.g., "High Spenders"). The problem is strictly about **interpreting** the mathematical results of clustering into actionable business insights.

**Why it matters:**  
- **Actionability:** Marketing teams need to know *who* to target (e.g., "Send coupons to the budget shoppers").
- **Trust:** Stakeholders trust models they understand. Black-box clusters are often rejected.
- **Revenue:** Correctly identifying high-value segments allows for targeted upsizing, directly impacting revenue.

**Real-world Relevance:**  
Used in Customer Segmentation (Retail), Patient Risk Stratification (Healthcare), and Anomaly Detection (Cybersecurity).

### 🪜 Steps to Solve the Problem
1.  **Load & Preprocess Data:** Clean the data and scale features (Income, Spending Score) so they have equal weight in clustering.
2.  **Apply K-Means Clustering:** Group customers into $K=5$ distinct segments based on similarities.
3.  **Inverse Transform Centroids:** Convert the cluster centers (which are scaled numbers like 1.2, -0.5) back to the original units (e.g., $80k Income, Score 90) to make them readable.
4.  **Profile the Clusters:** Create a summary table showing the average behavior and size of each cluster.
5.  **Visualize:** Use PCA to project the data into 2D and plot the clusters with annotations to separate them visually.
6.  **Interpret & Brief:** Assign business-friendly names to clusters and recommend actions in a stakeholder briefing.

### 🎯 Expected Output (OVERALL)
1.  **Cluster Profile Table:** A CSV/Table showing Cluster ID, Count, Profile (Avg Income, Avg Score).
2.  **Visualizations:** A 2D scatter plot showing distinct groups with labels.
3.  **Stakeholder Briefing:** A plain-English report explaining who the customers are and what to do with them.

---

### 💼 Exam Focus Points
- **Q:** Why do we inverse transform centroids?  
  **A:** To interpret them in the original units (e.g., Dollars) instead of scaled Standard Deviations.
- **Q:** What is a Cluster Profile?  
  **A:** A summary of the average feature values for each cluster.
