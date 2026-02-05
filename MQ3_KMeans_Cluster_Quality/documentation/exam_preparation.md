# Exam Preparation: K-Means Cluster Quality Evaluation

---

## Section A: Multiple Choice Questions (MCQ)

### Q1. What does K-Means algorithm minimize?
a) Silhouette score  
b) Inertia (WCSS)  
c) Distance between clusters  
d) Number of iterations  

**Answer: b) Inertia (WCSS)**

**Explanation**: K-Means minimizes the Within-Cluster Sum of Squares (WCSS), which is the sum of squared distances from each point to its assigned centroid.

**Why others are wrong**:
- a) Silhouette is calculated AFTER clustering, not minimized BY K-Means
- c) K-Means focuses on within-cluster distance, not between-cluster
- d) Iterations are just steps to reach the solution

---

### Q2. What is the range of silhouette score?
a) 0 to 1  
b) -1 to 0  
c) -1 to +1  
d) 0 to infinity  

**Answer: c) -1 to +1**

**Explanation**: Silhouette score ranges from -1 (wrong cluster) to +1 (perfect clustering), with 0 indicating boundary points.

**Why others are wrong**:
- a) Misses negative values
- b) Misses positive values
- d) This describes inertia, not silhouette

---

### Q3. Why do we standardize features before K-Means?
a) To make calculations faster  
b) To reduce the number of clusters  
c) To ensure all features contribute equally  
d) To increase inertia  

**Answer: c) To ensure all features contribute equally**

**Explanation**: Features with larger scales would dominate Euclidean distance calculations. Standardization (mean=0, std=1) makes all features comparable.

---

### Q4. What does k-means++ improve over random initialization?
a) Final number of clusters  
b) Speed of convergence  
c) Initial centroid placement  
d) All of the above  

**Answer: c) Initial centroid placement**

**Explanation**: k-means++ spreads initial centroids apart using probability proportional to D², leading to better starting points and often faster convergence.

**Note**: While it can lead to faster convergence (b), that's a consequence, not the direct improvement.

---

### Q5. In the Elbow Method, we look for:
a) The point with lowest inertia  
b) The point with highest silhouette  
c) The point where curve bends sharply  
d) The point with most clusters  

**Answer: c) The point where curve bends sharply**

**Explanation**: The "elbow" is where adding more clusters gives diminishing returns - the curve transitions from steep to gradual decline.

---

### Q6. What does a negative silhouette score for a point indicate?
a) The point is at the cluster center  
b) The point may be in the wrong cluster  
c) The clustering is perfect  
d) More clusters are needed  

**Answer: b) The point may be in the wrong cluster**

**Explanation**: Negative silhouette means the point is closer to another cluster than its own - likely misassigned.

---

### Q7. What is the formula for standardization?
a) z = x / mean  
b) z = (x - min) / (max - min)  
c) z = (x - mean) / std  
d) z = x * std  

**Answer: c) z = (x - mean) / std**

**Explanation**: StandardScaler uses the z-score formula: subtract mean, divide by standard deviation.

---

### Q8. Which is TRUE about K-Means?
a) It always finds the global minimum  
b) It requires specifying k beforehand  
c) It works best with non-spherical clusters  
d) It can handle categorical data directly  

**Answer: b) It requires specifying k beforehand**

**Explanation**: K-Means needs k as input. It finds local (not global) minimum, works best with spherical clusters, and requires numeric data.

---

### Q9. What does n_init parameter control in KMeans?
a) Number of clusters  
b) Number of iterations  
c) Number of times algorithm runs with different seeds  
d) Number of features  

**Answer: c) Number of times algorithm runs with different seeds**

**Explanation**: n_init specifies how many times K-Means runs with different centroid initializations. The best result is kept.

---

### Q10. As k increases, inertia:
a) Always increases  
b) Always decreases  
c) Stays the same  
d) Increases then decreases  

**Answer: b) Always decreases**

**Explanation**: More clusters mean points are closer to their centroids. At k=n (each point its own cluster), inertia=0.

---

### Q11. What is WCSS?
a) Weekly Customer Satisfaction Score  
b) Within-Cluster Sum of Squares  
c) Weighted Cluster Similarity Score  
d) Wide Cluster Selection Strategy  

**Answer: b) Within-Cluster Sum of Squares**

**Explanation**: WCSS = Inertia = sum of squared distances from points to their cluster centroids.

---

### Q12. Which metric considers BOTH cluster cohesion AND separation?
a) Inertia  
b) WCSS  
c) Silhouette Score  
d) Mean Squared Error  

**Answer: c) Silhouette Score**

**Explanation**: Silhouette = (b-a)/max(a,b) considers distance within cluster (a) AND distance to nearest cluster (b).

---

## Section B: Multiple Select Questions (MSQ)

### Q1. Which are valid values for the 'init' parameter in KMeans? (Select all that apply)
- [ ] a) 'k-means++'
- [ ] b) 'random'
- [ ] c) 'kmeans'
- [ ] d) ndarray of initial centroids

**Correct: a, b, d**

**Explanations**:
- a) k-means++ is the default and recommended method
- b) random picks k random points as initial centroids
- d) You can provide your own initial centroid positions as an array
- c) 'kmeans' is not a valid option

---

### Q2. When should you use StandardScaler? (Select all that apply)
- [ ] a) Before K-Means clustering
- [ ] b) Before Random Forest
- [ ] c) Before Neural Networks
- [ ] d) Before KNN

**Correct: a, c, d**

**Explanations**:
- a) K-Means uses distance - needs scaling
- b) Tree-based methods don't need scaling (splits are independent of scale)
- c) Neural networks benefit from normalized inputs
- d) KNN uses distance - needs scaling

---

### Q3. Which metrics help choose optimal k? (Select all that apply)
- [ ] a) Inertia (Elbow Method)
- [ ] b) Silhouette Score
- [ ] c) Accuracy Score
- [ ] d) Gap Statistic

**Correct: a, b, d**

**Explanations**:
- a) Elbow method uses inertia to find k
- b) Silhouette measures cluster quality
- c) Accuracy requires labels - not for unsupervised clustering
- d) Gap statistic compares to random clustering

---

### Q4. Which are limitations of K-Means? (Select all that apply)
- [ ] a) Assumes spherical clusters
- [ ] b) Sensitive to outliers
- [ ] c) Must specify k beforehand
- [ ] d) Cannot handle numerical data

**Correct: a, b, c**

**Explanations**:
- a) Uses Euclidean distance, assumes round clusters
- b) Mean is affected by extreme values
- c) k must be provided as parameter
- d) FALSE - K-Means ONLY handles numerical data

---

### Q5. Which of the following will change the K-Means clustering result? (Select all that apply)
- [ ] a) Different random_state
- [ ] b) Different n_init
- [ ] c) Feature scaling
- [ ] d) Changing feature names

**Correct: a, b, c**

**Explanations**:
- a) Different seed = different initial centroids
- b) More runs = potentially better result
- c) Scaling changes distances
- d) Names don't affect calculations

---

## Section C: Numerical Problems

### Q1. Calculate Inertia

Given 3 points in Cluster 1 with centroid at (0, 0):
- Point A: (1, 0)
- Point B: (0, 2)
- Point C: (-1, -1)

Calculate the inertia for Cluster 1.

**Solution**:
Step 1: Calculate squared distance for each point

- Point A: (1-0)² + (0-0)² = 1 + 0 = 1
- Point B: (0-0)² + (2-0)² = 0 + 4 = 4
- Point C: (-1-0)² + (-1-0)² = 1 + 1 = 2

Step 2: Sum all squared distances
Inertia = 1 + 4 + 2 = **7**

---

### Q2. Calculate Silhouette Score

For a point with:
- a (mean distance to own cluster) = 2
- b (mean distance to nearest other cluster) = 8

Calculate silhouette score.

**Solution**:
s = (b - a) / max(a, b)
s = (8 - 2) / max(2, 8)
s = 6 / 8
s = **0.75**

**Interpretation**: Good clustering (close to +1)

---

### Q3. Standardize a Feature

Feature values: [10, 20, 30, 40, 50]

Calculate standardized values.

**Solution**:
Step 1: Calculate mean
μ = (10 + 20 + 30 + 40 + 50) / 5 = 150 / 5 = **30**

Step 2: Calculate standard deviation
Variance = [(10-30)² + (20-30)² + (30-30)² + (40-30)² + (50-30)²] / 5
         = [400 + 100 + 0 + 100 + 400] / 5 = 1000 / 5 = 200
σ = √200 = **14.14**

Step 3: Standardize each value
- 10: (10-30)/14.14 = -1.41
- 20: (20-30)/14.14 = -0.71
- 30: (30-30)/14.14 = 0.00
- 40: (40-30)/14.14 = +0.71
- 50: (50-30)/14.14 = +1.41

**Answer**: [-1.41, -0.71, 0.00, +0.71, +1.41]

---

### Q4. Interpret Elbow Plot Data

Given:
| k | Inertia |
|---|---------|
| 2 | 500 |
| 3 | 300 |
| 4 | 250 |
| 5 | 240 |
| 6 | 235 |

What is the optimal k?

**Solution**:
Calculate decrease rates:
- k=2→3: 500-300 = 200 (large drop)
- k=3→4: 300-250 = 50 (smaller)
- k=4→5: 250-240 = 10 (tiny)
- k=5→6: 240-235 = 5 (tiny)

The "elbow" is at k=3 where:
- Sharp drop before (200)
- Gradual after (50, 10, 5)

**Answer**: k = **3**

---

### Q5. Compare Silhouette Scores

| k | Silhouette |
|---|------------|
| 2 | 0.58 |
| 3 | 0.48 |
| 4 | 0.39 |

Which k gives best cluster separation?

**Solution**:
Higher silhouette = better clustering

k=2 has highest silhouette (0.58)

BUT if domain knowledge says 3 clusters exist, choose k=3.

**Answer**: 
- Based on silhouette alone: k = **2**
- With domain knowledge: k = **3** (if applicable)

---

## Section D: Fill in the Blanks

### Q1. K-Means algorithm minimizes _____________.
**Answer**: Inertia (or WCSS, Within-Cluster Sum of Squares)

---

### Q2. The range of silhouette score is _____________ to _____________.
**Answer**: -1 to +1

---

### Q3. StandardScaler transforms features to have mean = _____________ and std = _____________.
**Answer**: 0, 1

---

### Q4. k-means++ initialization selects centroids with probability proportional to _____________.
**Answer**: D² (squared distance to nearest centroid)

---

### Q5. The Elbow Method plots _____________ against k.
**Answer**: Inertia (or WCSS)

---

### Q6. In silhouette formula s = (b-a)/max(a,b), 'a' represents _____________.
**Answer**: Mean distance to other points in the same cluster

---

### Q7. The Iris dataset has _____________ samples and _____________ features.
**Answer**: 150, 4

---

### Q8. If inertia keeps decreasing but silhouette drops after k=3, optimal k is likely _____________.
**Answer**: 3

---

### Q9. K-Means converges to a _____________ minimum, not a global minimum.
**Answer**: local

---

### Q10. The parameter _____________ in KMeans ensures reproducible results.
**Answer**: random_state
