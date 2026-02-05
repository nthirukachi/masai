# \ud83c\udfa4 Interview Questions

## 1. Basic Concepts

### Q1: What is the Silhouette Score?
- **Child's Answer**: It's a grade from -1 to 1 that tells us how happy a point is in its group. High is good, low is bad.
- **Pro Answer**: A metric used to evaluate the quality of clusters. It measures the mean distance between a sample and all other points in the same cluster (cohesion) versus the mean distance to points in the nearest neighboring cluster (separation).
- **Key Point**: Range is [-1, 1].

### Q2: Why do we need scaling for K-Means?
- **Child's Answer**: If we measure height in centimeters and weight in kilograms, a small change in height looks huge compared to weight. We need them to be fair.
- **Pro Answer**: K-Means uses Euclidean distance. Features with larger magnitudes (like Salary 100,000) will dominate the distance calculation over features with smaller magnitudes (like Age 50). Scaling normalizes them to unit variance or specific range.
- **Diagram**:
  ```mermaid
  graph LR
  A[Unscaled] --> B[Distance dominated by Income]
  C[Scaled] --> D[Distance considers Income and Age equally]
  ```

---

## 2. Advanced / Scenario Based

### Q3: What happens if Silhouette Score is negative?
- **Answer**: It means a point has been assigned to the wrong cluster. closer to a neighboring cluster than its own.
- **Action**: Check for outliers or try a different K.

### Q4: Can we use Silhouette Score for labeled data?
- **Answer**: We *can*, but it's designed for unsupervised learning (unlabeled). For labeled data, we usually use Accuracy, Precision, or Recall. However, Silhouette can still tell us if the classes are well-separated in feature space.

### Q5: Inertia vs Silhouette - which one wins?
- **Scenario**: Inertia says K=10 is best. Silhouette says K=3 is best.
- **Answer**: Usually trust Silhouette (or the Elbow in Inertia). Inertia naturally decreases as K increases, so K=10 will *always* have lower inertia than K=3. Silhouette penalizes overlapping, so it finds the "natural" structure.

---

## 3. Common Mistakes

- **Mistake**: Forgetting to scale data.
- **Mistake**: Interpreting a score of 0.5 as "bad". (0.5 is actually often decent for real-world noisy data).
- **Mistake**: Thinking K-Means finds "True" groups. It only finds *spherical* groups.
