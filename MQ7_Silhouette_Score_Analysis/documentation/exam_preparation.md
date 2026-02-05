# \ud83d\udcdd Exam Preparation

## Section A: Multiple Choice (MCQ)

**1. What is the range of the Silhouette Score?**
- A) 0 to 1
- B) -1 to 1
- C) 0 to 100
- D) -10 to 10
> **Correct Answer**: B.
> **Reason**: -1 means wrong cluster, 0 means overlapping, +1 means perfect.

**2. Which library do we use for K-Means in Python?**
- A) pandas
- B) numpy
- C) sklearn
- D) matplotlib
> **Correct Answer**: C.
> **Reason**: sklearn (scikit-learn) contains the machine learning algorithms.

**3. If Inertia decreases, does it guarantee a better model?**
- A) Yes, always.
- B) No, not necessarily.
- C) Only if K < 3.
- D) Only if K > 10.
> **Correct Answer**: B.
> **Reason**: Inertia decreases by definition as K increases. It doesn't mean the clusters are meaningful (e.g., K=N has 0 inertia but is useless).

---

## Section B: Multiple Select (MSQ)

**4. Which of the following are distance-based algorithms? (Select all that apply)**
- [x] K-Means
- [x] K-Nearest Neighbors (KNN)
- [ ] Decision Trees
- [x] SVM (Support Vector Machines)
> **Explanation**: Trees split data by rules (if x > 5). The others calculate distance between points in space.

**5. How can we improve Silhouette Score? (Select all that apply)**
- [x] Scale the data
- [x] Remove outliers
- [x] Select better features
- [ ] Just increase K to infinity
> **Explanation**: Increasing K blindly usually lowers the score after a certain peak. Preprocessing is key.

---

## Section C: Numerical & Fill in the Blanks

**6. Formula Check**
The Silhouette Score $s$ is calculated using $a$ (mean intra-cluster distance) and $b$ (mean nearest-cluster distance).
Formula: $s = \frac{b - a}{\max(a, b)}$

**7. Calculate**
Point P has:
- $a = 2$ (dist to own group)
- $b = 8$ (dist to next group)
What is the score?
> **Solution**:
> $s = (8 - 2) / \max(2, 8)$
> $s = 6 / 8$
> $s = 0.75$ (A very good score!)

**8. Fill in the Blank**
MinMaxScaler scales data to the range [___, ___].
> **Answer**: [0, 1]
