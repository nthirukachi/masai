# Exam Preparation: Cluster-Distance Feature Boost

---

## Section A: Multiple Choice Questions (MCQ)

### Q1. What does the K-Means `transform()` method return?
- A) Cluster labels for each point
- B) Cluster centers (centroids)
- C) **Distances from each point to each centroid** ✓
- D) Prediction probabilities

**Explanation:** `transform()` computes Euclidean distances, while `predict()` returns labels.

---

### Q2. Why should you NOT use `fit_transform()` on test data?
- A) It's slower than transform()
- B) **It causes data leakage** ✓
- C) It only works on training data
- D) It changes the random state

**Explanation:** Fitting on test data lets test statistics influence the model.

---

### Q3. What is the main purpose of StandardScaler?
- A) Remove outliers
- B) **Transform features to mean=0, std=1** ✓
- C) Reduce dimensionality
- D) Cluster the data

**Explanation:** StandardScaler standardizes features using z-score transformation.

---

### Q4. If baseline accuracy is 57% and enhanced is 92%, what is the improvement?
- A) 35 percentage points
- B) 61%
- C) **35 percentage points** ✓
- D) 57%

**Explanation:** 92% - 57% = 35 percentage points (not 61% which would be relative improvement).

---

### Q5. What classifier is used as the baseline in this project?
- A) Logistic Regression
- B) Decision Tree
- C) **Perceptron** ✓
- D) SVM

---

### Q6. How many distance features are created with k=3 clusters?
- A) 1
- B) 2
- C) **3** ✓
- D) 6

**Explanation:** One distance feature per centroid.

---

### Q7. What does ROC AUC = 0.5 indicate?
- A) Perfect classifier
- B) **Random guessing** ✓
- C) Always predicts positive
- D) Always predicts negative

---

### Q8. What is the formula for StandardScaler transformation?
- A) (x - min) / (max - min)
- B) **z = (x - μ) / σ** ✓
- C) x / max(x)
- D) log(x)

---

### Q9. Why does the enhanced model perform better?
- A) It uses more training data
- B) **Distance features capture cluster geometry** ✓
- C) The Perceptron learns non-linearly
- D) StandardScaler improves accuracy

---

### Q10. What makes cluster 0 easier to identify?
- A) It has more points
- B) **It has the smallest standard deviation (tightest cluster)** ✓
- C) It's at the origin
- D) It's randomly selected

---

## Section B: Multiple Select Questions (MSQ)

### Q1. Which of the following are TRUE about K-Means? (Select all)
- [x] **It minimizes within-cluster variance**
- [x] **It requires specifying k in advance**
- [ ] It can find non-spherical clusters
- [x] **Initial centroid placement affects results**
- [ ] It guarantees the global optimum

---

### Q2. Which preprocessing steps are used in this project? (Select all)
- [x] **StandardScaler**
- [ ] PCA
- [x] **K-Means distance features**
- [ ] One-hot encoding
- [x] **Train-test split**

---

### Q3. Which metrics improved by more than 30%? (Select all)
- [x] **Accuracy (34.76%)**
- [x] **Precision (58.01%)**
- [ ] Recall (27.93%)
- [x] **ROC AUC (49.50%)**

---

### Q4. When should you use StandardScaler? (Select all)
- [x] **Before K-Means**
- [x] **Before Perceptron**
- [ ] Before Decision Tree
- [x] **Before SVM**
- [ ] Before Random Forest

---

### Q5. Which of these can cause data leakage? (Select all)
- [x] **Fitting scaler on all data before splitting**
- [x] **Using test data for hyperparameter tuning**
- [x] **Fitting K-Means on combined train+test**
- [ ] Using random_state for reproducibility
- [ ] Averaging metrics over multiple splits

---

## Section C: Numerical Questions

### Q1. If X has shape (900, 2) and we add K-Means distances with k=3, what is the new shape?
**Answer:** (900, 5)

**Calculation:** 2 original features + 3 distance features = 5 features

---

### Q2. With 900 samples and 75/25 split, how many test samples are there?
**Answer:** 225

**Calculation:** 900 × 0.25 = 225

---

### Q3. If precision = 0.8967 and we predict 100 positives, how many are true positives?
**Answer:** ~90 (89.67)

**Calculation:** TP = Precision × Predicted Positives = 0.8967 × 100 ≈ 90

---

### Q4. If the Euclidean distance from point (3, 4) to origin (0, 0) is calculated, what is the value?
**Answer:** 5

**Calculation:** √(3² + 4²) = √(9 + 16) = √25 = 5

---

### Q5. With 3 clusters and cluster_std = [1.0, 1.2, 1.4], which cluster is most spread out?
**Answer:** Cluster 2 (index 2)

**Explanation:** Highest std = 1.4 means most spread.

---

## Section D: Fill in the Blanks

### Q1. The K-Means algorithm assigns each point to the _______ centroid.
**Answer:** nearest

---

### Q2. StandardScaler transforms data to have mean = __ and std = __.
**Answer:** 0, 1

---

### Q3. The Perceptron decision rule is: class = sign(_______).
**Answer:** w·x + b (or wx + b)

---

### Q4. Data leakage occurs when _______ data influences model training.
**Answer:** test

---

### Q5. ROC AUC measures the _______ ability of a classifier.
**Answer:** ranking

---

## Answer Key Summary

| Section | Question | Answer |
|---------|----------|--------|
| A | Q1 | C |
| A | Q2 | B |
| A | Q3 | B |
| A | Q4 | C |
| A | Q5 | C |
| A | Q6 | C |
| A | Q7 | B |
| A | Q8 | B |
| A | Q9 | B |
| A | Q10 | B |
| B | Q1 | A, B, D |
| B | Q2 | A, C, E |
| B | Q3 | A, B, D |
| B | Q4 | A, B, D |
| B | Q5 | A, B, C |
| C | Q1 | (900, 5) |
| C | Q2 | 225 |
| C | Q3 | ~90 |
| C | Q4 | 5 |
| C | Q5 | Cluster 2 |
| D | Q1 | nearest |
| D | Q2 | 0, 1 |
| D | Q3 | w·x + b |
| D | Q4 | test |
| D | Q5 | ranking |
