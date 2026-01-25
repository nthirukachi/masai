# 📝 Exam Preparation - Forest Fire & Smoke Detection

---

## Section A: Multiple Choice Questions (MCQ)

### Question 1
**What is the primary purpose of Random Forest in this project?**

A) To create a single decision tree  
B) To combine multiple decision trees through voting ✅  
C) To perform unsupervised clustering  
D) To reduce the number of features  

**Correct Answer: B**

**Explanation**: Random Forest is an ensemble method that creates multiple decision trees and combines their predictions through majority voting (classification) or averaging (regression).

**Why others are wrong:**
- A) Random Forest uses MANY trees, not one
- C) This is supervised classification, not clustering
- D) Feature reduction is done by PCA, not RF

---

### Question 2
**What is the recall of 85.8% in this project?**

A) 85.8% of all predictions were correct  
B) 85.8% of fire predictions were actual fires  
C) 85.8% of actual fires were correctly detected ✅  
D) 85.8% of safe tiles were correctly classified  

**Correct Answer: C**

**Explanation**: Recall = TP / (TP + FN). It measures what percentage of actual positive cases (fires) were correctly identified by the model.

**Why others are wrong:**
- A) This describes accuracy
- B) This describes precision
- D) This describes true negative rate (specificity)

---

### Question 3
**Which feature is MOST important for fire detection in this model?**

A) haze_index  
B) edge_density  
C) mean_red ✅  
D) local_contrast  

**Correct Answer: C**

**Explanation**: mean_red has the highest importance (0.273 or 27.3%) because fire emits red/orange light, making red intensity the strongest visual indicator.

**Why others are wrong:**
- A) haze_index has importance ~0.019
- B) edge_density has importance ~0.021
- D) local_contrast has importance ~0.020

---

### Question 4
**What does ROC-AUC = 0.969 indicate?**

A) The model correctly classifies 96.9% of samples  
B) The model has 96.9% probability of ranking a fire tile higher than a safe tile ✅  
C) 96.9% of predictions are fire  
D) The model has a 3.1% error rate  

**Correct Answer: B**

**Explanation**: AUC represents the probability that a randomly chosen positive example (fire) will be ranked higher than a randomly chosen negative example (safe).

**Why others are wrong:**
- A) This describes accuracy (which is 93%)
- C) This would be a class distribution metric
- D) Error rate ≠ 1 - AUC

---

### Question 5
**Why is stratification used in train-test split?**

A) To speed up training  
B) To maintain class proportions in both sets ✅  
C) To increase the dataset size  
D) To remove outliers  

**Correct Answer: B**

**Explanation**: Stratification ensures both training and testing sets have the same proportion of each class (65% safe, 35% fire), preventing biased evaluation.

**Why others are wrong:**
- A) Stratification doesn't affect speed
- C) It doesn't change dataset size
- D) Outlier removal is a different process

---

### Question 6
**What is a False Negative in fire detection?**

A) A safe tile predicted as safe  
B) A fire tile predicted as fire  
C) A safe tile predicted as fire  
D) A fire tile predicted as safe ✅  

**Correct Answer: D**

**Explanation**: False Negative = Actual Positive (Fire) predicted as Negative (Safe). This is the most dangerous error in fire detection.

**Why others are wrong:**
- A) This is True Negative (correct)
- B) This is True Positive (correct)
- C) This is False Positive (less dangerous)

---

### Question 7
**What is the formula for F1-Score?**

A) (Precision + Recall) / 2  
B) 2 × (Precision × Recall) / (Precision + Recall) ✅  
C) Precision × Recall  
D) TP / (TP + FP + FN)  

**Correct Answer: B**

**Explanation**: F1-Score is the harmonic mean of precision and recall, giving equal weight to both.

**Why others are wrong:**
- A) This is arithmetic mean (doesn't penalize imbalance)
- C) This would favor one over the other
- D) This is not a standard metric formula

---

### Question 8
**What is the purpose of StandardScaler?**

A) To remove outliers  
B) To transform features to mean=0, std=1 ✅  
C) To increase feature values  
D) To reduce the number of features  

**Correct Answer: B**

**Explanation**: StandardScaler normalizes features by removing the mean and scaling to unit variance: z = (x - mean) / std

**Why others are wrong:**
- A) Outlier removal is different (e.g., IQR method)
- C) Values can increase or decrease
- D) Feature reduction is PCA, not scaling

---

### Question 9
**How many tiles are classified as "Critical Risk" in the project?**

A) 153  
B) 189  
C) 862 ✅  
D) 1796  

**Correct Answer: C**

**Explanation**: 862 tiles (28.7%) have fire risk probability ≥ 75% and are classified as Critical Risk.

**Why others are wrong:**
- A) 153 is High Risk count
- B) 189 is Medium Risk count
- D) 1796 is Low Risk count

---

### Question 10
**What is bagging in Random Forest?**

A) Removing features  
B) Bootstrap Aggregating - training on random samples with replacement ✅  
C) Reducing tree depth  
D) Combining predictions by averaging  

**Correct Answer: B**

**Explanation**: Bagging = Bootstrap AGGregatING. Each tree is trained on a random sample (with replacement) of the original data.

**Why others are wrong:**
- A) Feature removal is feature selection
- C) Tree depth is controlled by max_depth parameter
- D) This is part of aggregating, but not bagging definition

---

### Question 11
**What is the accuracy of the trained model?**

A) 85.8%  
B) 89.7%  
C) 93.0% ✅  
D) 96.9%  

**Correct Answer: C**

**Explanation**: Accuracy = (TP + TN) / Total = (182 + 376) / 600 = 93.0%

**Why others are wrong:**
- A) 85.8% is recall
- B) 89.7% is F1-score
- D) 96.9% is ROC-AUC

---

### Question 12
**Which sklearn function is used to split data?**

A) sklearn.model_selection.cross_val_score  
B) sklearn.model_selection.train_test_split ✅  
C) sklearn.preprocessing.split  
D) sklearn.datasets.split  

**Correct Answer: B**

**Explanation**: train_test_split from sklearn.model_selection is the standard function for splitting data.

**Why others are wrong:**
- A) cross_val_score is for cross-validation
- C) This function doesn't exist
- D) This function doesn't exist

---

## Section B: Multiple Select Questions (MSQ)

### Question 1
**Which of the following are TRUE about Random Forest? (Select all that apply)**

☑️ A) It reduces overfitting compared to single decision trees  
☑️ B) It can provide feature importance  
☐ C) It requires data to be normally distributed  
☑️ D) It uses bootstrap sampling  
☐ E) It can only handle binary classification  

**Correct Answers: A, B, D**

**Explanations:**
- A) ✅ Ensemble averaging reduces variance/overfitting
- B) ✅ Tree-based importance is built-in
- C) ❌ No distribution assumption required
- D) ✅ Bagging uses bootstrap samples
- E) ❌ Can handle multi-class and regression too

---

### Question 2
**Which metrics should be prioritized for fire detection? (Select all that apply)**

☑️ A) Recall  
☐ B) Accuracy alone  
☑️ C) F1-Score  
☑️ D) ROC-AUC  
☐ E) Mean Squared Error  

**Correct Answers: A, C, D**

**Explanations:**
- A) ✅ Recall catches all fires (critical for safety)
- B) ❌ Accuracy alone can be misleading for imbalanced data
- C) ✅ F1 balances precision and recall
- D) ✅ ROC-AUC shows ranking ability
- E) ❌ MSE is for regression, not classification

---

### Question 3
**Which of the following are dataset limitations mentioned in the project? (Select all that apply)**

☑️ A) No temporal/time data  
☑️ B) Tiles analyzed independently (no spatial context)  
☑️ C) No weather data  
☐ D) Too few features  
☐ E) All features are categorical  

**Correct Answers: A, B, C**

**Explanations:**
- A) ✅ No timestamps to track fire progression
- B) ✅ Neighboring tiles not considered
- C) ✅ Wind/humidity not included
- D) ❌ 10 features is sufficient
- E) ❌ All features are numerical

---

### Question 4
**Which of the following contribute to fire risk probability > 0.75? (Select all that apply)**

☑️ A) High mean_red value  
☑️ B) High smoke_whiteness  
☑️ C) High hot_pixel_fraction  
☐ D) High mean_green (healthy vegetation)  
☐ E) Low intensity variation  

**Correct Answers: A, B, C**

**Explanations:**
- A) ✅ Fire appears red
- B) ✅ Smoke appears white/gray
- C) ✅ Fire creates hot spots
- D) ❌ High green indicates healthy forest (no fire)
- E) ❌ Fire typically has HIGH intensity variation (flickering)

---

### Question 5
**Which are valid ways to improve recall? (Select all that apply)**

☑️ A) Lower the classification threshold  
☑️ B) Use class_weight='balanced'  
☑️ C) Oversample the minority class (SMOTE)  
☐ D) Use a higher threshold  
☐ E) Remove features  

**Correct Answers: A, B, C**

**Explanations:**
- A) ✅ Lower threshold catches more positives
- B) ✅ Class weighting penalizes missing positives
- C) ✅ SMOTE creates more positive samples
- D) ❌ Higher threshold increases precision, not recall
- E) ❌ Feature removal unlikely to help recall

---

## Section C: Numerical Questions

### Question 1
**If we have 212 actual fire tiles and 30 were missed (False Negatives), calculate the recall.**

**Solution:**
```
True Positives (TP) = 212 - 30 = 182
False Negatives (FN) = 30

Recall = TP / (TP + FN)
Recall = 182 / (182 + 30)
Recall = 182 / 212
Recall = 0.858 or 85.8%
```

**Answer: 85.8% or 0.858**

---

### Question 2
**If there are 194 fire predictions and 182 are correct, calculate precision.**

**Solution:**
```
True Positives (TP) = 182
False Positives (FP) = 194 - 182 = 12

Precision = TP / (TP + FP)
Precision = 182 / (182 + 12)
Precision = 182 / 194
Precision = 0.938 or 93.8%
```

**Answer: 93.8% or 0.938**

---

### Question 3
**Calculate F1-Score given Precision = 0.938 and Recall = 0.858**

**Solution:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
F1 = 2 × (0.938 × 0.858) / (0.938 + 0.858)
F1 = 2 × 0.8048 / 1.796
F1 = 1.6096 / 1.796
F1 = 0.896 or 89.6%
```

**Answer: 0.896 or 89.6%**

---

### Question 4
**If 80% of 3000 samples are used for training, how many samples are in the test set?**

**Solution:**
```
Total samples = 3000
Training percentage = 80%
Testing percentage = 100% - 80% = 20%

Test samples = 3000 × 0.20 = 600
```

**Answer: 600 samples**

---

### Question 5
**Given the confusion matrix, calculate accuracy:**
```
TN = 376, FP = 12
FN = 30, TP = 182
```

**Solution:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy = (182 + 376) / (182 + 376 + 12 + 30)
Accuracy = 558 / 600
Accuracy = 0.93 or 93%
```

**Answer: 93% or 0.93**

---

### Question 6
**If a Random Forest has 100 trees and each tree takes 20ms to predict, what is the minimum prediction time for ensemble?**

**Solution:**
```
With parallel processing (n_jobs=-1):
  Time = Max individual tree time = 20ms (parallel)

Without parallel processing:
  Time = 100 × 20ms = 2000ms = 2 seconds

Minimum time (parallel) = 20ms
```

**Answer: 20ms (with parallelization)**

---

### Question 7
**Calculate the percentage of Critical Risk tiles if there are 862 critical tiles out of 3000 total.**

**Solution:**
```
Percentage = (Critical tiles / Total tiles) × 100
Percentage = (862 / 3000) × 100
Percentage = 0.2873 × 100
Percentage = 28.73%
```

**Answer: 28.7% (rounded)**

---

## Section D: Fill in the Blanks

### Question 1
Random Forest is an example of __________ learning method.

**Answer: ensemble**

---

### Question 2
The formula for Precision is TP / (TP + __________).

**Answer: FP (False Positives)**

---

### Question 3
ROC stands for __________ Operating Characteristic.

**Answer: Receiver**

---

### Question 4
In train_test_split, the parameter __________ ensures class proportions are maintained.

**Answer: stratify**

---

### Question 5
StandardScaler transforms features to have mean = __________ and standard deviation = __________.

**Answer: 0, 1**

---

### Question 6
The most important feature for fire detection in this project is __________.

**Answer: mean_red**

---

### Question 7
False __________ is more dangerous than False Positive in fire detection.

**Answer: Negative**

---

### Question 8
AUC value of 0.5 indicates __________ guessing performance.

**Answer: random**

---

### Question 9
The __________ matrix shows True Positives, True Negatives, False Positives, and False Negatives.

**Answer: confusion**

---

### Question 10
F1-Score is the __________ mean of Precision and Recall.

**Answer: harmonic**

---

## 🎯 Quick Formula Reference

| Metric | Formula |
|--------|---------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (P × R) / (P + R) |
| Specificity | TN / (TN + FP) |

## 📊 Our Results Summary

| Metric | Value |
|--------|-------|
| Accuracy | 93.0% |
| Precision | 93.8% |
| Recall | 85.8% |
| F1-Score | 89.7% |
| ROC-AUC | 0.969 |
| TN | 376 |
| FP | 12 |
| FN | 30 |
| TP | 182 |
