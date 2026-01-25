# 📝 Exam Preparation - AI Landing Zone Safety

## Section A: Multiple Choice Questions (MCQ)

### Q1. What type of machine learning problem is landing zone safety classification?
- [ ] A) Regression
- [ ] B) Clustering
- [x] C) Classification
- [ ] D) Dimensionality Reduction

**Answer**: C) Classification  
**Explanation**: We predict discrete categories (safe/unsafe), not continuous values.

---

### Q2. Why is accuracy alone insufficient for safety-critical systems?
- [ ] A) It's too hard to calculate
- [ ] B) It only works for regression
- [x] C) It can be misleading with imbalanced classes
- [ ] D) It requires too much data

**Answer**: C) It can be misleading with imbalanced classes  
**Explanation**: If 90% of zones are safe, predicting "always safe" gives 90% accuracy but misses all dangerous zones.

---

### Q3. What does a ROC-AUC value of 0.5 indicate?
- [x] A) Random guessing
- [ ] B) Perfect classifier
- [ ] C) Model always predicts positive
- [ ] D) Model always predicts negative

**Answer**: A) Random guessing  
**Explanation**: AUC = 0.5 means the model performs no better than random chance.

---

### Q4. In Random Forest, what is "bagging"?
- [ ] A) Feature selection
- [ ] B) Hyperparameter tuning
- [x] C) Bootstrap aggregating
- [ ] D) Gradient boosting

**Answer**: C) Bootstrap aggregating  
**Explanation**: Each tree is trained on a random sample of data with replacement.

---

### Q5. Which metric should be prioritized for drone landing safety?
- [ ] A) Accuracy
- [ ] B) Precision
- [x] C) Recall for unsafe class
- [ ] D) Training speed

**Answer**: C) Recall for unsafe class  
**Explanation**: We must catch all dangerous zones; missing an unsafe zone could cause a crash.

---

### Q6. What does StandardScaler do?
- [ ] A) Removes outliers
- [x] B) Transforms features to mean=0, std=1
- [ ] C) Removes missing values
- [ ] D) Encodes categorical variables

**Answer**: B) Transforms features to mean=0, std=1  
**Explanation**: Formula: `scaled = (value - mean) / std`

---

### Q7. Why use stratify=y in train_test_split?
- [ ] A) To speed up training
- [x] B) To maintain class proportions in both sets
- [ ] C) To shuffle the data
- [ ] D) To remove duplicates

**Answer**: B) To maintain class proportions in both sets  
**Explanation**: Especially important when classes are imbalanced.

---

### Q8. What does precision measure?
- [x] A) Of predicted positives, how many are actually positive
- [ ] B) Of actual positives, how many were predicted positive
- [ ] C) Overall correctness
- [ ] D) Model speed

**Answer**: A) Of predicted positives, how many are actually positive  
**Explanation**: Precision = TP / (TP + FP)

---

### Q9. In a confusion matrix, what is a False Negative?
- [ ] A) Predicted negative, actually negative
- [x] B) Predicted negative, actually positive
- [ ] C) Predicted positive, actually positive
- [ ] D) Predicted positive, actually negative

**Answer**: B) Predicted negative, actually positive  
**Explanation**: The model missed a positive case (dangerous for safety systems).

---

### Q10. Which feature would likely indicate an UNSAFE landing zone?
- [ ] A) Low slope_deg (flat ground)
- [ ] B) Low roughness (smooth surface)
- [x] C) High object_density (many obstacles)
- [ ] D) High confidence_score (certain detection)

**Answer**: C) High object_density (many obstacles)  
**Explanation**: Obstacles increase collision risk during landing.

---

## Section B: Multiple Select Questions (MSQ)

### Q11. Which of the following are advantages of Random Forest? (Select all that apply)
- [x] A) Handles non-linear relationships
- [x] B) Provides feature importance
- [ ] C) Very fast to train
- [x] D) Resistant to overfitting
- [ ] E) Works well with minimal data

**Answers**: A, B, D  
**Explanation**: RF handles non-linearity, gives feature importance, and is robust. It's not particularly fast and needs reasonable data.

---

### Q12. Which metrics are important for safety-critical classification? (Select all)
- [x] A) Recall
- [x] B) Precision
- [x] C) F1-Score
- [ ] D) Training Time
- [x] E) ROC-AUC

**Answers**: A, B, C, E  
**Explanation**: Training time is not a safety metric; all others measure prediction quality.

---

### Q13. What should happen when a drone encounters a low-confidence prediction? (Select all appropriate actions)
- [x] A) Perform secondary scan
- [x] B) Request human confirmation
- [ ] C) Land immediately anyway
- [x] D) Expand search radius
- [ ] E) Ignore the prediction

**Answers**: A, B, D  
**Explanation**: Low confidence = uncertainty = need verification, not blind trust.

---

### Q14. Which of the following are terrain features in the dataset? (Select all)
- [x] A) slope_deg
- [x] B) roughness
- [ ] C) temperature
- [x] D) ndvi_mean
- [ ] E) wind_speed

**Answers**: A, B, D  
**Explanation**: Temperature and wind_speed are not in this dataset (listed as limitations).

---

### Q15. Which are valid methods to improve the model? (Select all)
- [x] A) Collect more training data
- [x] B) Try different algorithms (XGBoost)
- [x] C) Engineer new features
- [ ] D) Remove the test set
- [x] E) Tune hyperparameters

**Answers**: A, B, C, E  
**Explanation**: Never remove the test set - that would prevent model evaluation!

---

## Section C: Numerical Questions

### Q16. If a model has TP=80, TN=50, FP=10, FN=20, what is the accuracy?

**Solution**:
```
Total = TP + TN + FP + FN = 80 + 50 + 10 + 20 = 160
Accuracy = (TP + TN) / Total = (80 + 50) / 160 = 130/160 = 0.8125
```

**Answer**: **81.25%** or **0.8125**

---

### Q17. Using the same values, calculate Precision.

**Solution**:
```
Precision = TP / (TP + FP) = 80 / (80 + 10) = 80 / 90 = 0.8889
```

**Answer**: **88.89%** or **0.889**

---

### Q18. Using the same values, calculate Recall.

**Solution**:
```
Recall = TP / (TP + FN) = 80 / (80 + 20) = 80 / 100 = 0.80
```

**Answer**: **80%** or **0.80**

---

### Q19. If we have 1000 samples and use test_size=0.2, how many samples are in the test set?

**Solution**:
```
Test samples = Total × test_size = 1000 × 0.2 = 200
```

**Answer**: **200 samples**

---

### Q20. A Random Forest has n_estimators=100 and each tree predicts: 55 say "safe", 45 say "unsafe". What is the final prediction?

**Solution**:
```
Majority vote: 55 > 45
55 trees predict "safe" → Final prediction = SAFE
```

**Answer**: **SAFE** (majority vote wins)

---

## Section D: Fill in the Blanks

### Q21. Random Forest uses __________ (technique) to train each tree on a random sample of data.
**Answer**: **Bagging** (Bootstrap Aggregating)

---

### Q22. The formula for F1-Score is __________.
**Answer**: **2 × (Precision × Recall) / (Precision + Recall)**

---

### Q23. To prevent data leakage, we should only __________ the scaler on training data.
**Answer**: **fit**

---

### Q24. ROC stands for __________.
**Answer**: **Receiver Operating Characteristic**

---

### Q25. In a confusion matrix, __________ is the most dangerous error in safety-critical systems.
**Answer**: **False Negative** (predicting safe when actually unsafe)

---

### Q26. The feature __________ indicates vegetation amount in the landing zone.
**Answer**: **ndvi_mean**

---

### Q27. StandardScaler transforms data to have mean __________ and standard deviation __________.
**Answer**: **0** and **1**

---

### Q28. In the decision framework, zones with safety score > 80% and confidence > 70% should __________.
**Answer**: **auto-land** (or be cleared for automatic landing)

---

### Q29. __________ is the Python library used for machine learning in this project.
**Answer**: **scikit-learn** (or sklearn)

---

### Q30. The target variable in this dataset is called __________ with values 0 (unsafe) and 1 (safe).
**Answer**: **label**
