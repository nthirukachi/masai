# 📝 Exam Preparation - AI Crop Health Monitoring

This document contains MCQ, MSQ, Numerical, and Fill-in-the-Blank questions.

---

## Section A: Multiple Choice Questions (MCQ)

### Q1. What does NDVI stand for?
a) Normalized Difference Vegetation Index ✅  
b) Natural Development Vegetation Indicator  
c) Numerical Data Vegetation Index  
d) Normalized Distribution Vegetation Index

**Answer:** (a) Normalized Difference Vegetation Index

**Explanation:** NDVI is calculated using near-infrared and red bands to measure vegetation health. Other options are made-up terms.

---

### Q2. Which range of NDVI values indicates healthy vegetation?
a) -1 to 0  
b) 0 to 0.2  
c) 0.2 to 0.4  
d) 0.6 to 0.9 ✅

**Answer:** (d) 0.6 to 0.9 indicates healthy, dense vegetation

---

### Q3. Random Forest is an example of which type of algorithm?
a) Unsupervised learning  
b) Ensemble learning ✅  
c) Deep learning  
d) Reinforcement learning

**Answer:** (b) Ensemble learning - it combines multiple decision trees

---

### Q4. What is the formula for NDVI?
a) (Red - NIR) / (Red + NIR)  
b) (NIR - Red) / (NIR + Red) ✅  
c) (NIR × Red) / (NIR - Red)  
d) (NIR + Red) / (NIR - Red)

**Answer:** (b) NDVI = (NIR - Red) / (NIR + Red)

---

### Q5. Which metric is best for imbalanced datasets?
a) Accuracy  
b) Mean Squared Error  
c) F1-Score ✅  
d) R-squared

**Answer:** (c) F1-Score balances precision and recall for imbalanced classes

---

### Q6. What does ROC-AUC measure?
a) Training speed  
b) Model's ability to distinguish classes ✅  
c) Memory usage  
d) Feature count

**Answer:** (b) ROC-AUC measures how well a model separates different classes

---

### Q7. What is the purpose of train-test split?
a) To reduce dataset size  
b) To test model on unseen data ✅  
c) To remove outliers  
d) To balance classes

**Answer:** (b) To evaluate model performance on data it hasn't seen during training

---

### Q8. Which vegetation index is better for dense vegetation areas?
a) NDVI  
b) EVI ✅  
c) SAVI  
d) None of these

**Answer:** (b) EVI (Enhanced Vegetation Index) handles saturation better than NDVI

---

### Q9. What does n_estimators parameter control in Random Forest?
a) Maximum tree depth  
b) Number of trees ✅  
c) Learning rate  
d) Batch size

**Answer:** (b) n_estimators is the number of decision trees in the forest

---

### Q10. Label encoding converts:
a) Numbers to text  
b) Text labels to numbers ✅  
c) Images to numbers  
d) Numbers to binary

**Answer:** (b) Label encoding converts categorical text labels to numerical values

---

### Q11. What does stratify parameter do in train_test_split?
a) Shuffles data randomly  
b) Preserves class proportions ✅  
c) Removes duplicates  
d) Normalizes features

**Answer:** (b) stratify ensures same class distribution in train and test sets

---

### Q12. Which of the following is NOT a component of multispectral imaging?
a) NIR band  
b) Red Edge band  
c) Audio band ✅  
d) Green band

**Answer:** (c) Audio is not part of spectral imaging

---

## Section B: Multiple Select Questions (MSQ)

### Q1. Which of the following are valid vegetation indices? (Select all that apply)
- [x] NDVI
- [x] GNDVI
- [x] EVI
- [x] SAVI
- [ ] RGB
- [ ] HTTP

**Answer:** NDVI, GNDVI, EVI, SAVI are vegetation indices. RGB is color space, HTTP is a protocol.

---

### Q2. Random Forest prevents overfitting through: (Select all that apply)
- [x] Bootstrap aggregating (bagging)
- [x] Random feature selection
- [x] Combining multiple trees
- [ ] Using single deep tree
- [ ] Memorizing training data

**Answer:** Bagging, random features, and ensemble averaging all reduce overfitting.

---

### Q3. Which metrics can be calculated from a confusion matrix? (Select all)
- [x] Precision
- [x] Recall
- [x] Accuracy
- [x] F1-Score
- [ ] RMSE
- [ ] R-squared

**Answer:** Precision, Recall, Accuracy, F1-Score are classification metrics from confusion matrix.

---

### Q4. Valid values for random_state parameter include: (Select all)
- [x] 42
- [x] 0
- [x] 123
- [x] Any integer
- [ ] "random"
- [ ] 3.14

**Answer:** random_state accepts any integer for reproducibility.

---

### Q5. Which are advantages of Random Forest? (Select all)
- [x] High accuracy
- [x] Feature importance
- [x] Handles many features
- [x] No feature scaling needed
- [ ] Always fastest
- [ ] Uses least memory

**Answer:** RF is accurate, interpretable, handles many features, but not always fastest or memory-efficient.

---

## Section C: Numerical Questions

### Q1. Calculate NDVI given NIR = 0.8 and Red = 0.2

**Solution:**
```
NDVI = (NIR - Red) / (NIR + Red)
NDVI = (0.8 - 0.2) / (0.8 + 0.2)
NDVI = 0.6 / 1.0
NDVI = 0.6
```

**Answer: 0.6** (Indicates moderately healthy vegetation)

---

### Q2. If precision = 0.85 and recall = 0.75, calculate F1-Score

**Solution:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
F1 = 2 × (0.85 × 0.75) / (0.85 + 0.75)
F1 = 2 × 0.6375 / 1.60
F1 = 1.275 / 1.60
F1 = 0.797
```

**Answer: 0.797 or approximately 0.80**

---

### Q3. A dataset has 1000 samples. With test_size=0.2, how many samples are in training set?

**Solution:**
```
Training samples = Total × (1 - test_size)
Training samples = 1000 × (1 - 0.2)
Training samples = 1000 × 0.8
Training samples = 800
```

**Answer: 800 samples**

---

### Q4. Confusion matrix shows: TP=85, TN=90, FP=10, FN=15. Calculate Accuracy.

**Solution:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy = (85 + 90) / (85 + 90 + 10 + 15)
Accuracy = 175 / 200
Accuracy = 0.875
```

**Answer: 87.5% or 0.875**

---

### Q5. If a Random Forest has n_estimators=100 and each tree takes 0.1 seconds to train, what is the minimum training time (assuming parallel processing with 10 cores)?

**Solution:**
```
Total work = 100 trees × 0.1 seconds = 10 seconds
With 10 parallel cores = 10 / 10 = 1 second minimum
```

**Answer: 1 second (assuming perfect parallelization)**

---

## Section D: Fill in the Blanks

### Q1. NDVI values range from _______ to _______.

**Answer:** -1 to +1

---

### Q2. Random Forest combines multiple _______ trees to make predictions.

**Answer:** decision

---

### Q3. The _______ parameter in train_test_split ensures reproducible results.

**Answer:** random_state

---

### Q4. In a confusion matrix, _______ Positive means the model incorrectly predicted the positive class.

**Answer:** False

---

### Q5. ROC stands for Receiver Operating _______.

**Answer:** Characteristic

---

### Q6. The harmonic mean of precision and recall is called _______.

**Answer:** F1-Score

---

### Q7. _______ encoding converts text labels like "Healthy" to numbers like 0.

**Answer:** Label

---

### Q8. EVI stands for _______ Vegetation Index.

**Answer:** Enhanced

---

### Q9. In Random Forest, _______ aggregating (bagging) helps prevent overfitting.

**Answer:** Bootstrap

---

### Q10. The _______ matrix shows True Positives, False Positives, True Negatives, and False Negatives.

**Answer:** Confusion

---

## 📊 Quick Reference Formulas

| Formula | Equation |
|---------|----------|
| NDVI | (NIR - Red) / (NIR + Red) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (P × R) / (P + R) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
