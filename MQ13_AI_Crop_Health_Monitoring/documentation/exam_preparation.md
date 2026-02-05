# 📚 Exam Preparation: AI Crop Health Monitoring

## Question Bank with Answers

---

## Section A: Multiple Choice Questions (MCQs)

### Q1. What does NDVI stand for?
- A) Normalized Difference Visual Index
- B) **Normalized Difference Vegetation Index** ✅
- C) Natural Drone Vegetation Index
- D) Numeric Data Vegetation Index

### Q2. Which light bands are used in NDVI calculation?
- A) Blue and Green
- B) **Red and Near-Infrared (NIR)** ✅
- C) Green and NIR
- D) Red and Blue

### Q3. What is the range of NDVI values?
- A) 0 to 1
- B) 0 to 100
- C) **-1 to +1** ✅
- D) -100 to +100

### Q4. Which model performed best in this project?
- A) Random Forest
- B) Decision Tree
- C) **Logistic Regression** ✅
- D) KNN

### Q5. What metric is best for imbalanced classification?
- A) Accuracy
- B) Mean Squared Error
- C) **F1-Score** ✅
- D) R-Squared

### Q6. What does StandardScaler do?
- A) Removes missing values
- B) **Transforms features to mean=0, std=1** ✅
- C) Selects best features
- D) Reduces dimensions

### Q7. In train-test split, why use stratify parameter?
- A) To shuffle data randomly
- B) **To maintain class proportions in both sets** ✅
- C) To remove duplicates
- D) To increase data size

### Q8. What is the formula for F1-Score?
- A) (Precision + Recall) / 2
- B) Precision × Recall
- C) **(2 × Precision × Recall) / (Precision + Recall)** ✅
- D) (TP + TN) / Total

### Q9. In Random Forest, what does n_estimators=100 mean?
- A) 100 features used
- B) **100 decision trees in the forest** ✅
- C) 100 samples per tree
- D) 100 iterations

### Q10. Which model is called a "lazy learner"?
- A) Logistic Regression
- B) Random Forest
- C) SVM
- D) **KNN** ✅

---

## Section B: Short Answer Questions

### Q1. Define vegetation indices and list any 3 examples. (3 marks)

**Answer:**
Vegetation indices are mathematical formulas combining different light bands to measure plant health.

Examples:
1. **NDVI** - Normalized Difference Vegetation Index
2. **GNDVI** - Green NDVI (uses green instead of red)
3. **SAVI** - Soil-Adjusted Vegetation Index (adds soil correction)

---

### Q2. Explain the difference between Precision and Recall with an example. (4 marks)

**Answer:**

| Metric | Definition | Formula |
|--------|------------|---------|
| Precision | Of predicted positive, how many are correct? | TP/(TP+FP) |
| Recall | Of actual positive, how many did we find? | TP/(TP+FN) |

**Example:**
- Model predicts 100 plants as "Stressed"
- Actually 90 are stressed, 10 are healthy
- **Precision = 90/100 = 90%**

- Actual stressed plants: 100 in field
- Model found only 90 of them
- **Recall = 90/100 = 90%**

---

### Q3. Why did Logistic Regression outperform Random Forest in this project? (3 marks)

**Answer:**
1. **Well-engineered features:** Vegetation indices already extracted meaningful patterns
2. **Linear separability:** Data had nearly linear decision boundary
3. **No overfitting:** Simpler model generalized better
4. **Small dataset:** 1,200 samples favor simpler models

---

### Q4. Explain train-test split and why it's important. (4 marks)

**Answer:**
**Definition:** Dividing data into training set (80%) and test set (20%).

**Purpose:**
- Training set: Model learns patterns
- Test set: Evaluate on unseen data

**Importance:**
1. Prevents overfitting detection
2. Gives honest accuracy estimate
3. Simulates real-world deployment

**Analogy:** Students can't be tested on questions they already practiced!

---

### Q5. What is a confusion matrix? Draw and explain. (4 marks)

**Answer:**

```
                Predicted
              Positive  Negative
Actual Positive   TP        FN
       Negative   FP        TN
```

**TP (True Positive):** Actually stressed, predicted stressed ✅
**TN (True Negative):** Actually healthy, predicted healthy ✅
**FP (False Positive):** Actually healthy, predicted stressed ❌
**FN (False Negative):** Actually stressed, predicted healthy ❌

---

## Section C: Long Answer Questions

### Q1. Describe the complete ML pipeline for crop health classification. (10 marks)

**Answer:**

**1. Data Collection (2 marks)**
- Drone captures multispectral imagery
- Data includes 13 vegetation indices per grid cell
- Target: Healthy/Stressed classification

**2. Data Preprocessing (2 marks)**
- Check for missing values (none found)
- Encode labels (Healthy→0, Stressed→1)
- Train-test split (80/20 stratified)
- Feature scaling (StandardScaler)

**3. Model Training (3 marks)**
- Train 5 classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
  - KNN
- Use training data only

**4. Evaluation (2 marks)**
- Calculate: Accuracy, Precision, Recall, F1, ROC-AUC
- Select best model by F1-Score
- Analyze confusion matrix

**5. Deployment (1 mark)**
- Generate predictions for all grid cells
- Create spatial stress heatmap
- Prioritize inspection zones

---

### Q2. Compare any three ML algorithms used in this project. (8 marks)

**Answer:**

| Aspect | Logistic Regression | Random Forest | SVM |
|--------|---------------------|---------------|-----|
| **Type** | Linear classifier | Ensemble of trees | Maximum margin |
| **How it works** | Sigmoid function on linear combination | 100 trees voting | Finds widest boundary |
| **Interpretability** | High (coefficients) | Low (black box) | Medium |
| **Speed** | Very fast | Slower | Medium |
| **Scaling needed** | Yes | No | Yes |
| **Overfitting risk** | Low | Medium | Low |
| **F1 in project** | 95.76% | 89.57% | 92.22% |
| **Best for** | Linear problems | Non-linear, tabular | High-dim data |

---

### Q3. Explain how spatial analysis helps in crop monitoring. (6 marks)

**Answer:**

**What is spatial analysis?**
Analyzing predictions based on geographic location (grid_x, grid_y).

**How we implemented it:**
1. Predicted stress for all 1,200 grid cells
2. Created pivot table by coordinates
3. Generated heatmap visualization

**Benefits:**
- **Pattern detection:** Stress clusters visible
- **Targeted action:** Focus on critical zones
- **Resource optimization:** Don't inspect healthy areas

**Findings:**
- 361 critical zones (≥80% stress probability)
- Stress appears clustered (localized issue)
- Action: Deploy drone to red zones first

---

## Section D: Numerical Problems

### Q1. Calculate F1-Score
Given: TP=79, FP=2, FN=5

**Solution:**
```
Precision = TP/(TP+FP) = 79/(79+2) = 79/81 = 0.975
Recall = TP/(TP+FN) = 79/(79+5) = 79/84 = 0.940
F1 = 2 × (P × R)/(P + R) = 2 × (0.975 × 0.940)/(0.975 + 0.940)
F1 = 2 × 0.9165/1.915 = 1.833/1.915 = 0.957
```
**Answer: F1-Score = 0.957 or 95.7%**

---

### Q2. Calculate NDVI
Given: NIR reflectance = 0.8, Red reflectance = 0.1

**Solution:**
```
NDVI = (NIR - Red)/(NIR + Red)
NDVI = (0.8 - 0.1)/(0.8 + 0.1)
NDVI = 0.7/0.9
NDVI = 0.778
```
**Answer: NDVI = 0.78 (Healthy vegetation)**

---

## Key Points to Memorize

1. **NDVI = (NIR-Red)/(NIR+Red)**, range -1 to +1
2. **F1 = 2×(P×R)/(P+R)**, harmonic mean
3. **Stratify** maintains class ratios
4. **StandardScaler**: z = (x-μ)/σ
5. **Logistic Regression** won with **95.76% F1**
6. **361 critical zones** identified
7. **80/20** train-test split
8. **Never fit scaler on test data!**
