# 📝 Interview Preparation Guide

## Rapid Revision for Interviews & Exams

---

## 1. High-Level Project Summary

**Problem:** Farmers need to monitor crop health across large fields efficiently.

**Solution:**
- Use drone with multispectral camera to capture vegetation data
- Apply ML classification to detect healthy vs stressed crops
- Generate spatial heatmap for targeted inspection
- Prioritize drone inspection zones

**Key Result:** Logistic Regression achieved 95.76% F1-Score

---

## 2. Core Concepts - Quick Reference

### Vegetation Indices
| Index | Formula | Use |
|-------|---------|-----|
| NDVI | (NIR-Red)/(NIR+Red) | General health |
| GNDVI | (NIR-Green)/(NIR+Green) | Chlorophyll |
| SAVI | (NIR-Red)/(NIR+Red+L)×(1+L) | Sparse vegetation |

### ML Models Compared
1. **Logistic Regression** - Linear boundary, interpret with coefficients
2. **Decision Tree** - Flowchart of yes/no questions
3. **Random Forest** - 100 trees voting together
4. **SVM** - Finds widest margin boundary
5. **KNN** - Classify by nearest neighbors

### Key Metrics
| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | (TP+TN)/All | Balanced classes |
| Precision | TP/(TP+FP) | False alarms costly |
| Recall | TP/(TP+FN) | Missing cases costly |
| F1-Score | 2PR/(P+R) | Imbalanced classes |

---

## 3. Frequently Asked Questions

### Q: Why did you choose Logistic Regression?
**A:** It provided the highest F1-Score (95.76%). The well-engineered vegetation indices created nearly linear separation, making a simple model optimal.

### Q: How did you handle class imbalance?
**A:** Used stratified train-test split to maintain 65/35 ratio, and chose F1-Score as primary metric.

### Q: What preprocessing did you apply?
**A:** StandardScaler to normalize features (mean=0, std=1), fit on training data only.

### Q: Why not use deep learning?
**A:** With tabular data (13 features, 1200 samples), traditional ML often outperforms deep learning. CNNs would be better for raw imagery.

---

## 4. Parameter & Argument Quick Reference

### train_test_split
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
- `test_size=0.2`: 20% for testing
- `random_state=42`: Reproducibility
- `stratify=y`: Keep class ratios

### StandardScaler
```python
scaler.fit_transform(X_train)  # ONLY on train
scaler.transform(X_test)       # NEVER fit on test
```

### RandomForestClassifier
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```
- `n_estimators=100`: 100 trees voting

---

## 5. Key Comparisons Table

### Logistic Regression vs Random Forest

| Aspect | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Type | Linear | Ensemble |
| Interpretability | High (coefficients) | Low (black box) |
| Speed | Fast | Slower |
| Overfitting | Low risk | Medium risk |
| Our Result | 95.76% F1 | 89.57% F1 |

### Precision vs Recall

| Scenario | Prioritize | Why |
|----------|------------|-----|
| False alarms expensive | Precision | Avoid treating healthy crops |
| Missing stress dangerous | Recall | Catch ALL stressed plants |
| Both matter equally | F1-Score | Balance needed |

### Training vs Testing Data

| Aspect | Training | Testing |
|--------|----------|---------|
| Purpose | Model learns patterns | Evaluate generalization |
| Size | 80% (960 samples) | 20% (240 samples) |
| Scaler | fit_transform | transform only |

---

## 6. Common Mistakes & Traps

### ❌ Mistake 1: Fitting scaler on all data
```python
# WRONG
scaler.fit_transform(X)  # Leaks test info!

# CORRECT
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

### ❌ Mistake 2: Using accuracy for imbalanced data
"Model got 65% accuracy!" - But 65% are healthy, so predicting all "healthy" gives 65% for free!

### ❌ Mistake 3: Not stratifying the split
Random split might give test set with 90% healthy, 10% stressed - unfair evaluation!

### ❌ Mistake 4: Confusing Precision with Recall
- Precision = quality of positive predictions
- Recall = coverage of actual positives

---

## 7. Output Interpretation

### What does F1=95.76% mean?
Model correctly identifies stressed crops 95.76% of the time while minimizing false alarms.

### What does ROC-AUC=99.81% mean?
Model almost perfectly ranks stressed plants higher than healthy ones.

### What does the heatmap show?
- Red zones = High stress probability
- Green zones = Healthy areas
- 361 CRITICAL cells need immediate inspection

---

## 8. One-Page Quick Revision

### Numbers to Remember
- **1,200** samples (grid cells)
- **13** vegetation features
- **5** models compared
- **95.76%** best F1-Score
- **361** critical zones
- **80/20** train-test split

### Key Algorithms
- **Logistic Regression** = Winner! (linear, fast, interpretable)
- **Random Forest** = Many trees voting (good but slower)
- **SVM** = Maximum margin boundary
- **KNN** = k=5 nearest neighbors vote
- **Decision Tree** = Flowchart decisions (overfit risk)

### Critical Formulas
```
NDVI = (NIR - Red) / (NIR + Red)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
z-score = (x - mean) / std
```

### What I Learned
1. Simple models can beat complex ones
2. Feature engineering matters more than algorithm choice
3. Always stratify imbalanced data
4. Never fit scaler on test data
5. F1-Score better than accuracy for imbalanced classes

---

## 9. 5-Minute Presentation Script

**Opening (30 sec):**
"Today I'll present an AI system for crop health monitoring using drone multispectral imagery."

**Problem (30 sec):**
"Farmers can't manually inspect large fields. Drones capture vegetation data, but we need AI to analyze it."

**Solution (1 min):**
"We trained 5 ML models on 13 vegetation indices. Logistic Regression achieved 95.76% F1-Score."

**Results (1 min):**
"We identified 361 critical zones needing immediate inspection. The spatial heatmap shows clustered stress patterns."

**Demo (1 min):**
"Here's the model comparison chart and stress heatmap..."

**Conclusion (1 min):**
"Simple models + good features = excellent results. Future work: temporal analysis and deep learning on raw imagery."

---

## 10. Confidence Boosters

### When interviewer says "Why not deep learning?"
**Answer:** "With tabular features and 1,200 samples, traditional ML is optimal. Deep learning shines with raw imagery (pixels) and millions of samples."

### When interviewer says "How would you deploy this?"
**Answer:** "Flask/FastAPI API endpoint, Docker containerization, cloud deployment (AWS/GCP), real-time prediction from drone data stream."

### When interviewer says "What if model performs poorly in production?"
**Answer:** "Monitor drift, collect ground truth feedback, retrain periodically, A/B test new models before deployment."
