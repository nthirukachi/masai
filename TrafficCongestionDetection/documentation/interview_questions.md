# 💼 Interview Questions & Answers

## 1. Basic Level (The "What")

### Q1: What is the goal of this project?
**Simple Answer:** To use AI to find traffic jams automatically using camera data so we can fix them faster.
**Technical Answer:** To build a supervised classification pipeline using Random Forest to detect traffic congestion based on aerial imagery features like speed and density.

### Q2: Why did you use Random Forest?
**Simple Answer:** It's like asking 100 experts instead of 1. It is very accurate and doesn't get confused easily.
**Technical Answer:** Random Forest is an ensemble method that reduces overfitting compared to a single Decision Tree and handles non-linear relationships between features well.

---

## 2. Intermediate Level (The "How")

### Q3: How do you handle different scales (e.g., Speed 0-100 vs Density 0-1)?
**Simple Answer:** We shrink them all to a similar size (scaling) so the AI treats them equally.
**Technical Answer:** We used `StandardScaler` to normalize features (mean=0, variance=1) to ensure convergence and fair weighting, especially if we were to switch to distance-based models like KNN.

### Q4: What does Precision and Recall mean in this project?
**Simple Answer:**
- **Precision:** When we said "Jam", were we right? (Don't want false alarms).
- **Recall:** Did we find *all* the jams? (Don't want to miss an accident).
**Technical Answer:** Precision is TP/(TP+FP). Recall is TP/(TP+FN). In emergency response, High Recall is usually preferred because missing an accident is worse than a false alarm.

---

## 3. Advanced Level (The "Why Not")

### Q5: Why not use a Neural Network (Deep Learning)?
**Simple Answer:** It's like using a cannon to kill a fly. Our problem is simple enough for Random Forest, which is faster and cheaper.
**Technical Answer:** While valid, Neural Networks require more data and compute. For tabular features, tree-based models (XGBoost/RF) often outperform simple NNs and are more interpretable.

### Q6: What happens if it rains?
**Answer:**
- The `illumination` feature might change.
- `optical_flow` might get noisy.
- The model might fail if not trained on "Rainy" data. We need to collect rainy data to fix this (Data Augmentation).

---

## 4. Visual Explanation for Interviews
```mermaid
graph LR
    Interviewer[Interviewer: Explain the flow] --> You
    You[You] --> Step1[Input: Vehicle Speed & Count]
    Step1 --> Step2[Model: Checks Rules]
    Step2 --> Step3[Output: Jam or Clear]
    Step3 --> Step4[Action: Alert Police]
```
