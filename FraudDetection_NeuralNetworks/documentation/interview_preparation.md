# 💼 Interview Preparation Guide

## 1. High-Level Project Summary
> "I built a Fraud Detection System using PyTorch. I handled severe class imbalance (98% vs 2%) using SMOTE and engineered separate Neural Network architectures (Shallow, Deep, Hybrid) to optimize for Recall. I achieved an AUC-ROC of high accuracy."

## 2. Core Concepts – Interview View
| Concept | One-Line Definition | Why use it? |
| :--- | :--- | :--- |
| **Imbalance** | When one class dominates the other (99 vs 1). | To understand why accuracy fails. |
| **SMOTE** | Creating fake examples to balance classes. | To stop the model from being biased. |
| **Recall** | Percentage of actual thieves caught. | Vital when missing a thief is expensive. |

## 3. Top 5 Interview Questions

### Q1: Why did you use Neural Networks instead of Random Forest?
**Answer:** "While Random Forest is great for tabular data, I wanted to experiment with custom loss functions and hybrid activation architectures to specifically target the minority class, which is more flexible in a Neural Network framework."

### Q2: What happens if you run SMOTE on the Test data?
**Answer:** "That is Data Leakage! You are cheating by creating synthetic data based on the answers. The test score will be fake and high, but the model will fail in production."

### Q3: Explain the 'Dying ReLU' problem.
**Answer:** "ReLU outputs 0 for negative inputs. If a neuron always gets negative inputs, it always outputs 0, its gradient becomes 0, and it never updates again—it 'dies'. We can fix this with Leaky ReLU."

### Q4: Why did you normalize (StandardScaler) the data?
**Answer:** "Neural Networks use Gradient Descent. If one feature ranges from 0-1 and another from 0-1000, the gradients will be erratic, and finding the minimum loss will take forever or fail."

### Q5: How do you choose the threshold for fraud?
**Answer:** "Default is 0.5. But for fraud, I might lower it to 0.3 to catch more thieves (increase Recall), even if I annoy a few more honest customers (lower Precision). It's a business decision."

## 4. Parameter & Argument Questions

### `learning_rate=0.001`
- **Why?** Controls how big a step we take during optimization.
- **Too High:** We miss the target (diverge).
- **Too Low:** It takes forever to learn.

### `patience=5` (Early Stopping)
- **Why?** To stop training when the model stops learning.
- **Impact:** Prevents overfitting and saves electricity/time.

## 5. One-Page Quick Revision
- **Problem:** Find rare fraud (2%).
- **Solution:** SMOTE + Deep Learning.
- **Key Metric:** Recall & AUC (Not Accuracy!).
- **Key Risk:** Overfitting (memorizing) & Data Leakage (SMOTE on Test).
