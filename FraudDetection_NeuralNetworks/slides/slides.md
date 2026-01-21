# Fraud Detection System
## Neural Network Architectures & Imbalanced Data

---

# 1. Title & Objective
- **Project**: Credit Card Fraud Detection using PyTorch
- **Goal**: Build an AI that catches thieves (Fraud) without stopping honest people (Normal).
- **Key Challenge**: 98% of data is Normal; only 2% is Fraud.
- **Tech Stack**: PyTorch, SMOTE, Python.

---

# 2. Problem Statement
- **Scenario**: You process 10,000 credit card swipes per minute.
- **The Problem**: You cannot check them manually. You need a fast, automated "Brain".
- **Cost**:
  - Missing a Fraud = Loss of Money ($$$)
  - Blocking a Good User = Loss of Trust ($)
- **Objective**: Maximize **Recall** (Catching Thieves) while keeping Precision acceptable.

---

# 3. Real-World Use Case
- **Bank**: Chase, Bank of America.
- **E-commerce**: Amazon, Shopify.
- **Scenario**:
  - User buys a $2000 TV at 3 AM from a new IP address.
  - System must decide in < 100ms: **Allow** or **Block**?

---

# 4. Input Data
- **Source**: Synthetic Data (simulating PCA features).
- **Features**: 30 columns (Time, Amount, V1, V2... V28).
- **Size**: 50,000 Transactions.
- **Imbalance**:
  - Normal: ~49,000 (98%)
  - Fraud: ~1,000 (2%)

---

# 5. Core Concepts
- **Neural Networks**: The "Brain" that learns patterns.
- **Imbalance**: The core difficulty.
- **SMOTE**: The solution to imbalance (Fake data generation).
- **Activation Functions**: ReLU (Fast thinking), Sigmoid (Final decision).

---

# 6. Solution Strategy
1.  **Generate Data**: Create realistic, difficult data.
2.  **Balance**: Use SMOTE to teach the model what fraud looks like.
3.  **Design Brain**: Try 3 different architectures (Shallow, Deep, Hybrid).
4.  **Train**: Use "Binary Cross Entropy" Loss.
5.  **Evaluate**: Check ROC-AUC and Recall.

---

# 7. Code Logic Summary
- **Load**: `make_classification()` -> 50k rows.
- **Preprocess**: Split Train/Test -> SMOTE (Train only) -> Scale.
- **Model**: `class DeepNarrowNet(nn.Module)`.
- **Train**: loop `optimizer.step()`, `loss.backward()`.
- **Eval**: `confusion_matrix()`.

---

# 8. Architectures
- **ShallowWide**: Good for simple, broad patterns.
- **DeepNarrow**: Good for finding deep, hidden connections.
- **Hybrid**:
  - **ReLU**: For internal learning (non-linear).
  - **Tanh**: For zero-centered processing.
  - **Sigmoid**: For final probability (0% to 100%).

---

# 9. Important Functions
- `train_test_split(stratify=y)`: keeps the 98/2 ratio in all splits.
- `SMOTE.fit_resample()`: Creates new fraud examples.
- `nn.BCELoss()`: The error calculator for Yes/No problems.
- `optimizer.zero_grad()`: Resetting the brain before learning new batch.

---

# 10. Observations
- **Without SMOTE**: Model predicts "Safe" for everyone. Accuracy = 98%, Recall = 0%.
- **With SMOTE**: Recall jumps to 85%+.
- **Shallow vs Deep**: Deep network generally finds more subtle fraud patterns but trains slower.

---

# 11. Insights
- **Recall is King**: We accept some False Positives to avoid False Negatives.
- **Data Leakage**: Never use SMOTE on Test data. It gives fake high scores.
- **Normalization**: Essential for Neural Nets to converge.

---

# 12. Conclusion
- We successfully built a fraud detector.
- **Winner**: The **DeepNarrow** network with **SMOTE** provided the best Recall.
- **Next Steps**: Deploy using ONNX or TorchScript for real-time inference.

---

# 13. Interview Key Takeaways
- **Q**: How do you handle imbalance? **A**: SMOTE or Class Weights.
- **Q**: Metric? **A**: AUC-ROC and Recall. Never Accuracy.
- **Q**: Activation? **A**: ReLU for hidden, Sigmoid for output.

---

# 14. Thank You
- **Built with**: PyTorch & Python.
- **Focus**: Teaching Concepts & Production Readiness.
