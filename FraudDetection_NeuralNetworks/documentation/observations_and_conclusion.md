# 📊 Observations and Conclusion

## 1. Execution Output Understanding
When you run the model, you will see a stream of "Epoch" logs.

### Sample Output Explanation
```
Training ShallowWide...
Epoch 1/50 - Train Loss: 0.6543 - Val Loss: 0.5432
...
Early stopping at epoch 12
```

- **Train Loss**: How well the model is learning from the "Practice Exam" (Training Data). We want this to go DOWN.
- **Val Loss**: How well the model performs on the "Mock Exam" (Validation Data). We want this to go DOWN.
- **Early Stopping**: The teacher (algorithm) stopped the training because the student (model) was not improving anymore on the Mock Exam.

## 2. Observations

### 🔍 Observation 1: The Imbalance Effect
- Without SMOTE, the model might achieve **98% Accuracy** immediately.
- **Why?** It simply guesses "Normal" for everything.
- **But:** Precision and Recall for the Fraud class would be 0.00.
- With SMOTE, accuracy might drop to 95%, but **Recall** shoots up to 80-90%. We catch more thieves!

### 🔍 Observation 2: Model Architecture
- **ShallowWide**: Learns quickly but might struggle with complex, subtle fraud patterns.
- **DeepNarrow**: Good at finding deep, hidden relationships but takes longer to train.
- **Hybrid**: Often gives the best balance because different activations capture different types of patterns.

## 3. Insights

### 💡 Insight 1: Recall is King
In fraud detection, **Recall** (catching all fraud) is usually more important than Precision.
- *Missed Fraud cost:* $10,000
- *Phone call to customer cost:* $5

We prefer to annoy 5 customers to stop 1 thief.

### 💡 Insight 2: Deep Learning needs Data
If we had only 500 rows of data, these neural networks would fail (Overfitting). Because we have 50,000 rows (plus SMOTE), they work well.

## 4. Conclusion
1.  **SMOTE is critical**: We cannot build a fraud detector without fixing the imbalance.
2.  **Architecture matters**: The Hybrid or Deep models generally outperform the simple Shallow model on complex synthetic data.
3.  **Real-world readiness**: The "CustomNet" with Dropout is the most robust for production because it aims to generalize better, not just memorize the training data.

## 5. Exam Focus Points
- **Q:** If your Training Loss goes down but Validation Loss goes up, what is happening?
- **A:** **Overfitting**. The model is memorizing the training data but failing on new data.
- **Q:** How do we fix Overfitting?
- **A:** Use **Dropout**, **Early Stopping**, or more data.
