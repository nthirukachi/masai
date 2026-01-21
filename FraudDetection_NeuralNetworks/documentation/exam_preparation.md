# Exam Preparation: Fraud Detection with Neural Networks

Practice questions for exams: MCQ, MSQ, Numerical, and Fill-in-the-Blanks.

---

## Section A: Multiple Choice Questions (MCQ)

### Q1. Which technique creates synthetic samples for the minority class?

- A) Random Undersampling
- B) Random Oversampling  
- C) SMOTE ✓
- D) Feature Scaling

**Correct Answer: C) SMOTE**

**Explanation:** SMOTE (Synthetic Minority Over-sampling Technique) creates new synthetic samples by interpolating between existing minority class samples.

**Why others are wrong:**
- A) Undersampling removes majority class samples
- B) Oversampling duplicates, doesn't create new samples
- D) Feature Scaling normalizes values, doesn't address imbalance

---

### Q2. In fraud detection with 99% legitimate transactions, why is accuracy misleading?

- A) It's too complex to calculate
- B) A model predicting "all legitimate" gets 99% accuracy ✓
- C) Accuracy is never used in machine learning
- D) Fraud detection doesn't use accuracy

**Correct Answer: B)**

**Explanation:** A naive model predicting the majority class achieves 99% accuracy but catches zero fraud.

---

### Q3. What is the output range of the Sigmoid activation function?

- A) (-∞, +∞)
- B) [-1, 1]
- C) (0, 1) ✓
- D) [0, +∞)

**Correct Answer: C) (0, 1)**

**Explanation:** Sigmoid: σ(x) = 1/(1+e^(-x)) maps any input to a value between 0 and 1.

---

### Q4. When should SMOTE be applied?

- A) To the entire dataset before splitting
- B) Only to the training set ✓
- C) Only to the test set
- D) After model training

**Correct Answer: B) Only to the training set**

**Explanation:** Applying SMOTE to test data would inflate metrics and violate data integrity.

---

### Q5. Which metric is most important for catching all fraud cases?

- A) Precision
- B) Recall ✓
- C) Accuracy
- D) Specificity

**Correct Answer: B) Recall**

**Explanation:** Recall = TP/(TP+FN) measures what fraction of actual fraud cases were caught.

---

### Q6. What is the "dying ReLU" problem?

- A) ReLU outputs become too large
- B) Neurons always output 0 and stop learning ✓
- C) Training becomes too fast
- D) ReLU uses too much memory

**Correct Answer: B)**

**Explanation:** If inputs are always negative, ReLU outputs 0, gradient is 0, and the neuron never updates.

---

### Q7. What does Dropout do during training?

- A) Removes features permanently
- B) Removes entire layers
- C) Randomly sets neurons to 0 ✓
- D) Increases learning rate

**Correct Answer: C)**

**Explanation:** Dropout randomly deactivates a fraction of neurons during training to prevent co-adaptation and overfitting.

---

### Q8. What is Binary Cross-Entropy loss used for?

- A) Multi-class classification
- B) Binary classification ✓
- C) Regression
- D) Clustering

**Correct Answer: B) Binary classification**

**Explanation:** BCE is the standard loss for binary classification (fraud/not fraud).

---

### Q9. What does AUC of 0.5 indicate?

- A) Perfect model
- B) Good model
- C) Random guessing ✓
- D) Worst possible model

**Correct Answer: C) Random guessing**

**Explanation:** AUC 0.5 = no discrimination ability, AUC 1.0 = perfect separation.

---

### Q10. Why do we use StandardScaler before training neural networks?

- A) To increase data size
- B) To normalize features for faster convergence ✓
- C) To remove outliers
- D) To reduce number of features

**Correct Answer: B)**

**Explanation:** Neural networks train faster and more stably with normalized inputs (mean=0, std=1).

---

### Q11. What is the purpose of validation set in training?

- A) Training the model
- B) Final performance evaluation
- C) Hyperparameter tuning and early stopping ✓
- D) Data augmentation

**Correct Answer: C)**

---

### Q12. Which activation function solves the vanishing gradient problem?

- A) Sigmoid
- B) Tanh  
- C) ReLU ✓
- D) Softmax

**Correct Answer: C) ReLU**

**Explanation:** ReLU's derivative is 1 for positive inputs, preventing gradient shrinking.

---

## Section B: Multiple Select Questions (MSQ)

### Q13. Which of the following are valid techniques to handle class imbalance? (Select all that apply)

- [x] A) SMOTE
- [x] B) Random Undersampling
- [x] C) Class weights in loss function
- [ ] D) Using Sigmoid activation
- [x] E) Ensemble methods (BalancedRandomForest)

**Correct Answers: A, B, C, E**

---

### Q14. Which metrics are better than accuracy for imbalanced data? (Select all that apply)

- [x] A) Precision
- [x] B) Recall
- [x] C) F1-Score
- [x] D) AUC-ROC
- [ ] E) Mean Squared Error

**Correct Answers: A, B, C, D**

---

### Q15. What are reasons to use early stopping? (Select all that apply)

- [x] A) Prevent overfitting
- [x] B) Save training time
- [ ] C) Increase training set size
- [x] D) Keep best performing model
- [ ] E) Add more features

**Correct Answers: A, B, D**

---

### Q16. Which are benefits of Dropout? (Select all that apply)

- [x] A) Regularization
- [x] B) Prevents overfitting
- [ ] C) Speeds up inference
- [x] D) Reduces co-adaptation
- [ ] E) Increases model capacity

**Correct Answers: A, B, D**

**Note:** C is wrong because Dropout is only used during training, not inference.

---

### Q17. Which statements about SMOTE are TRUE? (Select all that apply)

- [x] A) Creates synthetic samples
- [x] B) Uses k-nearest neighbors
- [ ] C) Should be applied to test data
- [x] D) Can create noise if classes overlap
- [x] E) Reduces overfitting compared to random oversampling

**Correct Answers: A, B, D, E**

---

## Section C: Numerical Questions

### Q18. If a fraud detection model has TP=80, FP=20, FN=10, TN=890, calculate Precision.

**Solution:**
```
Precision = TP / (TP + FP)
Precision = 80 / (80 + 20)
Precision = 80 / 100
Precision = 0.80 (or 80%)
```

**Final Answer: 0.80 or 80%**

---

### Q19. Using the same data (TP=80, FP=20, FN=10, TN=890), calculate Recall.

**Solution:**
```
Recall = TP / (TP + FN)
Recall = 80 / (80 + 10)
Recall = 80 / 90
Recall = 0.889 (or 88.9%)
```

**Final Answer: 0.889 or 88.9%**

---

### Q20. Calculate F1-Score using Precision=0.80 and Recall=0.889.

**Solution:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
F1 = 2 × (0.80 × 0.889) / (0.80 + 0.889)
F1 = 2 × 0.7112 / 1.689
F1 = 1.4224 / 1.689
F1 = 0.842 (or 84.2%)
```

**Final Answer: 0.842 or 84.2%**

---

### Q21. If original data has 100 fraud and 10,000 normal samples, and SMOTE balances classes, how many total samples after SMOTE?

**Solution:**
```
After SMOTE: Fraud = Normal = 10,000
Total = 10,000 + 10,000 = 20,000
```

**Final Answer: 20,000 samples**

---

### Q22. A neural network has layers: Input(30) → Dense(64) → Dense(32) → Dense(1). Calculate total trainable parameters.

**Solution:**
```
Layer 1 (30→64): 30 × 64 + 64 = 1,920 + 64 = 1,984
Layer 2 (64→32): 64 × 32 + 32 = 2,048 + 32 = 2,080
Layer 3 (32→1):  32 × 1 + 1 = 32 + 1 = 33

Total = 1,984 + 2,080 + 33 = 4,097
```

**Final Answer: 4,097 parameters**

---

### Q23. If batch_size=64 and training set has 10,000 samples, how many batches per epoch?

**Solution:**
```
Batches = ceil(10,000 / 64) = ceil(156.25) = 157
```

**Final Answer: 157 batches (or 156 if dropping incomplete)**

---

## Section D: Fill in the Blanks

### Q24. SMOTE stands for _____________ Minority Over-sampling Technique.

**Answer: Synthetic**

---

### Q25. The formula for Recall is TP / (TP + _____).

**Answer: FN (False Negatives)**

---

### Q26. ReLU stands for _____________ Linear Unit.

**Answer: Rectified**

---

### Q27. Dropout randomly sets _____________ to zero during training.

**Answer: neurons (or activations)**

---

### Q28. Binary Cross-Entropy loss is paired with _____________ activation in the output layer.

**Answer: Sigmoid**

---

### Q29. StandardScaler transforms data to have mean ______ and standard deviation ______.

**Answer: 0; 1**

---

### Q30. Early stopping monitors _____________ loss to prevent overfitting.

**Answer: validation**

---

### Q31. AUC stands for Area Under the _____________.

**Answer: Curve (specifically ROC Curve)**

---

### Q32. A model with 0% recall in fraud detection catches _____________ fraud cases.

**Answer: zero (or none)**

---

### Q33. BatchNorm normalizes the inputs to each layer during _____________.

**Answer: training**

---

## Answer Key

### MCQ
1. C, 2. B, 3. C, 4. B, 5. B, 6. B, 7. C, 8. B, 9. C, 10. B, 11. C, 12. C

### MSQ
13. A,B,C,E; 14. A,B,C,D; 15. A,B,D; 16. A,B,D; 17. A,B,D,E

### Numerical
18. 0.80; 19. 0.889; 20. 0.842; 21. 20,000; 22. 4,097; 23. 157

### Fill in the Blanks
24. Synthetic; 25. FN; 26. Rectified; 27. neurons; 28. Sigmoid; 29. 0, 1; 30. validation; 31. Curve; 32. zero; 33. training
