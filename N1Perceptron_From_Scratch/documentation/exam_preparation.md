# Exam Preparation: Perceptron From Scratch

---

## Section A: Multiple Choice Questions (MCQ)

### Q1. What is the output of a Perceptron?
a) Continuous value between -∞ and +∞  
b) Continuous value between 0 and 1  
c) **Binary value: 0 or 1** ✓  
d) One-hot encoded vector  

**Explanation:** Perceptron uses a step function that outputs 1 if z ≥ 0, else 0.

---

### Q2. The Perceptron learning algorithm updates weights:
a) After every sample regardless of prediction  
b) **Only when the prediction is incorrect** ✓  
c) At the end of each epoch  
d) Randomly during training  

**Explanation:** This is the fundamental "error-driven learning" principle.

---

### Q3. Which of these problems can a single Perceptron solve?
a) **AND gate** ✓  
b) XOR gate  
c) Circular classification  
d) Image recognition  

**Explanation:** AND, OR, NOT are linearly separable. XOR is not.

---

### Q4. The decision boundary in a 2D Perceptron is:
a) A point  
b) **A straight line** ✓  
c) A curve  
d) A circle  

**Explanation:** In 2D, w₁x₁ + w₂x₂ + b = 0 is a line.

---

### Q5. What does learning rate control?
a) Number of epochs  
b) Number of features  
c) **Size of weight updates** ✓  
d) Number of samples  

**Explanation:** LR multiplies the error to scale the weight adjustment.

---

### Q6. If learning rate is too high, the Perceptron will:
a) Learn very slowly  
b) Always converge  
c) **Oscillate and may not converge** ✓  
d) Produce probability outputs  

---

### Q7. What is one epoch?
a) One weight update  
b) One sample processed  
c) **One complete pass through all training data** ✓  
d) One correct prediction  

---

### Q8. The Perceptron convergence theorem guarantees convergence when:
a) Learning rate is small enough  
b) **Data is linearly separable** ✓  
c) Number of epochs is large enough  
d) Weights are initialized to zero  

---

### Q9. What is the Perceptron update rule?
a) w = w - η × error × x  
b) **w = w + η × (y_true - y_pred) × x** ✓  
c) w = w × η × error  
d) w = w / η × error × x  

---

### Q10. Why do we shuffle data each epoch?
a) To make training faster  
b) **To prevent learning order-dependent patterns** ✓  
c) To increase accuracy  
d) To reduce memory usage  

---

## Section B: Multiple Select Questions (MSQ)

### Q1. Which of the following are true about Perceptron? (Select all correct)
☑ **It is a linear classifier**  
☑ **It uses a step function for activation**  
☐ It can output probability values  
☑ **It updates only on misclassification**  
☐ It can solve XOR problem  

---

### Q2. Which hyperparameters affect Perceptron training? (Select all correct)
☑ **Learning rate**  
☑ **Number of epochs**  
☐ Kernel type  
☐ Hidden layer size  
☑ **Random seed for shuffling**  

---

### Q3. What happens when we increase class_sep in make_classification? (Select all)
☑ **Classes become more separated**  
☑ **Perceptron converges faster**  
☑ **Higher accuracy is achieved**  
☐ Training becomes slower  

---

### Q4. Which are valid weight initialization strategies for Perceptron?
☑ **All zeros**  
☑ **Small random values**  
☐ Large random values (cause instability)  
☐ One-hot initialization  

---

### Q5. Which problems can a single Perceptron solve? (Select all)
☑ **AND gate**  
☑ **OR gate**  
☑ **NOT gate**  
☐ XOR gate  
☐ Circular boundary  

---

## Section C: Numerical Questions

### Q1. Calculate the Perceptron output

**Given:**
- Weights: w = [0.5, -0.3]
- Bias: b = 0.1
- Input: x = [2, 1]

**Solution:**
```
z = w · x + b
z = (0.5 × 2) + (-0.3 × 1) + 0.1
z = 1.0 - 0.3 + 0.1
z = 0.8

Since z ≥ 0, output = 1
```

**Answer: 1**

---

### Q2. Calculate the weight update

**Given:**
- Current weights: w = [0.4, 0.2]
- Learning rate: η = 0.1
- Input: x = [1, 2]
- True label: y = 1
- Predicted: ŷ = 0

**Solution:**
```
error = y - ŷ = 1 - 0 = 1
w_new = w + η × error × x
w_new = [0.4, 0.2] + 0.1 × 1 × [1, 2]
w_new = [0.4, 0.2] + [0.1, 0.2]
w_new = [0.5, 0.4]
```

**Answer: w = [0.5, 0.4]**

---

### Q3. Calculate accuracy

**Given:**
- Predictions: [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
- True labels: [1, 0, 0, 1, 0, 1, 1, 1, 1, 0]

**Solution:**
```
Correct: [1, 0, _, 1, 0, 1, 1, _, 1, 0] = 8 correct
Total: 10

Accuracy = 8/10 = 0.8 = 80%
```

**Answer: 80%**

---

### Q4. How many updates in one epoch?

**Given:**
- 100 training samples
- Accuracy is 85%

**Solution:**
```
If 85% are correct, 15% are wrong
Wrong predictions = 0.15 × 100 = 15

Number of updates = 15
```

**Answer: 15 updates**

---

### Q5. Calculate decision boundary y-intercept

**Given:**
- Weights: w = [2, 4]
- Bias: b = -8

**Solution:**
```
Decision boundary: w₁x₁ + w₂x₂ + b = 0
2x₁ + 4x₂ - 8 = 0

Solving for x₂ (when x₁ = 0):
4x₂ - 8 = 0
x₂ = 2

Y-intercept = 2
```

**Answer: 2**

---

## Section D: Fill in the Blanks

### Q1. The Perceptron uses a _______ function for activation.
**Answer: step (or Heaviside)**

---

### Q2. One complete pass through all training data is called an _______.
**Answer: epoch**

---

### Q3. The Perceptron is guaranteed to converge if the data is _______ separable.
**Answer: linearly**

---

### Q4. The famous problem that cannot be solved by a single Perceptron is the _______ problem.
**Answer: XOR**

---

### Q5. The equation of the decision boundary is _______ = 0.
**Answer: w·x + b (or w₁x₁ + w₂x₂ + b)**

---

### Q6. Weight updates only occur when the prediction is _______.
**Answer: wrong (or incorrect)**

---

### Q7. To prevent learning order-dependent patterns, we _______ the data each epoch.
**Answer: shuffle**

---

### Q8. The learning rate controls the _______ of weight updates.
**Answer: size (or magnitude/step size)**

---

### Q9. If learning rate is too high, the model may _______ around the optimal solution.
**Answer: oscillate**

---

### Q10. The Perceptron was invented by _______ in 1958.
**Answer: Frank Rosenblatt**

---

## Quick Formula Reference

| Concept | Formula |
|---------|---------|
| Linear Output | z = w·x + b |
| Step Function | y = 1 if z ≥ 0, else 0 |
| Weight Update | w = w + η(y - ŷ)x |
| Bias Update | b = b + η(y - ŷ) |
| Accuracy | correct / total |
| Decision Boundary | w₁x₁ + w₂x₂ + b = 0 |
