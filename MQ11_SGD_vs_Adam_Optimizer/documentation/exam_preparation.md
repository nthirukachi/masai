# 📝 Exam Preparation

## Section A: Multiple Choice Questions (MCW)

### Q1. Which optimizer is generally considered to have "Method of Moments"?
A) SGD
B) Adam
C) ReLU
D) MSE
**Correct Answer:** B) Adam
**Explanation:** Adam (Adaptive Moment Estimation) uses both the first moment (mean) and second moment (variance) of the gradients.

### Q2. What happens if the Learning Rate is too high?
A) The model converges very slowly.
B) The model overshoots the minimum and diverges.
C) The model stops learning immediately.
D) The model becomes a regression model.
**Correct Answer:** B) The model overshoots the minimum and diverges.
**Explanation:** Large steps cause the optimizer to bounce out of the valley.

### Q3. Why do we split data into Train, Validation, and Test?
A) To make the code longer.
B) To check if the model works on unseen data.
C) To train three different models.
D) To use more memory.
**Correct Answer:** B) To check if the model works on unseen data.
**Explanation:** Validation/Test sets act as a proxy for real-world performance.

---

## Section B: Multiple Select Questions (MSQ)

### Q4. Select all TRUE statements about SGD with Momentum.
A) It helps accelerate gradients vectors in the right direction.
B) It dampens oscillations.
C) It calculates adaptive learning rates for each parameter.
D) It usually requires less memory than Adam.
**Correct Answers:** A, B, D
**Explanation:** Momentum helps navigation and stability. Adam stores more states (m and v), so SGD uses less memory.

### Q5. Which of the following are Activation Functions?
A) ReLU
B) Sigmoid
C) MSE
D) Tanh
**Correct Answers:** A, B, D
**Explanation:** MSE is a Loss function, not an activation function.

---

## Section C: Numerical Questions

### Q6. Calculate the Output Size
Given an input layer of size 40 and a first hidden layer of size 128. How many weights are in this connection (excluding bias)?
**Solution:**
$$ Weights = Input \times Output $$
$$ Weights = 40 \times 128 $$
$$ Weights = 5120 $$
**Answer:** 5120 weights.

### Q7. Batch Size Calculation
If we have 2000 samples and a batch size of 64. How many batches are in one epoch?
**Solution:**
$$ Batches = \lceil \frac{Total Samples}{Batch Size} \rceil $$
$$ Batches = \lceil \frac{2000}{64} \rceil $$
$$ 2000 / 64 = 31.25 $$
**Answer:** 32 batches (the last batch will be smaller).

---

## Section D: Fill in the Blanks

### Q8. ____________ is a technique to prevent overfitting by stopping training when validation loss increases.
**Answer:** Early Stopping

### Q9. In PyTorch, the _____________ function computes the gradients.
**Answer:** `.backward()`

### Q10. We use `torch.manual_seed()` to ensure _____________.
**Answer:** Reproducibility
