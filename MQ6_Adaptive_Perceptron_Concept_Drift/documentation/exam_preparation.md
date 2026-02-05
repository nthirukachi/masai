# 📝 Exam Preparation

## Section A: Multiple Choice Questions (MCQ)

**1. What is Concept Drift?**
A) Loss of data during transmission.
B) Change in the relationship between input data and target variable.
C) When the model forgets everything.
D) When the learning rate becomes zero.
**Correct:** B. **Reason:** Definition of concept drift.

**2. In the Perceptron update rule $w = w + \eta(y_{true} - y_{pred})x$, what is $\eta$?**
A) Bias
B) Error
C) Learning Rate
D) Input feature
**Correct:** C. **Reason:** $\eta$ (Eta) represents the step size or learning rate.

**3. Why do we reduce the Learning Rate over time?**
A) To speed up training.
B) To stop training.
C) To allow finer convergence and stability.
D) To increase the weights.
**Correct:** C. **Reason:** Smaller steps avoid overshooting the minimum.

**4. Which of these can solve the XOR problem?**
A) Single Perceptron
B) Linear Regression
C) Multi-Layer Perceptron (MLP)
D) K-Means
**Correct:** C. **Reason:** XOR is non-linear. Single Perceptron is linear. MLP introduces non-linearity.

**5. What is the main risk of a "Reset" mechanism?**
A) It uses too much memory.
B) It loses all previously learned knowledge.
C) It increases the learning rate.
D) It causes overfitting.
**Correct:** B. **Reason:** Resetting wipes $w$ to random, losing history.

---

## Section B: Multiple Select Questions (MSQ)

**1. Which of the following fit the definition of Online Learning? (Select 2)**
A) Learning from a static CSV file all at once.
B) Learning from a stream of data one instance at a time.
C) Updating the model as new data arrives.
D) Training for 1000 epochs on the same dataset.
**Correct:** B, C.

**2. How can we handle Concept Drift? (Select 3)**
A) Periodically retraining the model.
B) Freezing the model weights forever.
C) Using an ensemble of models.
D) Using a sliding window for training.
**Correct:** A, C, D. B is the opposite of handling it.

---

## Section C: Numerical Questions

**Q1. A Perceptron has $w = [0.5, -0.5]$, $b=0$, and activation is Step ($z \ge 0 \to 1$). Input $x = [1, 2]$. What is prediction?**
- Calculation: $z = (0.5 \times 1) + (-0.5 \times 2) + 0$
- $z = 0.5 - 1.0 = -0.5$
- Since $-0.5 < 0$, Output is **0**.

**Q2. Current LR is 0.1. Decay rate is 0.9. What is LR after 2 decays?**
- Step 1: $0.1 \times 0.9 = 0.09$
- Step 2: $0.09 \times 0.9 = 0.081$
- **Answer:** 0.081

---

## Section D: Fill in the Blanks

1. A Perceptron can only classify datasets that are **Linearly Separable**.
2. **Concept Drift** occurs when the statistical properties of the target variable change over time.
3. The **Bias** term allows the decision boundary to not pass through the origin (0,0).
4. **Validation Buffer** is used to evaluate the model on recent data to detect drops in performance.
5. If learning rate is too **High**, the model might oscillate and fail to converge.
