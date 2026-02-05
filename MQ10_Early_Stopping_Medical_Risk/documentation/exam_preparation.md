# 📝 Exam Preparation: Early Stopping & Regularization

## 🅰️ Section A: Multiple Choice Questions (MCQ)

**Q1. What is the primary goal of Early Stopping?**
A) To make the training faster.
B) To prevent underfitting.
C) To prevent overfitting by monitoring validation loss.
D) To increase the number of parameters.
> **Correct:** C
> **Reasoning:** It watches the validation metric and stops when it degrades, which is the signature of overfitting.

**Q2. In L2 Regularization, what value are we adding to the loss function?**
A) The sum of absolute weights.
B) The sum of squared weights.
C) The number of layers.
D) The learning rate.
> **Correct:** B
> **Reasoning:** L2 = Squared (Ridge). L1 = Absolute (Lasso).

**Q3. If `patience=5`, how many epochs do we wait after the best score seeing no improvement?**
A) 0
B) 1
C) 5
D) 100
> **Correct:** C
> **Reasoning:** Patience is literally the "wait time" in epochs.

---

## 🅱️ Section B: Multiple Select Questions (MSQ)

**Q4. Which of the following help reduce overfitting? (Select all that apply)**
[ ] Increase model complexity (more neurons).
[x] L2 Regularization (Weight Decay).
[x] Early Stopping.
[x] Getting more training data.
[ ] Training for more epochs.
> **Explanation:** Complexity and training longer usually *increase* overfitting. The others reduce it.

**Q5. When using `StandardScaler`, which datasets should be fit? (Select one strictly correct workflow)**
[x] Fit on Train, Transform on Train, Val, and Test.
[ ] Fit on Train+Test, Transform all.
[ ] Fit on Test, Transform Train.
> **Explanation:** You must NEVER look at the Test/Val data when calculating mean/std. That is data leakage.

---

## ➗ Section C: Numerical Problems

**Q6. Calculation:**
A model has a loss function $L = Error + \lambda \cdot w^2$.
- Error = 0.5
- Weight $w = 2$
- Regularization strength $\lambda = 0.1$
Calculate the Total Loss.

**Solution:**
1. $w^2 = 2^2 = 4$
2. Penalty = $\lambda \times 4 = 0.1 \times 4 = 0.4$
3. Total Loss = $Error + Penalty = 0.5 + 0.4 = 0.9$

**Answer:** 0.9

---

## ✍️ Section D: Fill in the Blanks

**Q7.** The technique of stopping training when validation error increases is called __________.
> **Answer:** Early Stopping

**Q8.** L2 Regularization is also commonly known as Weight __________.
> **Answer:** Decay

**Q9.** Splitting data while keeping the class ratios consistent is called __________ splitting.
> **Answer:** Stratified
