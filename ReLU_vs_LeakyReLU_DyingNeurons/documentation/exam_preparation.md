# Exam Preparation: ReLU vs Leaky ReLU

---

## Section A: Multiple Choice Questions (MCQ)

### Q1. What is the output of ReLU(-5)?
a) -5  
b) 5  
c) **0** ✓  
d) -0.05

**Answer: c) 0**  
*Explanation*: ReLU(x) = max(0, x). For negative inputs, output is always 0.  
*Why others are wrong*: (a) ReLU doesn't pass negatives, (b) ReLU doesn't take absolute value, (d) that would be Leaky ReLU.

---

### Q2. What is the derivative of ReLU when x = -3?
a) 1  
b) -1  
c) 3  
d) **0** ✓

**Answer: d) 0**  
*Explanation*: ReLU derivative is 0 for x ≤ 0, and 1 for x > 0.  
*Why others are wrong*: The derivative is either 0 or 1, never negative or the input value.

---

### Q3. What is the output of Leaky ReLU(-100) with alpha=0.01?
a) 0  
b) -100  
c) **-1** ✓  
d) 1

**Answer: c) -1**  
*Explanation*: Leaky ReLU = alpha × x for x < 0. So 0.01 × (-100) = -1.  
*Why others are wrong*: (a) is ReLU output, (b) no scaling, (d) wrong sign.

---

### Q4. Why does the dying ReLU problem occur?
a) ReLU outputs very large values  
b) **ReLU gradient is 0 for negative inputs** ✓  
c) ReLU is computationally expensive  
d) ReLU causes exploding gradients

**Answer: b) ReLU gradient is 0 for negative inputs**  
*Explanation*: Zero gradient means zero weight updates, causing neurons to get stuck.  
*Why others are wrong*: (a) unbounded but not the problem, (c) ReLU is cheap, (d) opposite problem.

---

### Q5. What is the derivative of Leaky ReLU when x = -5 (alpha=0.01)?
a) 0  
b) -5  
c) 1  
d) **0.01** ✓

**Answer: d) 0.01**  
*Explanation*: Leaky ReLU derivative is alpha for x ≤ 0, and 1 for x > 0.  
*Why others are wrong*: The derivative is alpha (0.01), not 0 like ReLU.

---

### Q6. Which initialization is best for ReLU activation?
a) Random uniform  
b) Xavier/Glorot  
c) **He initialization** ✓  
d) Zero initialization

**Answer: c) He initialization**  
*Explanation*: He initialization accounts for ReLU only using positive half, using sqrt(2/n).  
*Why others are wrong*: (a) no scaling, (b) designed for sigmoid/tanh, (d) causes symmetry.

---

### Q7. In a neural network with ReLU, a dead neuron means:
a) The neuron has very high weights  
b) **The neuron outputs 0 for all inputs** ✓  
c) The neuron has very low bias  
d) The neuron has negative bias

**Answer: b) The neuron outputs 0 for all inputs**  
*Explanation*: Dead = zero output for ALL samples, not just some.  
*Why others are wrong*: Weight/bias values alone don't define "dead."

---

### Q8. Which activation function CANNOT have dead neurons?
a) ReLU  
b) Sigmoid  
c) **Leaky ReLU** ✓  
d) Tanh

**Answer: c) Leaky ReLU**  
*Explanation*: Leaky ReLU always has non-zero gradient (alpha), so neurons always learn.  
*Why others are wrong*: ReLU can die, sigmoid/tanh have vanishing gradients.

---

### Q9. What does alpha represent in Leaky ReLU?
a) Learning rate  
b) **Slope for negative inputs** ✓  
c) Regularization strength  
d) Dropout probability

**Answer: b) Slope for negative inputs**  
*Explanation*: Alpha controls how much of negative values passes through (typically 0.01 = 1%).  
*Why others are wrong*: Alpha is specific to Leaky ReLU, not a general hyperparameter.

---

### Q10. Forward propagation in a 2-layer network computes:
a) Only the output  
b) Only the gradients  
c) **z and a for each layer** ✓  
d) Only the loss

**Answer: c) z and a for each layer**  
*Explanation*: We compute z (weighted sum) and a (activation) for each layer.  
*Why others are wrong*: We need intermediate values for backpropagation.

---

## Section B: Multiple Select Questions (MSQ)

### Q1. Which of the following can prevent the dying ReLU problem? (Select all that apply)
- [x] **Using Leaky ReLU**
- [x] **Proper weight initialization (He)**
- [x] **Lower learning rate**
- [ ] Using sigmoid activation
- [x] **Batch Normalization**

*Explanation*: Leaky ReLU prevents zero gradients, proper init keeps activations balanced, lower LR prevents wild weight updates, BatchNorm normalizes activations.

---

### Q2. Which statements about ReLU are TRUE? (Select all that apply)
- [x] **ReLU(x) = max(0, x)**
- [x] **ReLU is computationally efficient**
- [ ] ReLU always has non-zero gradient
- [x] **ReLU outputs are unbounded**
- [x] **ReLU creates sparse activations**

*Explanation*: ReLU is simple (just max), fast, outputs can grow large, and many outputs are 0 (sparse). However, gradient IS zero for x ≤ 0.

---

### Q3. Which are valid alternatives to ReLU? (Select all that apply)
- [x] **Leaky ReLU**
- [x] **ELU**
- [x] **Swish**
- [ ] Softmax
- [x] **SELU**

*Explanation*: Leaky ReLU, ELU, Swish, SELU are all activation functions for hidden layers. Softmax is for output layer classification.

---

### Q4. During backpropagation with ReLU, the gradient is zero when: (Select all that apply)
- [x] **The pre-activation z is negative**
- [x] **The pre-activation z is exactly zero**
- [ ] The pre-activation z is positive
- [x] **The neuron is "dead"**
- [ ] The learning rate is zero

*Explanation*: ReLU derivative is 0 for z ≤ 0. Dead neurons by definition have z ≤ 0 for all inputs.

---

### Q5. Which are advantages of Leaky ReLU over ReLU? (Select all that apply)
- [x] **Prevents dead neurons**
- [x] **Gradients always flow**
- [ ] Faster computation
- [ ] Simpler formula
- [x] **Works better in deep networks**

*Explanation*: Leaky ReLU adds slight complexity but prevents gradient death. ReLU is simpler and slightly faster.

---

## Section C: Numerical Questions

### Q1. Calculate ReLU output
Given: z = [-2, -1, 0, 1, 3]

**Solution:**
- ReLU(-2) = max(0, -2) = 0
- ReLU(-1) = max(0, -1) = 0
- ReLU(0) = max(0, 0) = 0
- ReLU(1) = max(0, 1) = 1
- ReLU(3) = max(0, 3) = 3

**Answer: [0, 0, 0, 1, 3]**

---

### Q2. Calculate Leaky ReLU output (alpha=0.1)
Given: z = [-10, -5, 0, 5, 10]

**Solution:**
- LeakyReLU(-10) = 0.1 × (-10) = -1
- LeakyReLU(-5) = 0.1 × (-5) = -0.5
- LeakyReLU(0) = 0.1 × 0 = 0
- LeakyReLU(5) = 5
- LeakyReLU(10) = 10

**Answer: [-1, -0.5, 0, 5, 10]**

---

### Q3. Calculate dead neuron percentage
A network has 50 hidden neurons. After training, 10 neurons output 0 for all samples.

**Solution:**
Dead neuron percentage = (10 / 50) × 100 = 20%

**Answer: 20% of neurons are dead**

---

### Q4. Calculate gradient during backpropagation
Given: upstream_gradient = 0.5, z = -3 (for ReLU)

**Solution:**
gradient = upstream_gradient × ReLU_derivative(z)
         = 0.5 × 0  (since z = -3 < 0)
         = 0

**Answer: 0** (gradient stops flowing - dead neuron!)

---

### Q5. Same as Q4, but with Leaky ReLU (alpha=0.01)
Given: upstream_gradient = 0.5, z = -3

**Solution:**
gradient = upstream_gradient × LeakyReLU_derivative(z)
         = 0.5 × 0.01  (since z = -3 < 0)
         = 0.005

**Answer: 0.005** (gradient still flows!)

---

## Section D: Fill in the Blanks

### Q1. ReLU stands for ________ Linear Unit.
**Answer: Rectified**

### Q2. The formula for ReLU is max(0, ___).
**Answer: z (or x)**

### Q3. In Leaky ReLU, the default value of alpha is typically ______.
**Answer: 0.01**

### Q4. A neuron is called "dead" when its output is ____ for all training samples.
**Answer: zero (or 0)**

### Q5. The derivative of ReLU for positive inputs is ____.
**Answer: 1**

### Q6. The derivative of ReLU for negative inputs is ____.
**Answer: 0**

### Q7. Forward propagation computes the formula z = W × X + ____.
**Answer: b (bias)**

### Q8. ______ initialization is recommended for ReLU networks.
**Answer: He (or Kaiming)**

### Q9. The dying ReLU problem occurs because the ________ becomes zero.
**Answer: gradient**

### Q10. Leaky ReLU prevents dead neurons by having a non-zero ________ for negative inputs.
**Answer: gradient (or derivative or slope)**

---

## Quick Reference Table

| Concept | ReLU | Leaky ReLU |
|---------|------|------------|
| Formula | max(0, z) | max(αz, z) |
| Derivative (z>0) | 1 | 1 |
| Derivative (z≤0) | 0 | α (0.01) |
| Dead neurons? | Yes | No |
| Complexity | O(1) | O(1) |

---

## Key Formulas to Remember

```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0, else 0

LeakyReLU(z) = max(αz, z) where α = 0.01
LeakyReLU'(z) = 1 if z > 0, else α

Forward: z = Wx + b, a = activation(z)
Backward: dz = upstream × activation'(z)
Weight update: W = W - learning_rate × dW
```
