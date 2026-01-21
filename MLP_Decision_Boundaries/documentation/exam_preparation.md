# Exam Preparation: MLP Decision Boundaries

---

## Section A: Multiple Choice Questions (MCQ)

### Q1. What is the output range of the Sigmoid activation function?
a) [-1, 1]
b) [0, ∞)
c) (0, 1)
d) (-∞, ∞)

**Correct Answer**: c) (0, 1)

**Explanation**: Sigmoid outputs values between 0 and 1 (exclusive), never exactly 0 or 1.

**Why others are wrong**:
- a) This is Tanh's range
- b) This is ReLU's range
- d) This is the input range, not output

---

### Q2. Which activation function is most likely to cause the vanishing gradient problem in deep networks?
a) ReLU
b) Sigmoid
c) LeakyReLU
d) Linear (Identity)

**Correct Answer**: b) Sigmoid

**Explanation**: Sigmoid's maximum gradient is 0.25 (at x=0), and it quickly approaches 0 for large |x|. In deep networks, these small gradients multiply, causing exponential decay.

---

### Q3. What does `hidden_layer_sizes=(8,)` specify in MLPClassifier?
a) 8 hidden layers with 1 neuron each
b) 1 hidden layer with 8 neurons
c) 8 neurons in the output layer
d) 8 input features

**Correct Answer**: b) 1 hidden layer with 8 neurons

**Explanation**: The tuple `(8,)` means one hidden layer containing 8 neurons. The comma makes it a tuple.

---

### Q4. What is the formula for ReLU?
a) 1/(1+e^-x)
b) max(0, x)
c) (e^x - e^-x)/(e^x + e^-x)
d) x²

**Correct Answer**: b) max(0, x)

**Explanation**: ReLU (Rectified Linear Unit) outputs the input if positive, otherwise 0.

---

### Q5. Which dataset is used in this experiment?
a) make_circles
b) make_classification
c) make_moons
d) load_iris

**Correct Answer**: c) make_moons

**Explanation**: make_moons creates two interleaving half-moon shapes, ideal for testing non-linear classifiers.

---

### Q6. What is the purpose of `random_state=42`?
a) Makes the code run faster
b) Increases accuracy
c) Ensures reproducibility
d) Prevents overfitting

**Correct Answer**: c) Ensures reproducibility

**Explanation**: random_state seeds the random number generator so results are identical each run.

---

### Q7. What does backpropagation compute?
a) Forward pass outputs
b) Gradients of loss with respect to weights
c) Final predictions
d) Dataset size

**Correct Answer**: b) Gradients of loss with respect to weights

**Explanation**: Backpropagation uses the chain rule to compute how each weight contributed to the error.

---

### Q8. Which is NOT a valid activation in sklearn's MLPClassifier?
a) 'relu'
b) 'logistic'
c) 'sigmoid'
d) 'tanh'

**Correct Answer**: c) 'sigmoid'

**Explanation**: In sklearn, the sigmoid activation is called 'logistic', not 'sigmoid'.

---

### Q9. What is the gradient of ReLU for x > 0?
a) 0
b) 0.25
c) 1
d) x

**Correct Answer**: c) 1

**Explanation**: For x > 0, ReLU(x) = x, so the derivative is 1.

---

### Q10. What does contourf do in matplotlib?
a) Creates line plots
b) Creates filled contour plots
c) Creates scatter plots
d) Creates bar charts

**Correct Answer**: b) Creates filled contour plots

**Explanation**: contourf creates colored regions between contour lines, used for decision boundaries.

---

## Section B: Multiple Select Questions (MSQ)

### Q1. Which of the following are advantages of ReLU over Sigmoid? (Select all that apply)
- [ ] a) Bounded output
- [x] b) No vanishing gradient for positive inputs
- [x] c) Computationally faster
- [ ] d) Zero-centered

**Correct answers**: b, c

**Explanation**:
- b) ReLU has constant gradient of 1 for positive inputs
- c) ReLU just computes max(0, x), very fast
- a) ReLU is unbounded (0 to ∞)
- d) ReLU is not zero-centered (only outputs ≥0)

---

### Q2. Which statements about decision boundaries are TRUE? (Select all that apply)
- [x] a) They separate different predicted classes
- [x] b) Different models create different boundaries
- [ ] c) They can only be straight lines
- [x] d) Neural networks can learn curved boundaries

**Correct answers**: a, b, d

**Explanation**:
- c) is false - neural networks learn non-linear (curved) boundaries

---

### Q3. Which are valid hidden_layer_sizes specifications? (Select all that apply)
- [x] a) (8,)
- [x] b) (16, 8)
- [ ] c) 8
- [x] d) (4, 4, 4)

**Correct answers**: a, b, d

**Explanation**:
- a) One layer with 8 neurons
- b) Two layers with 16 and 8 neurons
- d) Three layers with 4 neurons each
- c) Integer alone may work but proper form is tuple

---

### Q4. Which affect the shape of decision boundaries? (Select all that apply)
- [x] a) Activation function
- [x] b) Number of neurons
- [x] c) Learned weights
- [ ] d) random_state

**Correct answers**: a, b, c

**Explanation**: random_state affects initial weights but the final boundary depends on the learned weights.

---

### Q5. Which activations suffer from vanishing gradient? (Select all that apply)
- [ ] a) ReLU
- [x] b) Sigmoid
- [x] c) Tanh
- [ ] d) LeakyReLU

**Correct answers**: b, c

**Explanation**: Sigmoid (max gradient 0.25) and Tanh (max gradient 1.0 but decays) have vanishing gradients. ReLU and LeakyReLU maintain constant gradients.

---

## Section C: Numerical Questions

### Q1. If Sigmoid(0) = 0.5, what is Sigmoid'(0)?
**Given**: Sigmoid'(x) = Sigmoid(x) × (1 - Sigmoid(x))

**Solution**:
```
Sigmoid'(0) = Sigmoid(0) × (1 - Sigmoid(0))
            = 0.5 × (1 - 0.5)
            = 0.5 × 0.5
            = 0.25
```

**Answer**: 0.25

---

### Q2. How many total parameters (weights + biases) are in a network with:
- 2 input features
- 1 hidden layer with 8 neurons
- 1 output neuron

**Solution**:
```
Layer 1 (input to hidden):
  Weights: 2 × 8 = 16
  Biases: 8
  Subtotal: 24

Layer 2 (hidden to output):
  Weights: 8 × 1 = 8
  Biases: 1
  Subtotal: 9

Total: 24 + 9 = 33 parameters
```

**Answer**: 33 parameters

---

### Q3. If ReLU(x) = max(0, x), what is the output for inputs [-3, 0, 5]?

**Solution**:
```
ReLU(-3) = max(0, -3) = 0
ReLU(0) = max(0, 0) = 0
ReLU(5) = max(0, 5) = 5
```

**Answer**: [0, 0, 5]

---

### Q4. A model correctly classifies 265 out of 300 samples. What is the accuracy?

**Solution**:
```
Accuracy = Correct / Total × 100%
         = 265 / 300 × 100%
         = 0.8833... × 100%
         = 88.33%
```

**Answer**: 88.33%

---

### Q5. If Tanh(x) = (e^x - e^-x) / (e^x + e^-x), what is Tanh(0)?

**Solution**:
```
Tanh(0) = (e^0 - e^0) / (e^0 + e^0)
        = (1 - 1) / (1 + 1)
        = 0 / 2
        = 0
```

**Answer**: 0

---

## Section D: Fill in the Blanks

### Q1. 
The _____________ problem occurs when gradients become very small in deep networks using sigmoid/tanh activations.

**Answer**: vanishing gradient

---

### Q2.
In sklearn's MLPClassifier, the sigmoid activation is specified using activation='_____________'.

**Answer**: logistic

---

### Q3.
The make_moons dataset creates a ____________-linearly separable classification problem.

**Answer**: non

---

### Q4.
ReLU stands for _____________ _____________ Unit.

**Answer**: Rectified Linear

---

### Q5.
The default solver in MLPClassifier is '____________'.

**Answer**: adam
