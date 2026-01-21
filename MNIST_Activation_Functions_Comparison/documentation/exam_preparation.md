# Exam Preparation: MNIST Activation Functions Comparison

This document contains practice questions for exams, including MCQ, MSQ, Numerical, and Fill-in-the-Blanks.

---

## Section A: Multiple Choice Questions (MCQ)

### Q1. What is the maximum value of Sigmoid's derivative?

- A) 0.5
- B) 1.0
- C) 0.25 ✓
- D) 0.125

**Correct Answer: C) 0.25**

**Explanation:** 
σ'(x) = σ(x) × (1 - σ(x))
Maximum occurs when σ(x) = 0.5: 0.5 × 0.5 = 0.25

**Why others are wrong:**
- A) 0.5: This is σ(0), not the derivative
- B) 1.0: This is Tanh's max derivative, not Sigmoid's
- D) 0.125: This would be after two layers of multiplication

---

### Q2. Which activation function is most prone to the vanishing gradient problem?

- A) ReLU
- B) Sigmoid ✓
- C) Leaky ReLU
- D) Softmax

**Correct Answer: B) Sigmoid**

**Explanation:** 
Sigmoid's maximum derivative is only 0.25, and it saturates at extreme values (derivative → 0), making gradients shrink exponentially through layers.

**Why others are wrong:**
- A) ReLU: Derivative is 1 for positive inputs
- C) Leaky ReLU: Has small but non-zero gradient for negatives
- D) Softmax: Used for output, not hidden layers

---

### Q3. What is the output range of Tanh activation?

- A) (0, 1)
- B) (-∞, +∞)
- C) (-1, 1) ✓
- D) [0, +∞)

**Correct Answer: C) (-1, 1)**

**Explanation:** 
Tanh(x) = (e^x - e^-x) / (e^x + e^-x)
As x → -∞, output → -1
As x → +∞, output → +1

---

### Q4. Why is ReLU computationally faster than Sigmoid?

- A) ReLU uses less memory
- B) ReLU has fewer neurons
- C) ReLU uses max() instead of exp() ✓
- D) ReLU has higher accuracy

**Correct Answer: C) ReLU uses max() instead of exp()**

**Explanation:** 
- ReLU: max(0, x) - simple comparison operation
- Sigmoid: 1/(1+e^-x) - requires exponential calculation

---

### Q5. What happens in the "dying ReLU" problem?

- A) Training becomes too slow
- B) Neurons always output positive values
- C) Neurons always output zero and stop learning ✓
- D) Gradients explode to infinity

**Correct Answer: C) Neurons always output zero and stop learning**

**Explanation:** 
If a neuron's input is always negative, ReLU outputs 0 and its gradient is 0, so weights never update.

---

### Q6. Which activation function should be used for binary classification output?

- A) ReLU
- B) Softmax
- C) Sigmoid ✓
- D) Tanh

**Correct Answer: C) Sigmoid**

**Explanation:** 
Sigmoid outputs a value between 0 and 1, interpretable as probability for binary classification.

---

### Q7. For a network with 4 hidden layers using Sigmoid, what is the worst-case gradient reduction factor at the first layer?

- A) 0.25
- B) 0.0625
- C) 0.00390625 ✓
- D) 0.001

**Correct Answer: C) 0.00390625**

**Explanation:** 
0.25^4 = 0.00390625 (gradient shrinks by 75% at each layer)

---

### Q8. What does Softmax guarantee about its outputs?

- A) All outputs are between 0 and 1
- B) Outputs sum to 1.0 ✓
- C) Maximum output is 1.0
- D) Minimum output is 0.0

**Correct Answer: B) Outputs sum to 1.0**

**Explanation:** 
Softmax normalizes outputs: softmax(z_i) = e^z_i / Σe^z_j, ensuring sum = 1.

---

### Q9. Which statement about Tanh is TRUE?

- A) Tanh is always positive
- B) Tanh is zero-centered ✓
- C) Tanh never suffers from vanishing gradients
- D) Tanh is faster than ReLU

**Correct Answer: B) Tanh is zero-centered**

**Explanation:** 
Tanh outputs range from -1 to 1, with 0 as the center point. This makes gradients more balanced.

---

### Q10. What is the derivative of ReLU for positive inputs?

- A) 0
- B) 1 ✓
- C) x
- D) Undefined

**Correct Answer: B) 1**

**Explanation:** 
ReLU(x) = x for x > 0, so d(ReLU)/dx = 1 for positive inputs.

---

### Q11. Which loss function is typically used with Softmax for multi-class classification?

- A) Mean Squared Error
- B) Binary Cross-Entropy
- C) Categorical Cross-Entropy ✓
- D) Hinge Loss

**Correct Answer: C) Categorical Cross-Entropy**

**Explanation:** 
Categorical cross-entropy is the natural pairing with Softmax for multi-class problems.

---

### Q12. What is the MNIST dataset used for?

- A) Text classification
- B) Handwritten digit recognition ✓
- C) Object detection
- D) Speech recognition

**Correct Answer: B) Handwritten digit recognition**

---

## Section B: Multiple Select Questions (MSQ)

### Q13. Which of the following are advantages of ReLU? (Select all that apply)

- [x] A) No vanishing gradient for positive inputs
- [x] B) Computationally efficient
- [ ] C) Zero-centered outputs
- [x] D) Creates sparse activations
- [ ] E) Never has dead neurons

**Correct Answers: A, B, D**

**Explanation:**
- A) ✓ Derivative = 1 for positive inputs
- B) ✓ max(0, x) is simple to compute
- C) ✗ ReLU outputs are ≥ 0, not zero-centered
- D) ✓ Negative inputs become 0, creating sparsity
- E) ✗ ReLU can have dead neurons (dying ReLU problem)

---

### Q14. Which of the following can help solve vanishing gradients? (Select all that apply)

- [x] A) Using ReLU activation
- [x] B) Batch normalization
- [x] C) Residual connections (skip connections)
- [ ] D) Using deeper networks
- [x] E) Proper weight initialization

**Correct Answers: A, B, C, E**

**Explanation:**
- D is wrong: Deeper networks make vanishing gradients worse

---

### Q15. When should you use Sigmoid activation? (Select all that apply)

- [x] A) Binary classification output layer
- [ ] B) Hidden layers in deep networks
- [x] C) Gates in LSTM networks
- [x] D) Attention scores that need to be 0-1
- [ ] E) Multi-class classification output

**Correct Answers: A, C, D**

---

### Q16. Which are problems with Sigmoid in hidden layers? (Select all that apply)

- [x] A) Vanishing gradients
- [x] B) Not zero-centered
- [x] C) Computationally expensive
- [ ] D) Dying neurons
- [x] E) Saturation at extreme inputs

**Correct Answers: A, B, C, E**

**Explanation:**
- D) Dying neurons is a ReLU problem, not Sigmoid

---

### Q17. What does preprocessing MNIST data involve? (Select all that apply)

- [x] A) Flattening 28x28 images to 784 vectors
- [x] B) Normalizing pixel values to 0-1
- [ ] C) One-hot encoding images
- [ ] D) Applying PCA
- [x] E) Converting to float32

**Correct Answers: A, B, E**

---

## Section C: Numerical Questions

### Q18. Calculate the Sigmoid value for x = 0.

**Solution:**
```
σ(x) = 1 / (1 + e^(-x))
σ(0) = 1 / (1 + e^0)
σ(0) = 1 / (1 + 1)
σ(0) = 1 / 2
σ(0) = 0.5
```

**Final Answer: 0.5**

---

### Q19. If a network has 3 hidden layers with Sigmoid, and the initial gradient at the output is 1.0, what is the gradient at the first hidden layer in the worst case?

**Solution:**
```
Worst case: each layer multiplies by max derivative = 0.25

After layer 3: 1.0 × 0.25 = 0.25
After layer 2: 0.25 × 0.25 = 0.0625
After layer 1: 0.0625 × 0.25 = 0.015625
```

**Final Answer: 0.015625 (or 1/64)**

---

### Q20. How many trainable parameters are in a Dense layer with 784 inputs and 128 outputs?

**Solution:**
```
Weights: 784 × 128 = 100,352
Biases: 128

Total = 100,352 + 128 = 100,480
```

**Final Answer: 100,480 parameters**

---

### Q21. For our MNIST network (784→128→64→10), calculate total trainable parameters.

**Solution:**
```
Layer 1 (784→128): 784 × 128 + 128 = 100,480
Layer 2 (128→64):  128 × 64 + 64 = 8,256
Layer 3 (64→10):   64 × 10 + 10 = 650

Total = 100,480 + 8,256 + 650 = 109,386
```

**Final Answer: 109,386 parameters**

---

### Q22. If batch_size=128 and we have 60,000 training samples, how many batches per epoch?

**Solution:**
```
Batches per epoch = 60,000 / 128 = 468.75
Since we can't have partial batches: ceil(468.75) = 469

Or if dropping incomplete batch: floor(468.75) = 468
```

**Final Answer: 468 or 469 batches (depending on implementation)**

---

### Q23. Calculate Tanh(0).

**Solution:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh(0) = (e^0 - e^0) / (e^0 + e^0)
tanh(0) = (1 - 1) / (1 + 1)
tanh(0) = 0 / 2
tanh(0) = 0
```

**Final Answer: 0**

---

## Section D: Fill in the Blanks

### Q24. The derivative of Sigmoid σ(x) can be expressed as σ(x) × _____________.

**Answer: (1 - σ(x))**

---

### Q25. ReLU stands for _____________ Linear Unit.

**Answer: Rectified**

---

### Q26. In the MNIST dataset, each image has dimensions _______ × _______.

**Answer: 28 × 28**

---

### Q27. The _____________ problem occurs when gradients become too large during backpropagation.

**Answer: Exploding gradient**

---

### Q28. Softmax is typically paired with _____________ loss function for multi-class classification.

**Answer: Categorical cross-entropy**

---

### Q29. The Adam optimizer combines ideas from _____________ and _____________.

**Answer: AdaGrad and RMSprop**

---

### Q30. When ReLU neurons always output zero and never update, this is called the _____________ problem.

**Answer: Dying ReLU (or dead neuron)**

---

### Q31. To solve dying ReLU, we can use _____________ ReLU which allows small negative gradients.

**Answer: Leaky**

---

### Q32. The MNIST training set contains _____________ images, and test set contains _____________ images.

**Answer: 60,000; 10,000**

---

### Q33. For He initialization with n input neurons, variance is set to _____________.

**Answer: 2/n (or 2 divided by n)**

---

## Answer Key

### MCQ
1. C, 2. B, 3. C, 4. C, 5. C, 6. C, 7. C, 8. B, 9. B, 10. B, 11. C, 12. B

### MSQ
13. A,B,D; 14. A,B,C,E; 15. A,C,D; 16. A,B,C,E; 17. A,B,E

### Numerical
18. 0.5; 19. 0.015625; 20. 100,480; 21. 109,386; 22. 468-469; 23. 0

### Fill in the Blanks
24. (1 - σ(x)); 25. Rectified; 26. 28 × 28; 27. Exploding gradient; 28. Categorical cross-entropy; 29. AdaGrad and RMSprop; 30. Dying ReLU; 31. Leaky; 32. 60,000; 10,000; 33. 2/n
