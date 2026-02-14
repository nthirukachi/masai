# CL38: Backpropagation & ANN Fundamentals - Exam Preparation

---

## Section A: Multiple Choice Questions (MCQ) - 15 Questions

### MCQ 1
**Question:** In a forward pass through a neural network, what is the correct sequence of operations at each neuron?

**Options:**
- A) Activation → Weighted Sum → Bias
- B) Weighted Sum → Bias → Activation Function
- C) Bias → Activation → Weighted Sum
- D) Activation → Bias → Weighted Sum

**✅ Correct Answer:** B

**📖 Explanation:** 
1. First, calculate Z = Σ(weights × inputs) + bias (linear combination)
2. Then apply activation function: a = f(Z)

Telglish: "Mundu weights multiply chesi, bias add chesi, tarvata activation function apply chestam."

**❌ Why Others Are Wrong:**
- A) Activation comes LAST, not first
- C) Bias is added to weighted sum, not applied first
- D) Wrong order entirely

---

### MCQ 2
**Question:** If Z = W₁X₁ + W₂X₂ + b, and we apply ReLU, what is the output when Z = -3?

**Options:**
- A) -3
- B) 3
- C) 0
- D) 1

**✅ Correct Answer:** C

**📖 Explanation:** ReLU(x) = max(0, x). Since Z = -3 is negative, ReLU(-3) = max(0, -3) = 0

**❌ Why Others Are Wrong:**
- A) ReLU never outputs negative values
- B) ReLU doesn't take absolute value
- D) 1 is not related to the calculation

---

### MCQ 3
**Question:** In backpropagation, why do we use the chain rule of calculus?

**Options:**
- A) To calculate forward pass outputs
- B) To decompose gradients through nested functions (layers)
- C) To initialize weights
- D) To normalize inputs

**✅ Correct Answer:** B

**📖 Explanation:** Neural networks are composed of nested functions: f(g(h(x))). Chain rule multiplies partial derivatives through each layer.

**❌ Why Others Are Wrong:**
- A) Forward pass doesn't need chain rule
- C) Initialization uses random values, not chain rule
- D) Normalization is a separate technique

---

### MCQ 4
**Question:** Why is ReLU preferred over Sigmoid in hidden layers?

**Options:**
- A) ReLU is more complex mathematically
- B) ReLU avoids the vanishing gradient problem
- C) ReLU outputs values between 0 and 1
- D) ReLU requires more computation

**✅ Correct Answer:** B

**📖 Explanation:** 
- Sigmoid derivative max = 0.25, in deep networks → gradient → 0
- ReLU derivative = 1 for positive values → gradient stays intact

**❌ Why Others Are Wrong:**
- A) ReLU is simpler: max(0, x)
- C) Sigmoid outputs 0-1, not ReLU
- D) ReLU is computationally simpler

---

### MCQ 5
**Question:** During inference (after training), which operations are performed?

**Options:**
- A) Forward pass only
- B) Backward pass only
- C) Both forward and backward passes
- D) Neither forward nor backward

**✅ Correct Answer:** A

**📖 Explanation:** Inference only needs prediction. Weights are frozen. No learning/updates needed.

**❌ Why Others Are Wrong:**
- B) Backward pass is for training only
- C) Backpropagation is only during training
- D) Forward pass is needed to get predictions

---

### MCQ 6
**Question:** What does the weight update formula W_new = W_old - η × (∂L/∂W) do?

**Options:**
- A) Increases the loss
- B) Moves weight in direction of steepest descent
- C) Randomizes the weight
- D) Freezes the weight

**✅ Correct Answer:** B

**📖 Explanation:** 
- Gradient points to steepest ascent
- Subtracting moves opposite direction = descent = lower loss

**❌ Why Others Are Wrong:**
- A) Formula is designed to decrease loss
- C) No randomization involved
- D) Weight is actively updated, not frozen

---

### MCQ 7
**Question:** Why can't we initialize all weights to zero in a neural network?

**Options:**
- A) Computation will overflow
- B) Symmetry problem - all neurons learn the same thing
- C) Activation functions won't work
- D) Loss function becomes undefined

**✅ Correct Answer:** B

**📖 Explanation:** With all zeros, all neurons compute identically, get identical gradients, update identically → Never learn different features!

**❌ Why Others Are Wrong:**
- A) No overflow with zeros
- C) Activations still work on zero (e.g., ReLU(0)=0)
- D) Loss is still defined

---

### MCQ 8
**Question:** In the MNIST digit classification, why does the output layer have exactly 10 neurons?

**Options:**
- A) Because images are 10×10 pixels
- B) Because there are 10 digits (0-9) to classify
- C) Because 10 is optimal for softmax
- D) Because the training set has 10 images

**✅ Correct Answer:** B

**📖 Explanation:** Each output neuron corresponds to one class. Digits 0-9 = 10 classes = 10 neurons.

**❌ Why Others Are Wrong:**
- A) MNIST images are 28×28
- C) Softmax works with any number of classes
- D) MNIST has thousands of images

---

### MCQ 9
**Question:** During CNN backpropagation, what gets updated and what stays fixed?

**Options:**
- A) Filter weights updated, filter sizes fixed
- B) Filter sizes updated, filter weights fixed
- C) Both filter weights and sizes updated
- D) Neither gets updated

**✅ Correct Answer:** A

**📖 Explanation:** 
- Filter SIZE (3×3, 5×5) = hyperparameter, fixed at design time
- Filter VALUES = learned during training

**❌ Why Others Are Wrong:**
- B) Sizes don't change during training
- C) Sizes are architecture decisions, not learned
- D) Weights DO get updated

---

### MCQ 10
**Question:** What is the maximum value of sigmoid function's derivative?

**Options:**
- A) 1.0
- B) 0.5
- C) 0.25
- D) 0.1

**✅ Correct Answer:** C

**📖 Explanation:** σ'(x) = σ(x)(1 - σ(x)). Maximum at x=0 where σ(0)=0.5: 0.5 × 0.5 = 0.25

**❌ Why Others Are Wrong:**
- A) That's ReLU's gradient for positive values
- B) 0.5 is sigmoid output at x=0, not derivative
- D) Too small for maximum

---

### MCQ 11
**Question:** Why do we stack multiple layers in a neural network?

**Options:**
- A) To slow down computation
- B) To learn hierarchical features from simple to complex
- C) To reduce the number of parameters
- D) To avoid using activation functions

**✅ Correct Answer:** B

**📖 Explanation:** 
- Layer 1: Learns simple features (edges)
- Layer 2: Combines into patterns
- Layer 3: Recognizes objects

**❌ Why Others Are Wrong:**
- A) More layers = more computation
- C) More layers = more parameters
- D) Still need activation functions

---

### MCQ 12
**Question:** What happens if learning rate is too high?

**Options:**
- A) Training becomes very slow
- B) Model may overshoot minimum and diverge
- C) Model achieves perfect accuracy
- D) Gradients become zero

**✅ Correct Answer:** B

**📖 Explanation:** Large steps may jump over the minimum, causing loss to increase instead of decrease.

**❌ Why Others Are Wrong:**
- A) High LR = fast but unstable (opposite)
- C) May never converge, not achieve perfection
- D) High LR doesn't zero gradients

---

### MCQ 13
**Question:** Which activation function outputs a probability distribution summing to 1?

**Options:**
- A) ReLU
- B) Sigmoid
- C) Tanh
- D) Softmax

**✅ Correct Answer:** D

**📖 Explanation:** Softmax: eˣⁱ/Σeˣʲ normalizes outputs so they sum to 1 (probability distribution).

**❌ Why Others Are Wrong:**
- A) ReLU outputs [0, ∞), no normalization
- B) Sigmoid outputs (0, 1) for single value, not sum
- C) Tanh outputs (-1, 1)

---

### MCQ 14
**Question:** In backpropagation, error propagates from:

**Options:**
- A) Input layer to output layer
- B) Output layer to input layer
- C) Hidden layer to both input and output
- D) Randomly between layers

**✅ Correct Answer:** B

**📖 Explanation:** "BACK"propagation = Error calculated at output, propagated backward through layers.

**❌ Why Others Are Wrong:**
- A) That's forward pass direction
- C) Always starts from output
- D) Systematic, not random

---

### MCQ 15
**Question:** What does the term "fully connected" mean in neural networks?

**Options:**
- A) Only adjacent layers are connected
- B) Every neuron in one layer connects to every neuron in next layer
- C) Neurons in same layer are connected
- D) Only input and output are connected

**✅ Correct Answer:** B

**📖 Explanation:** "Fully connected" / "Dense" = Each neuron receives input from ALL neurons of previous layer.

**❌ Why Others Are Wrong:**
- A) Describes convolutional layers (local connections)
- C) Neurons in same layer don't connect to each other
- D) Hidden layers exist in between

---

## Section B: Multiple Select Questions (MSQ) - 10 Questions

### MSQ 1
**Question:** Which statements about backpropagation are TRUE? (Select ALL that apply)

**Options:**
- A) It calculates gradients of loss with respect to all weights and biases
- B) It starts from the output layer and moves backward
- C) It updates weights layer by layer from input to output
- D) The purpose is to minimize the loss function
- E) It uses gradient descent to update parameters

**✅ Correct Answers:** A, B, D, E

**📖 Explanation:**
- A) ✓ We need ∂L/∂W and ∂L/∂b for all parameters
- B) ✓ Error starts at output, propagates backward
- D) ✓ Ultimate goal is minimum loss
- E) ✓ Gradient descent uses calculated gradients

**❌ Why Others Are Wrong:**
- C) Updates go backward (output to input), not forward

---

### MSQ 2
**Question:** Select ALL TRUE statements about ReLU vs Sigmoid:

**Options:**
- A) ReLU avoids vanishing gradient problem
- B) Sigmoid always outputs values between 0 and 1
- C) ReLU outputs: max(0, x)
- D) Sigmoid derivative can be at most 0.25
- E) Sigmoid is best for multi-class output

**✅ Correct Answers:** A, B, C, D

**📖 Explanation:**
- A) ✓ ReLU gradient = 1 for positive
- B) ✓ Sigmoid formula ensures (0, 1) range
- C) ✓ Definition of ReLU
- D) ✓ Maximum at 0.5 × 0.5 = 0.25

**❌ Why Others Are Wrong:**
- E) Softmax is for multi-class, not sigmoid

---

### MSQ 3
**Question:** Which are components of a single forward pass? (Select ALL that apply)

**Options:**
- A) Matrix multiplication of weights and inputs
- B) Adding bias term
- C) Applying activation function
- D) Calculating loss gradient
- E) Computing predicted output

**✅ Correct Answers:** A, B, C, E

**📖 Explanation:**
- Forward pass: X → Z = WX + b → a = activation(Z) → ... → Y_hat

**❌ Why Others Are Wrong:**
- D) Gradient calculation is backward pass

---

### MSQ 4
**Question:** When comparing ANN and CNN for image processing, which are TRUE? (Select ALL)

**Options:**
- A) ANN requires many more parameters for large images
- B) CNN uses weight sharing through filters
- C) ANN is always faster than CNN
- D) CNN exploits spatial structure of images
- E) CNN filters are learned during backpropagation

**✅ Correct Answers:** A, B, D, E

**📖 Explanation:**
- A) ✓ 1000×1000 image = billions of ANN parameters
- B) ✓ Same filter slides across entire image
- D) ✓ Local patterns, spatial relationships
- E) ✓ Filter weights are trainable

**❌ Why Others Are Wrong:**
- C) CNN is often faster due to fewer parameters

---

### MSQ 5
**Question:** Which hyperparameters need to be decided before training? (Select ALL)

**Options:**
- A) Number of hidden layers
- B) Weight values
- C) Learning rate
- D) Number of neurons per layer
- E) Bias values
- F) Activation function type

**✅ Correct Answers:** A, C, D, F

**📖 Explanation:**
- Architecture decisions (A, D, F) and training decisions (C) are hyperparameters

**❌ Why Others Are Wrong:**
- B) Weights are learned, not hyperparameters
- E) Biases are learned during training

---

### MSQ 6
**Question:** Which cause the vanishing gradient problem? (Select ALL)

**Options:**
- A) Using sigmoid activation in many layers
- B) Using ReLU activation
- C) Very deep networks with small gradients
- D) Too small learning rate
- E) Chain rule multiplying many small numbers

**✅ Correct Answers:** A, C, E

**📖 Explanation:**
- Sigmoid max gradient = 0.25; multiply many → near zero
- Deep networks amplify this problem

**❌ Why Others Are Wrong:**
- B) ReLU gradient = 1, doesn't vanish
- D) Small LR is slow, but gradients still exist

---

### MSQ 7
**Question:** During training, which operations happen? (Select ALL)

**Options:**
- A) Forward pass to compute predictions
- B) Loss calculation
- C) Backward pass for gradients
- D) Weight updates based on gradients
- E) Model deployment to production

**✅ Correct Answers:** A, B, C, D

**📖 Explanation:**
Training loop: Forward → Loss → Backward → Update

**❌ Why Others Are Wrong:**
- E) Deployment is after training, not during

---

### MSQ 8
**Question:** Which are valid loss functions? (Select ALL)

**Options:**
- A) Mean Squared Error (MSE)
- B) Cross Entropy
- C) ReLU
- D) Binary Cross Entropy
- E) Categorical Cross Entropy

**✅ Correct Answers:** A, B, D, E

**📖 Explanation:**
All are legitimate loss functions for different tasks.

**❌ Why Others Are Wrong:**
- C) ReLU is activation function, not loss function

---

### MSQ 9
**Question:** What happens during inference? (Select ALL that apply)

**Options:**
- A) Forward pass is performed
- B) Predictions are generated
- C) Weights are updated
- D) Input data is processed through layers
- E) Gradients are calculated

**✅ Correct Answers:** A, B, D

**📖 Explanation:**
Inference = Forward pass only = Get predictions

**❌ Why Others Are Wrong:**
- C) Weights frozen during inference
- E) No gradients needed for prediction

---

### MSQ 10
**Question:** Which improve training stability? (Select ALL)

**Options:**
- A) Proper weight initialization
- B) Using appropriate learning rate
- C) Zero initialization everywhere
- D) Batch normalization
- E) Dropout regularization

**✅ Correct Answers:** A, B, D, E

**📖 Explanation:**
All techniques help training converge better.

**❌ Why Others Are Wrong:**
- C) Zero init causes symmetry problem

---

## Section C: Numerical/Calculation Questions - 8 Questions

### Numerical 1
**Question:** A neural network has:
- Layer 1: 784 inputs → 128 neurons (+ bias)
- Layer 2: 128 inputs → 64 neurons (+ bias)
- Layer 3: 64 inputs → 10 outputs (+ bias)

Calculate total trainable parameters.

**Solution:**
```
Layer 1: (784 × 128) + 128 = 100,352 + 128 = 100,480
Layer 2: (128 × 64) + 64 = 8,192 + 64 = 8,256
Layer 3: (64 × 10) + 10 = 640 + 10 = 650

Total = 100,480 + 8,256 + 650 = 109,386
```

**✅ Final Answer:** 109,386 parameters

---

### Numerical 2
**Question:** Calculate ReLU output for inputs: [-2, 3, -1, 5, 0]

**Solution:**
```
ReLU(x) = max(0, x)
ReLU(-2) = 0
ReLU(3) = 3
ReLU(-1) = 0
ReLU(5) = 5
ReLU(0) = 0
```

**✅ Final Answer:** [0, 3, 0, 5, 0]

---

### Numerical 3
**Question:** If sigmoid(0) = 0.5, what is the derivative σ'(0)?

**Solution:**
```
σ'(x) = σ(x) × (1 - σ(x))
σ'(0) = 0.5 × (1 - 0.5) = 0.5 × 0.5 = 0.25
```

**✅ Final Answer:** 0.25

---

### Numerical 4
**Question:** A 1000×1000 RGB image is input to a fully connected layer with 100 neurons. How many weights (excluding bias)?

**Solution:**
```
Input size = 1000 × 1000 × 3 = 3,000,000 pixels
Neurons = 100
Weights = 3,000,000 × 100 = 300,000,000
```

**✅ Final Answer:** 300 million weights

---

### Numerical 5
**Question:** If current weight W = 2.5, learning rate η = 0.1, and gradient ∂L/∂W = 0.5, what is W_new?

**Solution:**
```
W_new = W_old - η × ∂L/∂W
W_new = 2.5 - 0.1 × 0.5
W_new = 2.5 - 0.05 = 2.45
```

**✅ Final Answer:** 2.45

---

### Numerical 6
**Question:** In a network with 3 hidden layers using sigmoid, what is the maximum product of gradients through all layers?

**Solution:**
```
Max sigmoid gradient = 0.25
Through 3 layers: 0.25 × 0.25 × 0.25 = 0.25³ = 0.015625
```

**✅ Final Answer:** 0.015625 (approximately 0.016)

---

### Numerical 7
**Question:** MSE Loss for prediction Ŷ = 0.7 when true Y = 1?

**Solution:**
```
MSE = ½(Y - Ŷ)²
MSE = ½(1 - 0.7)²
MSE = ½(0.3)²
MSE = ½ × 0.09 = 0.045
```

**✅ Final Answer:** 0.045

---

### Numerical 8
**Question:** A network has 4 layers. Forward pass takes 0.5 seconds. If backward pass takes approximately the same time per layer, estimate total training time for one batch.

**Solution:**
```
Forward: 0.5 seconds
Backward: ~0.5 seconds (similar computation)
Loss calculation: ~0.01 seconds (minimal)
Total ≈ 0.5 + 0.5 = 1.0 second per batch
```

**✅ Final Answer:** ~1 second per batch

---

## Section D: Fill in the Blanks - 8 Questions

### Fill 1
**Question:** The formula Z = W×X + b is called a _______ combination.

**Answer:** Linear

---

### Fill 2
**Question:** ReLU function outputs _______ for negative inputs and _______ for positive inputs.

**Answer:** 0 ; the input value (x)

---

### Fill 3
**Question:** The weight update uses the formula: W_new = W_old - _______ × gradient.

**Answer:** Learning rate (η)

---

### Fill 4
**Question:** In backpropagation, we use the _______ rule to calculate gradients through multiple layers.

**Answer:** Chain

---

### Fill 5
**Question:** During _______, weights are frozen and only forward pass happens.

**Answer:** Inference

---

### Fill 6
**Question:** _______ activation outputs values that sum to 1 (probability distribution).

**Answer:** Softmax

---

### Fill 7
**Question:** The problem where gradients become extremely small in deep networks is called _______ gradient.

**Answer:** Vanishing

---

### Fill 8
**Question:** CNN uses _______ sharing to reduce the number of parameters compared to ANN.

**Answer:** Weight

---

## 📚 Quick Revision Points

### Key Formulas

| Formula | Description |
|---------|-------------|
| Z = WX + b | Linear combination at neuron |
| a = activation(Z) | Apply non-linearity |
| L = ½(Y - Ŷ)² | MSE Loss |
| ∂L/∂W = chain of ∂ | Backpropagation |
| W_new = W - η×∂L/∂W | Weight update |
| ReLU(x) = max(0, x) | ReLU activation |
| σ(x) = 1/(1+e⁻ˣ) | Sigmoid activation |
| σ'(x) = σ(x)(1-σ(x)) | Sigmoid derivative |

### Common Exam Traps

1. **Trap**: ReLU outputs 0-1 like sigmoid
   → **Truth**: ReLU outputs [0, ∞)

2. **Trap**: Backprop updates filter SIZES in CNN
   → **Truth**: Only filter WEIGHTS are updated

3. **Trap**: Forward pass calculates gradients
   → **Truth**: Forward = outputs, Backward = gradients

4. **Trap**: Zero weight initialization is valid
   → **Truth**: Causes symmetry problem

5. **Trap**: Inference needs backpropagation
   → **Truth**: Only forward pass needed

---

## 🚀 Section E: Shortcuts & Cheat Codes for Exam

### ⚡ One-Liner Shortcuts

| Concept | Shortcut |
|---------|----------|
| Forward Pass | X → Z → a → Y_hat |
| Backward Pass | L → ∂L/∂W → Update W |
| ReLU advantage | Gradient = 1, no vanishing |
| Sigmoid max derivative | 0.25 |
| Dense layer params | (inputs × neurons) + neurons |
| Training vs Inference | Training = both passes, Inference = forward only |

### 🎯 Memory Tricks

1. **BACK**prop = Error goes **BACK**ward
2. **ReLU** = **RE**ctified **L**inear **U**nit = max(0, x)
3. **η (eta)** = **E**asy **T**o **A**djust (learning rate)
4. **Softmax** = **Soft** probabilities that **max** sum to 1

### 🎓 Interview One-Liners

| Question | Answer |
|----------|--------|
| What is backprop? | "Algorithm to calculate gradients layer by layer using chain rule" |
| Why ReLU? | "Gradient is 1 for positive, avoids vanishing gradient" |
| Training vs Inference? | "Training = forward + backward, Inference = forward only" |
| Why chain rule? | "To decompose complex function derivative into simpler parts" |

### ⚠️ "If You Forget Everything, Remember This"

1. **Golden Rule 1**: Forward pass = compute output, Backward pass = compute gradients
2. **Golden Rule 2**: W_new = W_old - η × gradient (subtract to go downhill)
3. **Golden Rule 3**: ReLU > Sigmoid for hidden layers (no vanishing gradient)
4. **Golden Rule 4**: Inference = Forward pass ONLY (weights frozen)
5. **Golden Rule 5**: CNN filters learn weights, not sizes

---

## 🏆 Final Exam Success Checklist

- [ ] Explain forward pass step-by-step
- [ ] Write chain rule for backpropagation
- [ ] Calculate ReLU outputs for given inputs
- [ ] Explain why ReLU > Sigmoid
- [ ] Calculate parameters in multi-layer network
- [ ] Apply weight update formula
- [ ] Distinguish training vs inference
- [ ] Compare ANN vs CNN for images
- [ ] Explain vanishing gradient problem
- [ ] Calculate sigmoid derivative at x=0

---

**Good luck with your exam! 🍀**
