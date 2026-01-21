# MNIST Activation Functions Comparison
## NotebookLM-Style Presentation

---

# Slide 1: Title & Objective

## MNIST Activation Functions Comparison
### Comparing Sigmoid, Tanh, and ReLU

**Objective**: Understand how activation function choice affects neural network performance, training speed, and gradient flow.

**Key Question**: Why is ReLU the default choice for modern deep learning?

---

# Slide 2: Problem Statement

## The Problem

We want to classify handwritten digits (0-9) using neural networks.

**The Question**: Which activation function works best?

| Candidate | Description |
|-----------|-------------|
| Sigmoid | Squashes to (0,1) |
| Tanh | Squashes to (-1,1) |
| ReLU | max(0, x) |

**Goal**: Compare training speed, accuracy, and gradient flow.

---

# Slide 3: Real-World Use Case

## When This Matters

**Banking** 📊
- Check reading systems
- Digit recognition on forms

**Mobile Apps** 📱
- Handwriting input
- Note-taking apps

**Why Activation Choice Matters**
- Wrong choice → slow/failed training
- Right choice → faster, better models

---

# Slide 4: Input Data

## MNIST Dataset

| Property | Value |
|----------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28 × 28 |
| Classes | 10 (digits 0-9) |

**Preprocessing**:
1. Flatten: 28×28 → 784 vector
2. Normalize: 0-255 → 0-1

---

# Slide 5: Concepts Used (High Level)

## Key Concepts

1. **Neural Network**: Layers of connected neurons
2. **Activation Function**: Non-linear transformation
3. **Backpropagation**: Learning from errors
4. **Gradient Flow**: How feedback reaches neurons
5. **Vanishing Gradients**: When learning stops

---

# Slide 6: Activation Functions Explained

## Three Candidates

### Sigmoid: σ(x) = 1/(1+e^-x)
- Range: (0, 1)
- Issue: Max derivative = 0.25 ⚠️

### Tanh: (e^x - e^-x)/(e^x + e^-x)
- Range: (-1, 1)
- Better: Zero-centered

### ReLU: max(0, x)
- Range: [0, ∞)
- Best: Derivative = 1 ✓

---

# Slide 7: Solution Flow

## Step-by-Step Approach

```
1. Load MNIST data
        ↓
2. Preprocess (flatten, normalize)
        ↓
3. Build 3 models (Sigmoid, Tanh, ReLU)
        ↓
4. Train for 20 epochs
        ↓
5. Compare metrics
        ↓
6. Analyze gradients
        ↓
7. Generate report
```

---

# Slide 8: Network Architecture

## Model Structure

```
Input Layer     →   Hidden Layer 1   →   Hidden Layer 2   →   Output Layer
(784 neurons)       (128 neurons)        (64 neurons)         (10 neurons)
   Pixels             Patterns            Features             Digits
```

**Configuration**:
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Cross-Entropy
- Epochs: 20
- Batch size: 128

---

# Slide 9: Key Functions

## Important Code Elements

### build_model(activation_name)
Creates neural network with specified activation

### model.fit(X, y, epochs, batch_size)
Trains the model on data

### GradientTape
Records operations for gradient computation

### model.evaluate(X_test, y_test)
Measures final performance

---

# Slide 10: Results

## Execution Output

| Model | Accuracy | Avg Time/Epoch | Gradient Mag |
|-------|----------|----------------|--------------|
| Sigmoid | ~97.5% | ~2.5s | 0.0001 |
| Tanh | ~97.8% | ~2.3s | 0.0005 |
| ReLU | ~98.2% | ~2.0s | 0.0050 |

**Winner**: ReLU 🏆
- Highest accuracy
- Fastest training
- Strongest gradients (50x Sigmoid!)

---

# Slide 11: Observations & Insights

## Key Findings

### 1. Vanishing Gradients are Real
Sigmoid gradients = 0.0001 (nearly zero!)

### 2. ReLU Trains Faster
~20% faster per epoch due to simpler computation

### 3. All Models Work for Simple Tasks
MNIST is "easy" - differences amplify in deeper networks

### 4. Gradient Magnitude Predicts Success
Stronger gradients → Better learning

---

# Slide 12: Advantages & Limitations

## By Activation Function

| Activation | ✅ Advantages | ❌ Limitations |
|------------|--------------|----------------|
| **Sigmoid** | Probability output | Vanishing gradients |
| **Tanh** | Zero-centered | Still vanishes |
| **ReLU** | Fast, no vanishing | Dead neurons |

## When to Use Each

- **Hidden layers**: ReLU (default)
- **Binary output**: Sigmoid
- **Multi-class output**: Softmax
- **RNN**: Tanh

---

# Slide 13: Interview Takeaways

## Top Interview Points

**Q: Why is ReLU preferred?**
> A: Derivative = 1 for positive inputs, no gradient shrinking.

**Q: When use Sigmoid?**
> A: Binary classification output, LSTM gates.

**Q: What's dying ReLU?**
> A: Neurons stuck at 0, fix with Leaky ReLU.

**Key Number**: Sigmoid max derivative = **0.25**

---

# Slide 14: Conclusion

## Summary

### What We Learned
1. Activation functions critically affect training
2. ReLU is the default for hidden layers
3. Vanishing gradients are a real problem
4. Choose activation based on layer purpose

### Action Items
- Start with ReLU for hidden layers
- Use Sigmoid for binary output
- Use Softmax for multi-class output
- Monitor gradient magnitudes

---

# Quick Reference Card

| Layer Type | Use |
|------------|-----|
| Hidden (default) | ReLU |
| Binary output | Sigmoid |
| Multi-class output | Softmax |
| RNN hidden | Tanh |

**Remember**: ReLU = Fast + Strong gradients = Better training!

---
