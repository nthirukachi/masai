# Problem Statement: ReLU Activation Function

## What Problem is Being Solved?

Both sigmoid and tanh suffer from the **vanishing gradient problem** - gradients become tiny for extreme inputs, making deep networks hard to train.

**ReLU (Rectified Linear Unit)** solves this by having a **constant gradient of 1** for all positive inputs, enabling effective training of very deep networks.

## Why It Matters

| Problem | How ReLU Solves It |
|---------|-------------------|
| Vanishing gradient | Gradient = 1 for positive inputs |
| Slow computation | Just max(0, z) - no exponentials |
| Deep network training | Enables 100+ layer networks |
| Training speed | Converges faster than sigmoid/tanh |

## Real-World Relevance

- **CNNs (ImageNet, ResNet)**: Default activation in image classification
- **Transformers**: Used in feed-forward layers
- **Deep Learning**: Enabled training of very deep architectures
- **Modern AI**: Foundation of most neural network applications

---

## Steps to Solve the Problem

### Step 1: Implement ReLU Function
- Formula: f(z) = max(0, z)
- Extremely simple: return z if positive, else 0

### Step 2: Implement ReLU Derivative
- Formula: f'(z) = 1 if z > 0, else 0
- Binary: gradient flows (1) or blocked (0)

### Step 3: Create Visualizations
- Plot ReLU function (linear for positive, flat at 0 for negative)
- Plot derivative (step function)
- Highlight dead neuron region

### Step 4: Numerical Analysis
- Calculate outputs for test inputs
- Show gradient is 1 for ALL positive values
- Demonstrate dead neuron problem

### Step 5: Written Analysis
- Explain no vanishing gradient advantage
- Document dead neuron problem
- Compare with sigmoid and tanh

---

## Expected Output

### Numerical Table

| Input (z) | ReLU(z) | Derivative |
|-----------|---------|------------|
| -5.0 | 0.0 | 0.0 |
| -2.0 | 0.0 | 0.0 |
| -0.5 | 0.0 | 0.0 |
| 0.0 | 0.0 | 0.0 |
| 0.5 | 0.5 | 1.0 |
| 2.0 | 2.0 | 1.0 |
| 5.0 | 5.0 | 1.0 |

### Key Insights
- Gradient is EXACTLY 1 for ALL positive inputs
- No saturation, no decay - gradients flow perfectly
- Dead neurons: gradient = 0 for negative inputs

---

## Exam Focus Points

1. **Formula**: f(z) = max(0, z)
2. **Output Range**: [0, infinity)
3. **Derivative**: 1 if z > 0, else 0
4. **Key Advantage**: No vanishing gradient
5. **Key Limitation**: Dead neurons
6. **Primary Use**: Hidden layers in deep networks
