# Concepts Explained: Tanh Activation Function

## 1. Hyperbolic Tangent (Tanh)

### Definition
The **Tanh** (hyperbolic tangent) function is defined as:

tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

**Exam-friendly wording**: "A zero-centered activation function that maps inputs to the range (-1, 1)."

### Why It Is Used
- **Problem it solves**: Sigmoid outputs are not centered around zero
- **Why needed**: Zero-centered outputs improve gradient flow during optimization
- **Advantage**: Maximum gradient is 1.0 (4x better than sigmoid's 0.25)

### When to Use It
- RNNs and LSTMs (hidden states)
- When zero-centered outputs are beneficial
- Image normalization to [-1, 1]
- Older neural network architectures

### Where to Use It
- Natural Language Processing (word embeddings)
- Time series prediction
- Sequence-to-sequence models
- Generative models (early GANs)

### Is This the Only Way?

| Approach | Pros | Cons |
|----------|------|------|
| Sigmoid | Probability output | Not zero-centered, max grad 0.25 |
| **Tanh** | Zero-centered, max grad 1.0 | Still vanishing gradient |
| ReLU | No vanishing for positive | Dead neurons, not bounded |
| LeakyReLU | No dead neurons | Not bounded, not zero-centered |

**Why Tanh is chosen**: When zero-centered output is crucial and bounded output is needed.

### How to Use It

```python
import numpy as np

def tanh(z):
    exp_z = np.exp(z)
    exp_neg_z = np.exp(-z)
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)

# Examples
tanh(0)    # Returns 0.0 (center point)
tanh(2)    # Returns 0.964
tanh(-2)   # Returns -0.964
```

### How It Works Internally

```
For z = 1:
1. Compute e^1 = 2.718
2. Compute e^(-1) = 0.368
3. Numerator: 2.718 - 0.368 = 2.350
4. Denominator: 2.718 + 0.368 = 3.086
5. Result: 2.350 / 3.086 = 0.762
```

### Visual Summary
- Input: Any real number (-inf to +inf)
- Output: Strictly between -1 and 1
- Center point: tanh(0) = 0
- Symmetric: tanh(-z) = -tanh(z)

### Advantages
- Zero-centered output
- Maximum gradient is 1.0
- Symmetric around origin
- Bounded output

### Disadvantages / Limitations
- Still has vanishing gradient (for |z| > 3)
- Computationally expensive (exponentials)
- Saturates faster than sigmoid

### Exam & Interview Points

**Key Points to Memorize:**
1. Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
2. Output range: (-1, 1)
3. Derivative: tanh'(z) = 1 - tanh^2(z)
4. Maximum derivative: 1.0 at z = 0
5. Relationship: tanh(z) = 2 * sigmoid(2z) - 1

**Common Questions:**
- Q: What is the main advantage of tanh over sigmoid?
- A: Zero-centered output, which helps with optimization

- Q: What is tanh(0)?
- A: Exactly 0 (the center point)

---

## 2. Tanh Derivative

### Definition
The derivative of tanh is:

tanh'(z) = 1 - tanh^2(z)

### Why It Is Used
- Required for backpropagation
- Determines how fast the network learns at each point
- Maximum of 1.0 means no gradient shrinkage at center

### How to Use It

```python
def tanh_derivative(z):
    t = tanh(z)
    return 1 - t ** 2

# Examples
tanh_derivative(0)   # Returns 1.0 (maximum!)
tanh_derivative(2)   # Returns 0.0707
tanh_derivative(-2)  # Returns 0.0707
```

### How It Works Internally

For z = 0:
1. Compute tanh(0) = 0
2. Square it: 0^2 = 0
3. Subtract from 1: 1 - 0 = 1.0

For z = 2:
1. Compute tanh(2) = 0.964
2. Square it: 0.964^2 = 0.929
3. Subtract from 1: 1 - 0.929 = 0.071

### Exam & Interview Points

**Key Points:**
- Maximum value is 1.0 (not 0.25 like sigmoid!)
- Occurs at z = 0
- Rapidly decreases for |z| > 2
- For |z| > 3, gradient nearly zero

---

## 3. Zero-Centered Output

### Definition
An output is **zero-centered** if its mean is 0. Tanh outputs range from -1 to 1 with center at 0.

### Why It Matters

With sigmoid (not zero-centered):
- All outputs are positive (0 to 1)
- Gradients for weights are all same sign
- Weights only update in one direction
- Optimization zigzags to minimum

With tanh (zero-centered):
- Outputs can be positive or negative
- Gradients can have mixed signs
- More efficient optimization path

### Visual Explanation

```
Sigmoid optimization path (zigzag):
       /\
      /  \
     /    \
    /      \---> Minimum

Tanh optimization path (more direct):
    \
     \
      \---> Minimum
```

---

## 4. Comparison: Tanh vs Sigmoid

### Side-by-Side

| Property | Sigmoid | Tanh |
|----------|---------|------|
| Formula | 1/(1+e^(-z)) | (e^z-e^(-z))/(e^z+e^(-z)) |
| Output Range | (0, 1) | (-1, 1) |
| Center Point | 0.5 | 0 |
| Zero-centered | No | Yes |
| Max Gradient | 0.25 | 1.0 |
| f(0) | 0.5 | 0 |
| Saturation | \|z\| > 4 | \|z\| > 3 |
| Relationship | -- | tanh(z) = 2*sigmoid(2z) - 1 |

### Mathematical Relationship

tanh(z) = 2 * sigmoid(2z) - 1

This means:
- Tanh is a scaled and shifted version of sigmoid
- They share the same fundamental S-curve shape
- Tanh is "stretched" to fill (-1, 1)

---

## 5. Saturation in Tanh

### Definition
**Saturation** occurs when the output approaches the extreme values (-1 or +1) and the gradient becomes nearly zero.

### Where It Occurs
- **Left saturation**: z < -3 (output approaches -1)
- **Right saturation**: z > 3 (output approaches +1)

### Impact on Learning
- Gradients become very small (~0.001)
- Weight updates are negligible
- Learning effectively stops for saturated neurons

### Why Tanh Saturates Faster Than Sigmoid
- Sigmoid saturates at |z| > 4
- Tanh saturates at |z| > 3
- Tanh's steeper gradient causes earlier saturation
