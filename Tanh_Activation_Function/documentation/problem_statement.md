# Problem Statement: Tanh Activation Function

## What Problem is Being Solved?

The **Tanh (Hyperbolic Tangent)** activation function addresses a key limitation of sigmoid: **non-zero-centered output**. 

When outputs are not centered around zero:
- Gradients push weights in the same direction
- Optimization zigzags towards the minimum
- Training becomes slower

Tanh solves this by outputting values in range **(-1, 1)**, centered around **0**.

## Why It Matters

| Issue with Sigmoid | How Tanh Fixes It |
|-------------------|-------------------|
| Output range (0, 1) | Output range (-1, 1) |
| Not zero-centered | Zero-centered at y=0 |
| Max gradient 0.25 | Max gradient 1.0 |
| Slower optimization | Faster optimization |

## Real-World Relevance

- **RNNs and LSTMs**: Tanh is often used in hidden states
- **Image Processing**: Normalizing pixel values to [-1, 1]
- **NLP**: Word embeddings often normalized with tanh
- **Older Neural Networks**: Before ReLU became popular

---

## Steps to Solve the Problem

### Step 1: Implement Tanh Function
- Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
- Alternative: tanh(z) = 2 * sigmoid(2z) - 1

### Step 2: Implement Tanh Derivative
- Formula: tanh'(z) = 1 - tanh^2(z)
- Maximum value of 1.0 at z=0 (4x better than sigmoid!)

### Step 3: Create Visualizations
- Plot tanh function over range [-6, 6]
- Plot derivative showing gradient behavior
- Annotate saturation regions

### Step 4: Numerical Analysis
- Calculate outputs for test inputs [-5, -2, -0.5, 0, 0.5, 2, 5]
- Analyze gradient strength at key points
- Compare with sigmoid

### Step 5: Written Analysis
- Explain vanishing gradient (still present but better)
- Compare with sigmoid
- Recommend use cases

---

## Expected Output

### Visualizations
1. **tanh_function.png**: S-shaped curve from -1 to 1 (zero-centered)
2. **tanh_derivative.png**: Bell-shaped curve, max 1.0 at z=0
3. **tanh_combined.png**: Side-by-side comparison

### Numerical Table

| Input (z) | Tanh(z) | Derivative |
|-----------|---------|------------|
| -5.0 | -0.999909 | 0.000182 |
| -2.0 | -0.964028 | 0.070651 |
| -0.5 | -0.462117 | 0.786448 |
| 0.0 | 0.000000 | 1.000000 |
| 0.5 | 0.462117 | 0.786448 |
| 2.0 | 0.964028 | 0.070651 |
| 5.0 | 0.999909 | 0.000182 |

### Key Insights
- Maximum gradient of 1.0 at z=0 (vs 0.25 for sigmoid)
- Zero-centered output improves optimization
- Still saturates for |z| > 3

---

## Exam Focus Points

1. **Formula**: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
2. **Output Range**: (-1, 1) - zero-centered
3. **Derivative**: tanh'(z) = 1 - tanh^2(z)
4. **Maximum Gradient**: 1.0 at z = 0
5. **Key Advantage**: Zero-centered output
6. **Relationship**: tanh(z) = 2 * sigmoid(2z) - 1
