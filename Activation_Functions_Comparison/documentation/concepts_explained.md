# Concepts Explained: Activation Functions Comparison

## The Big Picture

### Why Three Activation Functions?

Each solves different problems:

| Function | Era | Main Innovation |
|----------|-----|-----------------|
| Sigmoid | 1990s | First differentiable activation |
| Tanh | 2000s | Zero-centered output |
| ReLU | 2012+ | No vanishing gradient |

---

## Complete Comparison Table

| Property | Sigmoid | Tanh | ReLU |
|----------|---------|------|------|
| **Formula** | 1/(1+e^(-z)) | (e^z-e^(-z))/(e^z+e^(-z)) | max(0, z) |
| **Output Range** | (0, 1) | (-1, 1) | [0, inf) |
| **Zero-Centered** | No | Yes | No |
| **Max Gradient** | 0.25 | 1.0 | 1.0 |
| **Gradient Decay** | Yes | Yes | No |
| **Vanishing Gradient** | Severe | Moderate | None (positive) |
| **Dead Neurons** | No | No | Yes |
| **Saturation** | \|z\| > 4 | \|z\| > 3 | None |
| **Computation** | Slow | Slow | Fast |
| **Modern Use** | Output layer | RNNs | Hidden layers |

---

## Vanishing Gradient Explained

### The Problem

During backpropagation, gradients are **multiplied** across layers:

```
Final gradient = grad(L10) x grad(L9) x ... x grad(L1)
```

If each gradient < 1, the product shrinks exponentially.

### Visual Comparison

```
SIGMOID (10 layers):
1.0 -> 0.25 -> 0.0625 -> 0.0156 -> 0.0039 -> ... -> 0.00000095

TANH (10 layers):
1.0 -> 1.0 -> 0.1 -> 0.01 -> ... -> (still shrinks)

RELU (10 layers, positive path):
1.0 -> 1.0 -> 1.0 -> 1.0 -> ... -> 1.0 (no shrinking!)
```

---

## When to Use Each

### Sigmoid
- **Use for**: Binary classification OUTPUT layer
- **Why**: Outputs probability (0-1)
- **Examples**: Spam detection, disease prediction

### Tanh
- **Use for**: RNNs, LSTMs, when zero-centered needed
- **Why**: Zero-centered helps optimization
- **Examples**: Text generation, time series

### ReLU
- **Use for**: Hidden layers in deep networks (DEFAULT)
- **Why**: No vanishing gradient, fast computation
- **Examples**: CNNs, Transformers, most modern architectures

---

## Dead Neurons vs Vanishing Gradient

### Trade-off Summary

| Problem | Affects | Solution |
|---------|---------|----------|
| Vanishing Gradient (Sigmoid/Tanh) | All neurons weakly | Use ReLU |
| Dead Neurons (ReLU) | Some neurons completely | Use LeakyReLU |

**Key Insight**: Dead neurons affect SOME neurons completely; vanishing gradient affects ALL neurons weakly. In practice, dead neurons are usually the lesser problem.

---

## Exam & Interview Summary

### Quick Formulas
```
Sigmoid: 1/(1+e^(-z)), derivative: s(1-s), max=0.25
Tanh: (e^z-e^-z)/(e^z+e^-z), derivative: 1-t^2, max=1.0
ReLU: max(0,z), derivative: 0 or 1
```

### Decision Tree
```
Need probability output? -> Sigmoid
Need bounded output [-1,1]? -> Tanh
Hidden layer in deep network? -> ReLU
Dead neurons a problem? -> LeakyReLU
```
