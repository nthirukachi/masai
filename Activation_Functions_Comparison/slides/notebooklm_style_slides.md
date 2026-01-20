# Activation Functions Comparison - NotebookLM Style Slides

---

## Slide 1: Title & Objective

# Activation Functions Comparison
## Sigmoid vs Tanh vs ReLU

**Objective**: Compare all three major activation functions to understand when to use each.

---

## Slide 2: The Evolution

### Timeline

| Era | Function | Innovation |
|-----|----------|------------|
| 1990s | Sigmoid | First differentiable |
| 2000s | Tanh | Zero-centered |
| 2012+ | ReLU | No vanishing gradient |

ReLU enabled deep learning revolution!

---

## Slide 3: All Formulas

### Quick Reference

```
Sigmoid: 1/(1+e^-z)
Tanh: (e^z-e^-z)/(e^z+e^-z)
ReLU: max(0, z)
```

### Output Ranges
- Sigmoid: (0, 1)
- Tanh: (-1, 1)
- ReLU: [0, infinity)

---

## Slide 4: Gradient Comparison

### The Key Difference

| Function | Max Gradient | At All Positive? |
|----------|--------------|------------------|
| Sigmoid | 0.25 | Decays |
| Tanh | 1.0 | Decays |
| **ReLU** | **1.0** | **Constant!** |

ReLU never decays for positive inputs!

---

## Slide 5: Vanishing Gradient

### 10-Layer Network Gradient

| Function | Layer 1 Gradient |
|----------|------------------|
| Sigmoid | 0.00000095 |
| Tanh | ~0.0001 |
| **ReLU** | **1.0** |

ReLU enabled 100+ layer networks!

---

## Slide 6: Numerical Comparison

### Key Points (z = 5)

| Function | Output | Gradient |
|----------|--------|----------|
| Sigmoid | 0.993 | **0.007** (tiny!) |
| Tanh | 0.999 | **0.0002** (tiny!) |
| ReLU | 5.0 | **1.0** (perfect!) |

---

## Slide 7: Trade-offs

### The Key Trade-off

| Problem | Sigmoid/Tanh | ReLU |
|---------|--------------|------|
| Vanishing Gradient | YES | NO |
| Dead Neurons | NO | YES |

Dead neurons < Vanishing gradient (usually)

---

## Slide 8: When to Use Each

### Decision Tree

```
Binary output? -> Sigmoid
Multi-class? -> Softmax
Hidden layer? -> ReLU
RNN/LSTM? -> Tanh
Dead neurons? -> LeakyReLU
```

---

## Slide 9: Saturation Regions

### Where Learning Stops

| Function | Saturates When |
|----------|---------------|
| Sigmoid | \|z\| > 4 |
| Tanh | \|z\| > 3 |
| ReLU | z <= 0 (dead) |

---

## Slide 10: Interview Key Points

### Must Know

1. Sigmoid max gradient = 0.25
2. Tanh max gradient = 1.0 (but decays)
3. ReLU gradient = 1.0 for ALL positive
4. ReLU enabled deep learning
5. Dead neurons are the trade-off

---

## Slide 11: Conclusion

### The Bottom Line

- **Sigmoid**: Binary classification output only
- **Tanh**: RNNs, when zero-centered needed
- **ReLU**: Default for all hidden layers

### Why ReLU Won
Constant gradient = 1 for positive inputs enabled training of very deep networks.
