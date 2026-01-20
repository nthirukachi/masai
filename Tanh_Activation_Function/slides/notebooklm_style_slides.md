# Tanh Activation Function - NotebookLM Style Slides

---

## Slide 1: Title & Objective

# Tanh Activation Function
## Zero-Centered Implementation from Scratch

**Objective**: Implement and understand tanh as a zero-centered alternative to sigmoid.

**Key Formula**: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

---

## Slide 2: Problem Statement

### Why Zero-Centered Matters

Sigmoid problem:
- All outputs positive (0 to 1)
- Gradients push weights same direction
- Optimization zigzags

**Tanh Solution**: Output (-1, 1) centered at 0!

---

## Slide 3: Real-World Use Case

### Where Tanh is Used

| Application | Why Tanh |
|-------------|----------|
| RNNs/LSTMs | Hidden state normalization |
| Image Processing | Normalize to [-1, 1] |
| NLP | Word embedding normalization |
| GANs | Generator output layer |

---

## Slide 4: Input Data / Inputs

### Test Configuration

**Input Range**: [-6, 6] for visualization

**Specific Test Points**:
```
z = [-5, -2, -0.5, 0, 0.5, 2, 5]
```

**Gradient Analysis**: x = -2, 0, 2

---

## Slide 5: Concepts Used

### Core Mathematical Concepts

1. **Tanh Function**: (e^z - e^(-z)) / (e^z + e^(-z))
2. **Tanh Derivative**: 1 - tanh^2(z)
3. **Zero-Centered**: Output mean = 0
4. **Relationship**: tanh(z) = 2*sigmoid(2z) - 1

---

## Slide 6: Concepts Breakdown

### Tanh Explained Simply

**Input**: Any number (-inf to +inf)
**Output**: Number between -1 and 1

**Analogy**: Like a volume dial that goes negative
- Very negative input -> Output near -1
- Zero -> Output is 0
- Very positive -> Output near +1

---

## Slide 7: Step-by-Step Flow

### Implementation Steps

```
1. Import NumPy
   |
2. Define tanh(z)
   |
3. Define tanh_derivative(z)
   |
4. Plot functions
   |
5. Compute numerical table
   |
6. Compare with sigmoid
   |
7. Document findings
```

---

## Slide 8: Code Logic Summary

### Key Functions

```python
def tanh(z):
    exp_z = np.exp(z)
    exp_neg_z = np.exp(-z)
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)

def tanh_derivative(z):
    t = tanh(z)
    return 1 - t ** 2
```

---

## Slide 9: Important Parameters

### Function Details

| Function | Max Value | At Point |
|----------|-----------|----------|
| tanh(z) | 1.0 | z -> +inf |
| tanh(z) | -1.0 | z -> -inf |
| tanh'(z) | 1.0 | z = 0 |

---

## Slide 10: Execution Output

### Numerical Results

| Input | Tanh | Derivative |
|-------|------|------------|
| -5.0 | -0.9999 | 0.0002 |
| -2.0 | -0.9640 | 0.0707 |
| 0.0 | **0.0000** | **1.0000** |
| 2.0 | 0.9640 | 0.0707 |
| 5.0 | 0.9999 | 0.0002 |

---

## Slide 11: Observations & Insights

### Key Findings

1. **Max gradient 1.0** (4x better than sigmoid!)
2. **Zero-centered** at origin
3. **Symmetric**: tanh(-z) = -tanh(z)
4. **Still vanishing gradient** for |z| > 3

---

## Slide 12: Advantages & Limitations

### Pros and Cons

| Advantages | Limitations |
|------------|-------------|
| Zero-centered | Vanishing gradient |
| Max gradient 1.0 | Saturation at |z|>3 |
| Bounded output | Expensive computation |
| Better than sigmoid | ReLU often preferred |

---

## Slide 13: Interview Key Takeaways

### Must-Know Points

1. **Formula**: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))
2. **Range**: (-1, 1)
3. **Max Gradient**: 1.0 at z = 0
4. **Zero-centered**: Yes!
5. **Relationship**: tanh = 2*sigmoid(2z) - 1

---

## Slide 14: Conclusion

### Summary

- Implemented tanh from scratch
- Max gradient 1.0 (4x sigmoid)
- Zero-centered output helps optimization
- Still has vanishing gradient for |z| > 3

### Use For
- RNNs/LSTMs
- When zero-centered output needed
