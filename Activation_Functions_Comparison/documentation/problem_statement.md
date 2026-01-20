# Problem Statement: Activation Functions Comparison

## What Problem is Being Solved?

Neural networks need **activation functions** to introduce non-linearity. Three main options have evolved:

1. **Sigmoid** (1990s) - First popular choice
2. **Tanh** (2000s) - Improvement over sigmoid
3. **ReLU** (2012+) - Modern standard

This project compares all three to understand when to use each.

## Why Comparison Matters

| Without Understanding | With Understanding |
|----------------------|-------------------|
| Random activation choice | Informed selection |
| Training failures | Effective training |
| Slow convergence | Fast convergence |
| Mysterious bugs | Clear reasoning |

---

## Expected Output

### 1. All Activations Plot
All three functions on same graph, range [-6, 6].

### 2. All Derivatives Plot
All three gradients on same graph, highlighting vanishing gradient.

### 3. Side-by-Side Comparison
2x3 grid showing each function with its derivative.

### 4. Numerical Table

| Input | Sigmoid | Tanh | ReLU |
|-------|---------|------|------|
| -5 | 0.0067 | -0.9999 | 0 |
| 0 | 0.5 | 0 | 0 |
| 5 | 0.9933 | 0.9999 | 5 |

### 5. Written Analysis (200-300 words)
- Vanishing gradient explanation
- Use case recommendations
- Saturation regions

---

## Exam Focus Points

1. Know ALL THREE formulas
2. Know max gradients: 0.25, 1.0, 1.0
3. Know trade-offs: vanishing gradient vs dead neurons
4. Know use cases for each
