# Observations and Conclusion: Activation Functions Comparison

## Execution Output

### Generated Plots
1. **all_activations.png** - All 3 functions on same graph
2. **all_derivatives.png** - All 3 gradients, showing vanishing problem
3. **side_by_side_comparison.png** - 2x3 grid comparison

### Numerical Results

| Input | Sigmoid | Sig.Grad | Tanh | Tanh.Grad | ReLU | ReLU.Grad |
|-------|---------|----------|------|-----------|------|-----------|
| -5.0 | 0.0067 | 0.0066 | -0.9999 | 0.0002 | 0.0 | 0.0 |
| -2.0 | 0.1192 | 0.1050 | -0.9640 | 0.0707 | 0.0 | 0.0 |
| 0.0 | 0.5000 | 0.2500 | 0.0000 | 1.0000 | 0.0 | 0.0 |
| 2.0 | 0.8808 | 0.1050 | 0.9640 | 0.0707 | 2.0 | 1.0 |
| 5.0 | 0.9933 | 0.0066 | 0.9999 | 0.0002 | 5.0 | 1.0 |

---

## Key Observations

### 1. Gradient Comparison at z=0
| Function | Gradient | Interpretation |
|----------|----------|----------------|
| Sigmoid | 0.25 | Max, but only 25% |
| Tanh | 1.00 | Perfect at center |
| ReLU | 0.00 | Dead at zero |

### 2. Gradient at z=5
| Function | Gradient | Problem |
|----------|----------|---------|
| Sigmoid | 0.0066 | Vanishing! |
| Tanh | 0.0002 | Vanishing! |
| ReLU | 1.00 | Perfect! |

### 3. Key Trade-offs
- Sigmoid/Tanh: No dead neurons, but vanishing gradient
- ReLU: No vanishing gradient, but dead neurons

---

## Conclusion

### Summary of Findings
1. All three functions implemented from scratch
2. Visualizations clearly show gradient behavior
3. Numerical analysis confirms theoretical properties
4. Written analysis explains when to use each

### Recommendations

| Situation | Use |
|-----------|-----|
| Binary classification output | Sigmoid |
| Multi-class output | Softmax |
| Hidden layers (default) | ReLU |
| RNNs/LSTMs | Tanh |
| Dead neuron concern | LeakyReLU |

### Final Answer
**ReLU** is the modern default for hidden layers because it solved the vanishing gradient problem that made deep learning possible.
