
# Metrics Comparison: Sigmoid vs ReLU Activation

| Metric | Sigmoid (Logistic) | ReLU |
|--------|-------------------|------|
| **Accuracy** | 0.8750 (87.50%) | 0.9583 (95.83%) |
| **Final Loss** | 0.3060 | 0.1105 |
| **Iterations Used** | 273 | 300 |
| **Converged Within 300** | ✅ Yes | ✅ Yes |

## Confusion Matrix Summary

### Sigmoid Activation
```
[[104  15]
 [ 15 106]]
```

### ReLU Activation
```
[[112   7]
 [  3 118]]
```

## Key Observations

1. **Accuracy Comparison**: ReLU is better
2. **Convergence Speed**: Sigmoid converged faster
3. **Final Loss**: ReLU achieved lower loss
