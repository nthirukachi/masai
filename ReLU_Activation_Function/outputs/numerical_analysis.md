# ReLU Numerical Analysis

## Output Table

| Input (z) | ReLU(z) | Derivative |
|-----------|---------|------------|
| -5.0 | 0.0 | 0.0 |
| -2.0 | 0.0 | 0.0 |
| -0.5 | 0.0 | 0.0 |
| 0.0 | 0.0 | 0.0 |
| 0.5 | 0.5 | 1.0 |
| 2.0 | 2.0 | 1.0 |
| 5.0 | 5.0 | 1.0 |

## Gradient Analysis

| Point | Gradient | Status |
|-------|----------|--------|
| x = -2 | 0.0 | DEAD |
| x = 0 | 0.0 | DEAD |
| x = 2 | 1.0 | ACTIVE |
