# Numerical Comparison of All Activation Functions

## Complete Comparison Table

| Input | Sigmoid | Sig.Grad | Tanh | Tanh.Grad | ReLU | ReLU.Grad |
|-------|---------|----------|------|-----------|------|----------|
| -5.0 | 0.0067 | 0.0066 | -0.9999 | 0.0002 | 0.0 | 0.0 |
| -2.0 | 0.1192 | 0.1050 | -0.9640 | 0.0707 | 0.0 | 0.0 |
| -0.5 | 0.3775 | 0.2350 | -0.4621 | 0.7864 | 0.0 | 0.0 |
| 0.0 | 0.5000 | 0.2500 | 0.0000 | 1.0000 | 0.0 | 0.0 |
| 0.5 | 0.6225 | 0.2350 | 0.4621 | 0.7864 | 0.5 | 1.0 |
| 2.0 | 0.8808 | 0.1050 | 0.9640 | 0.0707 | 2.0 | 1.0 |
| 5.0 | 0.9933 | 0.0066 | 0.9999 | 0.0002 | 5.0 | 1.0 |

## Key Insights

| Property | Sigmoid | Tanh | ReLU | Winner |
|----------|---------|------|------|--------|
| Max Gradient | 0.25 | 1.0 | 1.0 | ReLU/Tanh |
| Gradient Decay | Yes | Yes | No | **ReLU** |
| Zero-Centered | No | Yes | No | Tanh |
| Dead Neurons | No | No | Yes | Sigmoid/Tanh |
| Speed | Slow | Slow | Fast | **ReLU** |
