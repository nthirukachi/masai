# Tanh Activation Function

## Project Overview

This project implements the **Tanh (Hyperbolic Tangent)** activation function from scratch as part of Question 14: Compare Activation Functions.

## Key Formula

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
tanh'(z) = 1 - tanh^2(z)
```

## Key Advantage Over Sigmoid

| Property | Sigmoid | Tanh |
|----------|---------|------|
| Output Range | (0, 1) | (-1, 1) |
| Zero-Centered | No | **Yes** |
| Max Gradient | 0.25 | **1.0** |

## Project Structure

```
Tanh_Activation_Function/
├── notebook/
├── documentation/
│   ├── Original_Problem.md
│   ├── problem_statement.md
│   ├── concepts_explained.md
│   ├── observations_and_conclusion.md
│   └── interview_preparation.md
├── slides/
│   └── notebooklm_style_slides.md
├── src/
│   └── tanh_activation.py
├── outputs/
│   ├── tanh_function.png
│   ├── tanh_derivative.png
│   └── tanh_combined.png
└── README.md
```

## Quick Start

```powershell
python src/tanh_activation.py
```

## Key Results

| Input | Tanh | Derivative |
|-------|------|------------|
| -2 | -0.964 | 0.071 |
| 0 | 0.000 | 1.000 |
| 2 | 0.964 | 0.071 |

## Key Insights

1. **Zero-centered** output helps optimization
2. **Max gradient 1.0** (4x better than sigmoid)
3. Still has **vanishing gradient** for |z| > 3
4. Use for hidden layers, RNNs, LSTMs
