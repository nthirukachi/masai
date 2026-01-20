# Activation Functions Comparison

## Project Overview

This project compares **Sigmoid, Tanh, and ReLU** activation functions side-by-side, addressing Question 14: Compare Activation Functions Mathematically and Visually.

## Quick Reference

| Function | Formula | Max Gradient | Use Case |
|----------|---------|--------------|----------|
| Sigmoid | 1/(1+e^-z) | 0.25 | Binary output |
| Tanh | (e^z-e^-z)/(e^z+e^-z) | 1.0 | RNNs |
| ReLU | max(0, z) | 1.0 always | Hidden layers |

## Project Structure

```
Activation_Functions_Comparison/
├── documentation/
│   ├── Original_Problem.md
│   ├── problem_statement.md
│   ├── concepts_explained.md
│   ├── observations_and_conclusion.md
│   └── interview_preparation.md
├── slides/
│   └── notebooklm_style_slides.md
├── src/
│   └── activation_comparison.py
├── outputs/
│   ├── all_activations.png
│   ├── all_derivatives.png
│   └── side_by_side_comparison.png
└── README.md
```

## Quick Start

```powershell
python src/activation_comparison.py
```

## Key Insight

**ReLU revolutionized deep learning** because its gradient is constant at 1 for all positive inputs, while sigmoid (max 0.25) and tanh (max 1.0 at z=0 only) suffer from vanishing gradient.

## Related Projects

- [Sigmoid_Activation_Function](../Sigmoid_Activation_Function/)
- [Tanh_Activation_Function](../Tanh_Activation_Function/)
- [ReLU_Activation_Function](../ReLU_Activation_Function/)
