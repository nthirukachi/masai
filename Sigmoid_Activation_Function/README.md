# Sigmoid Activation Function

## Project Overview

This project implements the **Sigmoid activation function** from scratch as part of Question 14: Compare Activation Functions Mathematically and Visually.

## Key Formula

```
σ(z) = 1 / (1 + e^(-z))
σ'(z) = σ(z) × (1 - σ(z))
```

## Project Structure

```
Sigmoid_Activation_Function/
├── notebook/
│   └── sigmoid_activation.ipynb
├── documentation/
│   ├── Original_Problem.md
│   ├── problem_statement.md
│   ├── concepts_explained.md
│   ├── observations_and_conclusion.md
│   └── interview_preparation.md
├── slides/
│   ├── notebooklm_style_slides.md
│   └── notebooklm_style_slides.pdf
├── src/
│   └── sigmoid_activation.py
├── outputs/
│   ├── sigmoid_function.png
│   ├── sigmoid_derivative.png
│   ├── sigmoid_combined.png
│   └── numerical_analysis.md
└── README.md
```

## Quick Start

### Run the Script

```powershell
python src/sigmoid_activation.py
```

### Expected Output
- 3 visualization plots in `outputs/`
- Numerical analysis table
- Gradient analysis at x = -2, 0, 2
- Written analysis of vanishing gradient

## Key Results

| Input | Sigmoid | Derivative |
|-------|---------|------------|
| -5 | 0.0067 | 0.0066 |
| 0 | 0.5000 | 0.2500 |
| 5 | 0.9933 | 0.0066 |

## Key Insights

1. **Maximum gradient is 0.25** (at z=0)
2. **Vanishing gradient** occurs for |z| > 4
3. **Use sigmoid** for binary classification output layer
4. **Don't use** in hidden layers of deep networks

## Files

- `src/sigmoid_activation.py` - Main implementation
- `documentation/` - Detailed explanations
- `slides/` - Presentation slides
- `outputs/` - Generated plots and analysis
