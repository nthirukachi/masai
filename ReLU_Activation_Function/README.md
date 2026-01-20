# ReLU Activation Function

## Project Overview

This project implements the **ReLU (Rectified Linear Unit)** activation function from scratch - the activation that revolutionized deep learning.

## Key Formula

```
f(z) = max(0, z)
f'(z) = 1 if z > 0, else 0
```

## Why ReLU Matters

| Before ReLU | After ReLU |
|-------------|------------|
| Vanishing gradients | Gradient = 1 (always!) |
| 2-3 layer networks | 100+ layer networks |
| Slow training | Fast convergence |

## Project Structure

```
ReLU_Activation_Function/
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
│   └── relu_activation.py
├── outputs/
└── README.md
```

## Quick Start

```powershell
python src/relu_activation.py
```

## Key Results

| Input | ReLU | Derivative |
|-------|------|------------|
| -5 | 0 | 0 (dead) |
| 0 | 0 | 0 |
| 5 | 5 | 1 (perfect) |

## Key Insights

1. **Gradient = 1** for ALL positive inputs
2. **Dead neurons** for negative inputs
3. **Default** for hidden layers in modern networks
4. Use **LeakyReLU** if dead neurons are a problem
