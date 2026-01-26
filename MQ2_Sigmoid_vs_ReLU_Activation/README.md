# 🧠 Sigmoid vs ReLU Activation Comparison

## 📋 Project Overview

This teaching project compares **Sigmoid (Logistic)** and **ReLU** activation functions in a shallow neural network (MLP) trained on the **make_moons** dataset.

### 🎯 What You'll Learn

1. **Activation Functions** - How different neurons "activate" or "turn on"
2. **Convergence Speed** - How fast each network learns the pattern
3. **Decision Boundaries** - How each network draws lines to separate classes
4. **Gradient Behavior** - Why ReLU often trains faster than Sigmoid

---

## 📁 Project Structure

```
Sigmoid_vs_ReLU_Activation/
│
├── 📁 notebook/
│   └── sigmoid_vs_relu.ipynb          # Teaching notebook
│
├── 📁 src/
│   └── sigmoid_vs_relu.py             # Python implementation
│
├── 📁 documentation/
│   ├── Original_Problem.md            # Exact problem statement
│   ├── problem_statement.md           # Simplified explanation
│   ├── concepts_explained.md          # Deep dive into concepts
│   ├── observations_and_conclusion.md # Results analysis
│   ├── interview_questions.md         # Q&A for interviews
│   ├── exam_preparation.md            # MCQ/MSQ/Numerical
│   └── interview_preparation.md       # Quick revision
│
├── 📁 slides/
│   └── slides.md                      # NotebookLM-style presentation
│
├── 📁 outputs/
│   ├── loss_curves.png                # Combined loss plot
│   ├── confusion_matrices.png         # Both confusion matrices
│   └── metrics_table.md               # Accuracy comparison
│
└── README.md                          # This file
```

---

## 🚀 How to Run

### Using UV (Recommended)
```powershell
cd c:\masai\Sigmoid_vs_ReLU_Activation
uv run python src/sigmoid_vs_relu.py
```

### Using Regular Python
```powershell
cd c:\masai\Sigmoid_vs_ReLU_Activation
python src/sigmoid_vs_relu.py
```

---

## 📊 Key Findings

| Metric | Sigmoid (Logistic) | ReLU |
|--------|-------------------|------|
| Final Accuracy | TBD | TBD |
| Convergence Speed | Slower | Faster |
| Gradient Vanishing | Yes (common) | No |
| Training Iterations | ≤300 | ≤300 |

---

## 🎓 Prerequisites

- Python 3.8+
- scikit-learn
- matplotlib
- numpy

---

## 👨‍🏫 For Beginners

Think of activation functions like **light switches**:
- **Sigmoid**: Like a dimmer switch - smoothly goes from OFF to ON
- **ReLU**: Like a regular switch - OFF until a point, then fully ON

This project helps you understand which "switch" works better for different problems!
