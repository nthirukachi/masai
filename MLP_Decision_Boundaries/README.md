# MLP Decision Boundaries: Activation Functions Comparison

## 🎯 Project Overview

This project demonstrates how different activation functions in neural networks create different **decision boundaries** on a non-linearly separable dataset.

### What You'll Learn
- How MLPClassifier works in scikit-learn
- How ReLU, Sigmoid (logistic), and Tanh activations differ in practice
- How to visualize and interpret decision boundaries
- Why activation function choice matters for non-linear data

---

## 📊 Dataset

**make_moons** from sklearn - a classic non-linearly separable dataset resembling two interleaving half-circles.

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
```

---

## 🧠 Models

Three MLPClassifier models with identical architecture but different activations:

| Model | Activation | Hidden Layer | Neurons |
|-------|------------|--------------|---------|
| Model 1 | ReLU | 1 | 8 |
| Model 2 | Logistic (Sigmoid) | 1 | 8 |
| Model 3 | Tanh | 1 | 8 |

All models use `random_state=42` for fair comparison.

---

## 📁 Project Structure

```
MLP_Decision_Boundaries/
├── README.md
├── notebook/
│   └── mlp_decision_boundaries.ipynb
├── src/
│   └── mlp_decision_boundaries.py
├── documentation/
│   ├── Original_Problem.md
│   ├── problem_statement.md
│   ├── concepts_explained.md
│   ├── observations_and_conclusion.md
│   ├── interview_questions.md
│   ├── exam_preparation.md
│   └── interview_preparation.md
├── slides/
│   ├── slides.md
│   └── slides.pdf
└── outputs/
    ├── decision_boundaries.png
    └── comparison_table.md
```

---

## 🚀 How to Run

### Using UV (Recommended)
```powershell
cd c:\masai
uv run python MLP_Decision_Boundaries/src/mlp_decision_boundaries.py
```

### Using Jupyter Notebook
```powershell
cd c:\masai
uv run jupyter lab MLP_Decision_Boundaries/notebook/mlp_decision_boundaries.ipynb
```

---

## 📈 Expected Output

1. **Visualization**: 3-subplot figure showing decision boundaries for each activation
2. **Accuracy Table**: Training accuracy comparison
3. **Written Analysis**: 250-350 word analysis of results
