# MNIST Activation Functions Comparison

## 🎯 Project Overview

This teaching project demonstrates how different **activation functions** (Sigmoid, Tanh, ReLU) affect neural network performance when classifying handwritten digits using the MNIST dataset.

### What You'll Learn
- How Sigmoid, Tanh, and ReLU activation functions work
- Why ReLU typically trains faster than Sigmoid/Tanh
- The **vanishing gradient problem** and which activations suffer from it
- How to analyze and compare neural network performance

---

## 📁 Project Structure

```
MNIST_Activation_Functions_Comparison/
├── notebook/
│   └── mnist_activation_comparison.ipynb    # Teaching notebook
├── src/
│   └── mnist_activation_comparison.py       # Python implementation
├── documentation/
│   ├── Original_Problem.md                  # Exact problem statement
│   ├── problem_statement.md                 # Simplified explanation
│   ├── concepts_explained.md                # Core concepts (12 points each)
│   ├── observations_and_conclusion.md       # Results analysis
│   ├── interview_questions.md               # Q&A for interviews
│   ├── exam_preparation.md                  # MCQ/MSQ practice
│   └── interview_preparation.md             # Quick revision
├── slides/
│   ├── slides.md                            # Presentation slides
│   └── slides.pdf                           # PDF version
├── outputs/
│   └── [visualization files]                # Generated plots
└── README.md
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- UV package manager
- TensorFlow 2.x

### Run the Python Script
```powershell
cd c:\masai\MNIST_Activation_Functions_Comparison
uv run python src/mnist_activation_comparison.py
```

### Open the Jupyter Notebook
```powershell
cd c:\masai\MNIST_Activation_Functions_Comparison
uv run jupyter notebook notebook/mnist_activation_comparison.ipynb
```

---

## 📊 Expected Outputs

1. **Training History Plot** - Accuracy/loss curves for all 3 models
2. **Accuracy Comparison Bar Chart** - Final test accuracies
3. **Training Time Comparison** - Time per epoch for each model
4. **Gradient Magnitude Analysis** - Identifying vanishing gradients
5. **Deep Analysis Report** - 400-500 word analysis

---

## 🧠 Key Concepts

| Activation | Range | Pros | Cons |
|------------|-------|------|------|
| **Sigmoid** | (0, 1) | Smooth, probabilistic | Vanishing gradients |
| **Tanh** | (-1, 1) | Zero-centered | Vanishing gradients |
| **ReLU** | [0, ∞) | Fast, no vanishing | Dead neurons possible |

---

## 📚 Documentation

See the `documentation/` folder for:
- Detailed concept explanations
- Interview preparation materials
- Exam practice questions
- Observations and conclusions

---

## 👨‍💻 Author

Created as a teaching project for learning neural network activation functions.
