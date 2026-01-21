# ReLU vs Leaky ReLU: Investigating the Dying ReLU Problem

## 🎯 Project Objective

This teaching project investigates the **dying ReLU problem** by implementing both ReLU and Leaky ReLU activation functions from scratch, building a 2-layer neural network, and comparing their behavior.

## 🧠 What You'll Learn

- What is the "dying ReLU" problem and why it matters
- How ReLU and Leaky ReLU work mathematically
- Building a neural network from scratch with NumPy
- Forward and backward propagation implementation
- How to detect and count "dead neurons"
- When to choose Leaky ReLU over ReLU

## 📁 Project Structure

```
ReLU_vs_LeakyReLU_DyingNeurons/
├── notebook/                    # Teaching-oriented Jupyter Notebook
├── src/                         # Python source code
├── documentation/               # Detailed explanations and study materials
├── slides/                      # Presentation slides (MD + PDF)
├── outputs/                     # Generated plots and outputs
└── README.md                    # This file
```

## 🚀 Quick Start

```powershell
# Navigate to project
cd c:\masai\ReLU_vs_LeakyReLU_DyingNeurons

# Run the Python script
uv run python src/ReLU_vs_LeakyReLU_DyingNeurons.py
```

## 📊 Expected Outputs

1. **Training Loss Comparison Plot**: Shows loss curves for both ReLU and Leaky ReLU
2. **Dead Neuron Analysis**: Percentage of neurons that became "dead" for each activation
3. **Accuracy Comparison**: Final training accuracy for both versions
4. **Written Analysis**: 200-300 word comparison explaining results

## 🔑 Key Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| ReLU | Outputs the input if positive, else 0 (like a one-way door) |
| Leaky ReLU | Like ReLU but allows a small "leak" for negative values |
| Dying ReLU | When neurons stop learning because they always output 0 |
| Dead Neurons | Neurons with zero gradient that never update |

## 📚 Documentation

- `problem_statement.md` - Problem explained simply with analogies
- `concepts_explained.md` - Deep dive into all concepts
- `observations_and_conclusion.md` - Results and insights
- `interview_questions.md` - Common interview Q&A
- `exam_preparation.md` - MCQ, MSQ, Numerical questions
- `interview_preparation.md` - Quick revision cheat sheet

## 🎴 Slides

Open `slides/slides.md` or `slides/slides.pdf` for a visual presentation.

---

*Created as part of a comprehensive teaching project on neural network activation functions.*
