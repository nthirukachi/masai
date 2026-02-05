# MQ5: K-Means Feature Augmentation + Perceptron Cross-Validation

A teaching project demonstrating how to combine **unsupervised learning (K-Means clustering)** with **supervised learning (Perceptron)** and evaluate performance using **stratified 5-fold cross-validation**.

## 🎯 Objective

Determine if augmenting features with K-Means cluster information (one-hot membership + centroid distances) improves Perceptron classification performance on the Wine dataset.

## 📊 Dataset

- **Source:** scikit-learn's `load_wine`
- **Samples:** 178
- **Features:** 13 (chemical measurements)
- **Original Classes:** 3 (wine cultivars)
- **Binary Task:** Class 0 (positive) vs. Classes 1 & 2 (negative)

## 🔧 Key Concepts

| Concept | Description |
|---------|-------------|
| **K-Means Clustering** | Unsupervised algorithm that groups data into k clusters |
| **Perceptron** | Simplest neural network (single-layer, linear classifier) |
| **Feature Augmentation** | Adding new features derived from clustering |
| **Stratified Cross-Validation** | Preserves class distribution in each fold |
| **Data Leakage Prevention** | K-Means fit ONLY on training data per fold |

## 📁 Project Structure

```
MQ5_KMeans_Perceptron_CrossVal/
├── notebook/
│   └── kmeans_perceptron_crossval.ipynb    # Teaching notebook
├── src/
│   └── kmeans_perceptron_crossval.py       # Python implementation
├── documentation/
│   ├── Original_Problem.md                  # Raw problem statement
│   ├── problem_statement.md                 # Simplified explanation
│   ├── concepts_explained.md                # 12-point concept breakdown
│   ├── observations_and_conclusion.md       # Results analysis
│   ├── interview_questions.md               # Q&A for interviews
│   ├── exam_preparation.md                  # MCQ/MSQ/Numerical
│   └── interview_preparation.md             # Quick revision sheet
├── slides/
│   └── slides.md                            # NotebookLM-style presentation
├── outputs/
│   ├── cross_validation_metrics.csv         # Metric table
│   └── comparison_plot.png                  # Bar chart comparison
└── README.md                                # This file
```

## 🚀 Quick Start

```powershell
# Navigate to project directory
cd c:\masai\MQ5_KMeans_Perceptron_CrossVal

# Run Python script with UV
uv run python src/kmeans_perceptron_crossval.py

# Or run the Jupyter notebook
uv run jupyter notebook notebook/kmeans_perceptron_crossval.ipynb
```

## 📈 Deliverables

1. **Cross-Validation Metric Table** - Fold-wise Accuracy, F1, Average Precision
2. **Comparison Plots** - Bar charts with error bars
3. **Executive Summary** - 400-450 word recommendation for production

## 🎓 Learning Outcomes

After completing this project, you will understand:

- How to combine unsupervised and supervised learning
- Why K-Means must be fit per fold (data leakage prevention)
- How to create and evaluate feature augmentation pipelines
- How to use stratified cross-validation for imbalanced data
- How to perform statistical significance testing
- How to write production recommendations based on evidence

## 📝 Author

Created as part of the Masai Teaching Project Series.
