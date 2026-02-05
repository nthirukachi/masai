# MQ3_KMeans_Cluster_Quality

## 📌 Project Overview

This teaching project evaluates **K-Means cluster quality** using **inertia** and **silhouette analysis** on the **Iris dataset**.

---

## 🎯 What You Will Learn

1. **K-Means Clustering** - How to group similar data points together
2. **Feature Standardization** - Why we scale features before clustering
3. **Inertia (WCSS)** - Measuring cluster cohesion (tightness)
4. **Silhouette Score** - Measuring cluster separation (distinctness)
5. **Elbow Method** - Finding the optimal number of clusters

---

## 📁 Folder Structure

```
MQ3_KMeans_Cluster_Quality/
├── notebook/               # Teaching-oriented Jupyter notebook
├── src/                    # Python source code
├── documentation/          # Concept explanations, problem statement
├── slides/                 # NotebookLM-style presentation
├── outputs/                # Generated plots and metrics
└── README.md               # This file
```

---

## 🚀 How to Run

```powershell
# Navigate to project directory
cd c:\masai\MQ3_KMeans_Cluster_Quality

# Run with UV virtual environment
uv run python src/kmeans_cluster_quality.py
```

---

## 📊 Deliverables

| Deliverable | Description |
|-------------|-------------|
| Metrics Table | Inertia and Silhouette scores for k=2 to k=6 |
| Elbow Plot | Visual representation of inertia vs k |
| Silhouette Plot | Silhouette analysis for the chosen k |
| Justification | Written explanation for optimal k choice |

---

## 📚 Documentation Files

- `Original_Problem.md` - Exact problem statement
- `problem_statement.md` - Simplified explanation with analogy
- `concepts_explained.md` - Detailed concept breakdowns
- `observations_and_conclusion.md` - Results analysis
- `interview_questions.md` - Q&A for interviews
- `exam_preparation.md` - MCQ, MSQ, Numerical questions
- `interview_preparation.md` - Quick revision sheet

---

## 🌸 Dataset: Iris

The Iris dataset contains 150 flower samples with 4 features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Real species: 3 (Setosa, Versicolor, Virginica)

---

## ✍️ Author

Teaching Project created following the full-teaching-project workflow.
