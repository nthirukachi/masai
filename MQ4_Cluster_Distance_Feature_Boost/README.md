# MQ4: Cluster-Distance Feature Boost

## 📋 Project Overview

This teaching project demonstrates how cluster-distance features from K-Means clustering can dramatically improve a simple Perceptron classifier's performance.

**Key Result:** Accuracy improved from **57.7%** to **92.4%** (+35 percentage points)

---

## 🎯 Problem Statement

A binary Perceptron cannot effectively separate one cluster from others in 2D space. By adding distance-to-centroid features, we give the classifier extra information about cluster geometry.

---

## 📁 Project Structure

```
MQ4_Cluster_Distance_Feature_Boost/
├── notebook/
│   └── cluster_distance_feature_boost.ipynb
├── src/
│   └── cluster_distance_feature_boost.py
├── documentation/
│   ├── Original_Problem.md
│   ├── problem_statement.md
│   ├── concepts_explained.md
│   ├── observations_and_conclusion.md
│   ├── interview_questions.md
│   ├── exam_preparation.md
│   └── interview_preparation.md
├── slides/
│   └── slides.md
├── outputs/
└── README.md
```

---

## 🚀 Quick Start

### Run with UV

```powershell
cd c:\masai\MQ4_Cluster_Distance_Feature_Boost
uv run python src/cluster_distance_feature_boost.py
```

### Expected Output

```
RESULTS: AVERAGED OVER 5 RANDOM SPLITS

Metric           Baseline     Enhanced    Improvement
----------------------------------------------------
ACCURACY           0.5769       0.9244       +34.76% [OK]
PRECISION          0.3166       0.8967       +58.01% [OK]
RECALL             0.6065       0.8857       +27.93% [OK]
ROC_AUC            0.4899       0.9849       +49.50% [OK]

[OK] SUCCESS: At least one metric improved by >=5 percentage points!
```

---

## 📊 Key Metrics

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| Accuracy | 57.7% | 92.4% | **+34.8%** |
| Precision | 31.7% | 89.7% | **+58.0%** |
| Recall | 60.6% | 88.6% | **+28.0%** |
| ROC AUC | 49.0% | 98.5% | **+49.5%** |

---

## 🔑 Key Concepts

1. **make_blobs** - Generate synthetic clustered data
2. **StandardScaler** - Normalize features
3. **K-Means** - Find cluster centers
4. **transform()** - Compute distance features
5. **Perceptron** - Simple linear classifier

---

## 💡 Why It Works

Distance-to-centroid features capture **cluster geometry** that raw coordinates cannot express. A linear boundary in the enhanced 5D space maps to a non-linear boundary in the original 2D space.

---

## 📚 Documentation

- **Concepts Explained:** Deep dive into each technique
- **Interview Questions:** 20 Q&A with diagrams
- **Exam Preparation:** MCQ, MSQ, Numerical questions
- **Interview Prep:** Quick revision sheet

---

## ✅ Success Criteria

- [x] Enhanced model improves at least one metric by ≥5%
- [x] Explanation references cluster geometry
- [x] Explanation references boundary shifts
