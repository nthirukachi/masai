# 🌾 AI-Based Crop Health Monitoring Using Drone Multispectral Data

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Teaching%20Project-green.svg)]()

## 📋 Project Overview

This teaching project demonstrates how to build an **end-to-end AI pipeline** to detect crop stress using multispectral vegetation indices derived from drone imagery. You will learn to train a machine learning model, generate spatial stress maps, and interpret results from agricultural and drone-operations perspectives.

### 🎯 What You Will Learn

| Topic | Description |
|-------|-------------|
| **Vegetation Indices** | NDVI, GNDVI, SAVI, EVI - How plants "speak" through light |
| **Machine Learning** | Random Forest classification for stress detection |
| **Spatial Analysis** | Creating field-level heatmaps from predictions |
| **Drone Operations** | Interpreting AI outputs for agricultural decisions |

---

## 🌱 Real-Life Analogy

> **Think of it like a doctor's checkup for plants!**
>
> Just like a doctor uses a thermometer to check if you have a fever, drones use special cameras to check if plants are "sick" (stressed). The cameras see colors that human eyes cannot see, and these colors tell us if the plant is healthy or needs help.

---

## 📁 Project Structure

```
AI_Crop_Health_Monitoring/
├── 📁 notebook/
│   └── ai_crop_health_monitoring.ipynb    # Teaching notebook with explanations
├── 📁 src/
│   └── ai_crop_health_monitoring.py       # Complete Python implementation
├── 📁 documentation/
│   ├── Original_Problem.md                # Original problem statement
│   ├── problem_statement.md               # Simplified explanation
│   ├── concepts_explained.md              # Deep dive into concepts
│   ├── observations_and_conclusion.md     # Results analysis
│   ├── interview_questions.md             # Q&A for interviews
│   ├── exam_preparation.md                # MCQ/MSQ/Numerical
│   └── interview_preparation.md           # Quick revision sheet
├── 📁 slides/
│   ├── slides.md                          # Markdown slides
│   └── slides.pdf                         # PDF presentation
├── 📁 outputs/
│   └── sample_outputs/                    # Generated visualizations
└── README.md                              # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- UV package manager

### Installation & Run

```powershell
# Navigate to project
cd c:\masai\AI_Crop_Health_Monitoring

# Initialize UV environment (if not already done)
uv init

# Install dependencies
uv add pandas numpy scikit-learn matplotlib seaborn

# Run the main script
uv run python src/ai_crop_health_monitoring.py
```

---

## 📊 Dataset Overview

| Feature | Description |
|---------|-------------|
| `ndvi_mean` | Average plant greenness (higher = healthier) |
| `gndvi` | Green-focused vegetation index |
| `savi` | Vegetation index adjusted for soil |
| `evi` | Enhanced vegetation index |
| `red_edge_1/2` | Plant stress indicators |
| `moisture_index` | Water content in plants |
| `grid_x`, `grid_y` | Location in the field |
| `crop_health_label` | **Target: Healthy or Stressed** |

---

## 📚 Learning Materials

1. **Start Here**: [problem_statement.md](documentation/problem_statement.md) - Understand the problem
2. **Deep Dive**: [concepts_explained.md](documentation/concepts_explained.md) - Learn each concept
3. **Practice**: [notebook/ai_crop_health_monitoring.ipynb](notebook/ai_crop_health_monitoring.ipynb) - Run the code
4. **Review**: [interview_preparation.md](documentation/interview_preparation.md) - Quick revision
5. **Test Yourself**: [exam_preparation.md](documentation/exam_preparation.md) - MCQ & exercises

---

## 🎓 Target Audience

- **Students** learning AI/ML for agriculture
- **Beginners** in data science
- **Professionals** exploring precision agriculture
- **Interview Preparation** seekers

---

## 📝 License

This is an educational project created for learning purposes.
