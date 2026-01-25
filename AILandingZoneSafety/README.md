# 🛬 AI-Based Landing Zone Safety Classification

A comprehensive teaching project demonstrating machine learning for drone landing zone safety classification using aerial imagery-derived features.

## 🎯 Project Overview

This project classifies drone landing zones as **safe** or **unsafe** using terrain features extracted from aerial imagery. The goal is to support autonomous drone landing operations.

### Real-Life Analogy
Think of a **pilot** looking for a good place to land:
- 👀 Checks if ground is **flat** (slope)
- 🪨 Checks if it's **smooth** (roughness)
- 🌿 Avoids **plants/trees** (vegetation)
- 🚧 Avoids **obstacles** (object density)

Our AI does the same thing, but automatically!

## 📁 Project Structure

```
c:\masai\AILandingZoneSafety\
│
├── 📁 data/
│   └── landing_zone_data.csv         # Dataset
│
├── 📁 notebook/
│   └── landing_zone_safety.ipynb     # Teaching notebook
│
├── 📁 src/
│   └── landing_zone_safety.py        # Python implementation
│
├── 📁 documentation/
│   ├── Original_Problem.md           # Original problem statement
│   ├── problem_statement.md          # Simplified explanation
│   ├── concepts_explained.md         # Key concepts
│   ├── observations_and_conclusion.md
│   ├── interview_questions.md
│   ├── exam_preparation.md
│   └── interview_preparation.md
│
├── 📁 slides/
│   ├── slides.md                     # Presentation
│   └── slides.pdf                    # PDF version
│
├── 📁 outputs/
│   └── sample_outputs/               # Generated visualizations
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- UV package manager

### Setup
```powershell
cd c:\masai\AILandingZoneSafety
uv sync
```

### Run the Project
```powershell
uv run python src/landing_zone_safety.py
```

## 📊 Dataset Features

| Feature | Description | Safety Impact |
|---------|-------------|---------------|
| `slope_deg` | Slope angle (0-20°) | Steep = Unsafe |
| `roughness` | Surface roughness (0-1) | Rough = Unsafe |
| `edge_density` | Edge detection (0-1) | High = Obstacles |
| `ndvi_mean` | Vegetation index (0-1) | Dense = Unsafe |
| `shadow_fraction` | Shadow coverage (0-1) | High = Visibility |
| `brightness_std` | Brightness variation | High = Inconsistent |
| `object_density` | Obstacle density (0-1) | High = Collision |
| `confidence_score` | Detection confidence | Low = Uncertain |
| `label` | 1=Safe, 0=Unsafe | **Target** |

## 🎓 Learning Objectives

1. ✅ Understand terrain features affecting drone landing
2. ✅ Apply ML classification for safety assessment
3. ✅ Evaluate models using precision, recall, F1, ROC-AUC
4. ✅ Create spatial safety heatmaps
5. ✅ Interpret AI outputs for autonomous decision-making

## 📚 Capstone Tasks

| Task | Description |
|------|-------------|
| Task 1 | Data Understanding - Explore features |
| Task 2 | ML Model - Train & evaluate classifier |
| Task 3 | Spatial Analysis - Create safety heatmaps |
| Task 4 | Autonomy - Recommend landing strategies |
| Task 5 | Reflection - Discuss limitations |

## 🔧 Dependencies

- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning)

## 📄 License

MIT License - Educational Use
