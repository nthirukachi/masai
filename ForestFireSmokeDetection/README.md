# 🔥 AI-Based Forest Fire & Smoke Detection Using Aerial Imagery

## 📋 Project Overview

This capstone project implements an **end-to-end AI pipeline** to detect forest fire and smoke regions from aerial imagery using feature-level analysis. The project uses machine learning techniques for drone-based disaster monitoring.

## 🎯 Objectives

1. **Understand** visual indicators of fire and smoke in aerial imagery
2. **Apply** supervised machine learning for disaster detection
3. **Evaluate** model reliability using precision, recall, F1-score, and ROC-AUC
4. **Perform** spatial aggregation and risk visualization
5. **Interpret** AI outputs for drone-based emergency response

## 📁 Project Structure

```
ForestFireSmokeDetection/
│
├── 📁 notebook/
│   └── ForestFireSmokeDetection.ipynb    # Teaching-oriented Jupyter Notebook
│
├── 📁 documentation/
│   ├── Original_Problem.md               # Exact problem statement
│   ├── problem_statement.md              # Simplified explanation
│   ├── concepts_explained.md             # Core concepts (12 points each)
│   ├── observations_and_conclusion.md    # Results analysis
│   ├── interview_questions.md            # 10-20 Q&A
│   ├── exam_preparation.md               # MCQ/MSQ/Numerical
│   └── interview_preparation.md          # Quick revision
│
├── 📁 slides/
│   ├── slides.md                         # NotebookLM-style slides
│   └── slides.pdf                        # PDF version
│
├── 📁 src/
│   └── ForestFireSmokeDetection.py       # Complete Python implementation
│
├── 📁 outputs/
│   ├── execution_output.md               # Captured outputs
│   └── sample_outputs/                   # Generated visualizations
│
└── README.md                             # This file
```

## 📊 Dataset Features

| Feature | Description | Relevance to Fire/Smoke |
|---------|-------------|------------------------|
| mean_red | Average red channel intensity | Fire appears red/orange |
| mean_green | Average green channel intensity | Healthy vegetation is green |
| mean_blue | Average blue channel intensity | Sky/water reference |
| red_blue_ratio | Ratio of red to blue | High ratio indicates fire |
| intensity_std | Standard deviation of intensity | Fire has high variability |
| edge_density | Density of edges in tile | Smoke has blurred edges |
| smoke_whiteness | How white/gray the tile is | Smoke appears white/gray |
| haze_index | Amount of haze/fog effect | Smoke creates haze |
| hot_pixel_fraction | Fraction of very bright pixels | Fire creates hot spots |
| local_contrast | Contrast within tile | Fire creates contrast |

## 🚀 Running the Project

### Using UV (Recommended)

```powershell
# Navigate to project directory
cd c:\masai\ForestFireSmokeDetection

# Run the Python script
uv run python src/ForestFireSmokeDetection.py
```

### Running the Notebook

```powershell
# Start Jupyter
uv run jupyter notebook notebook/ForestFireSmokeDetection.ipynb
```

## 📈 Capstone Tasks

| Task | Description |
|------|-------------|
| Task 1 | Data Understanding - Explore dataset and explain feature relevance |
| Task 2 | ML Model - Train classifier with full evaluation metrics |
| Task 3 | Spatial Risk Analysis - Generate fire-risk heatmaps |
| Task 4 | Drone Response - Recommend deployment strategies |
| Task 5 | Reflection - Discuss limitations and improvements |

## 📚 Learning Outcomes

After completing this project, you will understand:

- How aerial imagery features relate to fire/smoke detection
- Supervised classification for disaster monitoring
- Evaluation metrics (precision, recall, F1, ROC-AUC)
- Spatial data visualization and risk mapping
- Practical drone deployment for emergency response

## 🛠️ Dependencies

- Python 3.11+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

*This is a teaching-oriented project designed for complete beginners in Machine Learning and Data Science.*
