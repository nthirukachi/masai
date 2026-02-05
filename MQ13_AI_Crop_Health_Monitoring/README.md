# AI-Based Crop Health Monitoring Using Drone Multispectral Data

## Overview

This project builds an end-to-end AI pipeline to detect crop stress using vegetation indices from drone imagery. It trains multiple ML classification models, compares their performance, generates spatial stress maps, and provides drone inspection recommendations.

## Quick Start

```powershell
cd c:\masai\MQ13_AI_Crop_Health_Monitoring
uv run src/MQ13_AI_Crop_Health_Monitoring.py
```

## Project Structure

```
MQ13_AI_Crop_Health_Monitoring/
├── notebook/           # Teaching Jupyter Notebook
├── src/                # Python source code
├── documentation/      # All teaching documentation
├── slides/             # Presentation slides
├── outputs/            # Generated visualizations
└── README.md           # This file
```

## Dataset

- **Source:** Drone multispectral imagery (pre-processed)
- **Features:** 15 vegetation indices + grid coordinates
- **Target:** Binary classification (Healthy/Stressed)

## Models Compared

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

## Key Deliverables

- ✅ Model comparison with metrics
- ✅ Field-level stress heatmap
- ✅ Drone inspection recommendations
- ✅ Comprehensive documentation
