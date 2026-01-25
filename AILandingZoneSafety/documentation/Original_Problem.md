# AI-Based Landing Zone Safety Capstone Project

## AI-Based Landing Zone Safety Classification Using Drone Imagery

### Dataset
https://docs.google.com/spreadsheets/d/1tCQf9YVzj8zET1bjTlettAV5WfyeNpo4EBEjo5H1Z9Y/edit?usp=sharing

---

## Capstone Overview

In this capstone, you will design an end-to-end AI pipeline to classify drone landing zones as safe or unsafe using aerial imagery-derived features. The project focuses on feature-level terrain analysis (not raw image processing), machine learning classification, and spatial safety assessment to support autonomous and remote drone landing operations.

---

## Dataset Provided

You are provided with a dataset representing tile-level features extracted from aerial imagery and elevation data. Each row corresponds to a spatial landing zone tile evaluated for landing safety.

Features include slope, surface roughness, vegetation indicators, obstacle density, shadow coverage, brightness variation, and detection confidence scores. Labels indicate whether a tile is considered safe or unsafe for landing.

**Important:**
You are not expected to process raw images. The dataset simulates realistic outputs after perception, segmentation, and feature extraction pipelines used in drone autonomy.

---

## Objectives

1. Understand terrain and visual indicators affecting drone landing safety
2. Apply machine learning for safety classification problems
3. Evaluate model reliability using appropriate performance metrics
4. Perform spatial aggregation for landing zone risk mapping
5. Interpret AI outputs for autonomous drone decision-making

---

## Capstone Tasks

### Task 1: Data Understanding
Explore the dataset and explain the physical meaning of each feature and its relevance to landing safety assessment.

### Task 2: Machine Learning Model
Train a classification model to predict landing zone safety. Evaluate performance using precision, recall, F1-score, confusion matrix, and ROC-AUC. Justify why accuracy alone is insufficient for safety-critical systems.

### Task 3: Spatial Safety Analysis & Visualization
Aggregate predictions across spatial grid cells and generate a landing safety heatmap indicating preferred and restricted landing zones.

### Task 4: Drone Autonomy Interpretation
Recommend landing strategies and fallback behaviors based on spatial safety patterns and confidence levels.

### Task 5: Reflection
Discuss dataset limitations and propose improvements using real-time perception, multi-view imagery, or onboard sensing.

---

## Deliverables

- Code and outputs
- Landing zone safety heatmaps
- Short written technical interpretation
