# Original Problem Statement

## AI Based Thermal Powerline Hotspot Capstone Project
**AI-Based Power Line & Tower Hotspot Detection Using Thermal Data**

### Dataset
https://docs.google.com/spreadsheets/d/1E9QD-FNidcYT8-Ae5km2aQidDvlspmpp0fUcYIOkvTg/edit?usp=sharing

---

## Capstone Overview
In this capstone, you will design an end-to-end AI pipeline to detect thermal hotspots in power lines and transmission towers using drone-based thermal inspection data. The project focuses on feature-level thermal analysis (not raw image processing), machine learning classification, and spatial risk visualization for predictive maintenance.

---

## Dataset Provided
You are provided with a dataset that represents tile-level thermal features extracted from drone thermal imagery. Each row corresponds to a spatial tile along a power corridor or substation component.

Features include temperature statistics, hotspot density, thermal gradients, ambient conditions, and operational load indicators. Labels indicate whether a tile represents a potential thermal anomaly.

**Important:**
You are not expected to process raw thermal images. The dataset simulates real-world outputs after thermal tiling and feature extraction.

---

## Objectives
- Understand thermal indicators used in power infrastructure inspection
- Apply machine learning to detect thermal anomalies
- Evaluate model reliability using appropriate metrics
- Perform spatial aggregation for corridor-level risk mapping
- Interpret AI outputs for drone-based maintenance planning

---

## Capstone Tasks

### Task 1: Data Understanding
Explore the dataset and explain the physical meaning of each thermal feature and its relevance to hotspot detection.

### Task 2: Machine Learning Model
Train a classification model to predict thermal anomalies. Evaluate performance using precision, recall, F1-score, confusion matrix, and ROC-AUC. Justify why accuracy alone is insufficient.

### Task 3: Spatial Risk Analysis & Visualization
Aggregate predictions across spatial grid cells and generate a thermal risk heatmap representing inspection priority zones.

### Task 4: Power System & Drone Interpretation
Recommend drone inspection and maintenance actions based on hotspot severity and spatial clustering.

### Task 5: Reflection
Discuss dataset limitations and propose improvements using real thermal imagery or temporal monitoring.

---

## Deliverables
- Code and outputs
- Thermal hotspot risk heatmaps
- Short written technical interpretation
