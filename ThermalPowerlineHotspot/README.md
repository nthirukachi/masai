# AI-Based Thermal Powerline Hotspot Detection

## 🎯 Project Overview
This capstone project implements an end-to-end AI pipeline to detect thermal hotspots in power lines and transmission towers using drone-based thermal inspection data. The project focuses on feature-level thermal analysis, machine learning classification, and spatial risk visualization for predictive maintenance.

## 📁 Project Structure
```
ThermalPowerlineHotspot/
├── notebook/                    # Jupyter notebooks for teaching
│   └── thermal_powerline_hotspot.ipynb
├── src/                         # Python source code
│   └── thermal_powerline_hotspot.py
├── documentation/               # All documentation files
│   ├── Original_Problem.md
│   ├── problem_statement.md
│   ├── concepts_explained.md
│   ├── observations_and_conclusion.md
│   ├── interview_questions.md
│   ├── exam_preparation.md
│   └── interview_preparation.md
├── slides/                      # Presentation slides
│   └── slides.md
├── outputs/                     # Generated outputs (heatmaps, plots)
└── README.md
```

## 🔥 Capstone Tasks
| Task | Description |
|------|-------------|
| **Task 1** | Data Understanding - Explore thermal features |
| **Task 2** | ML Model - Classification with evaluation metrics |
| **Task 3** | Spatial Risk Analysis - Thermal heatmaps |
| **Task 4** | Drone Interpretation - Maintenance recommendations |
| **Task 5** | Reflection - Limitations and improvements |

## 📊 Dataset Features
| Feature | Description |
|---------|-------------|
| `temp_mean` | Mean temperature in tile (°C) |
| `temp_max` | Maximum temperature in tile (°C) |
| `temp_std` | Temperature standard deviation |
| `delta_to_neighbors` | Temperature difference from adjacent tiles |
| `hotspot_fraction` | Fraction of pixels above threshold |
| `edge_gradient` | Temperature gradient at edges |
| `ambient_temp` | Ambient environmental temperature (°C) |
| `load_factor` | Electrical load factor (0-1) |
| `fault_label` | Target: 0=Normal, 1=Anomaly |

## 🚀 How to Run

### Using UV Virtual Environment
```powershell
cd c:\masai\ThermalPowerlineHotspot
uv run python src/thermal_powerline_hotspot.py
```

## 📚 Learning Objectives
1. Understand thermal indicators for power infrastructure inspection
2. Apply machine learning for thermal anomaly detection
3. Evaluate model reliability using appropriate metrics
4. Perform spatial aggregation for corridor-level risk mapping
5. Interpret AI outputs for drone-based maintenance planning

## 🛠️ Technologies Used
- Python 3.x
- Pandas, NumPy
- Scikit-learn (Random Forest)
- Matplotlib, Seaborn
- UV Virtual Environment

---
*Created as a teaching-oriented capstone project*
