"""
================================================================================
🚀 MQ12: CLASSIFICATION MODEL COMPARISON LEADERBOARD
================================================================================
A project to compare 5 major classification algorithms side-by-side.
Models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN
"""

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import pandas for data manipulation
# 2.2 WHY: To store our results in a nice table format
# 2.3 WHEN: Any time you need to handle tabular data
# 2.4 WHERE: Data processing and results storage
# 2.5 HOW: import pandas as pd
# 2.6 INTERNAL: Loads the library into memory
# 2.7 OUTPUT: Access to DataFrame objects
# ------------------------------------------------------------------------------
import pandas as pd

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import numpy for mathematical operations
# 2.2 WHY: Needed for numerical array handling
# 2.3 WHEN: Working with arrays or math functions
# 2.4 WHERE: Global use
# 2.5 HOW: import numpy as np
# 2.6 INTERNAL: High-performance array library
# 2.7 OUTPUT: Access to np tools
# ------------------------------------------------------------------------------
import numpy as np

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import matplotlib and seaborn for visualization
# 2.2 WHY: To create the "Leaderboard" chart
# 2.3 WHEN: After getting performance metrics
# 2.4 WHERE: Result reporting
# 2.5 HOW: import matplotlib.pyplot as plt
# 2.6 INTERNAL: Drawing engines for charts
# 2.7 OUTPUT: Visual plots
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import scikit-learn tools
# 2.2 WHY: The industry standard for ML models and evaluation
# 2.3 WHEN: Creating any ML pipeline
# 2.4 WHERE: Training and evaluation
# 2.5 HOW: from sklearn... import ...
# 2.6 INTERNAL: Implementation of algorithms
# 2.7 OUTPUT: Trained models and scores
# ------------------------------------------------------------------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import the 5 models we want to compare
# 2.2 WHY: Each has different strengths and weaknesses
# 2.3 WHEN: Comparing algorithms
# 2.4 WHERE: ML Model Arena
# 2.5 HOW: from sklearn.ensemble import RandomForestClassifier
# 2.6 INTERNAL: Logic for LR, DT, RF, SVM, KNN
# 2.7 OUTPUT: Model templates
# ------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import os
import warnings
warnings.filterwarnings('ignore')

# Create output folder
OUTPUT_PATH = "outputs/sample_outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*60)
print("🏆 THE ULTIMATE CLASSIFICATION MODEL FACE-OFF")
print("="*60)

# ==============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Load the Breast Cancer dataset
# 2.2 WHY: Standard classification dataset (Target: Malignant vs Benign)
# 2.3 WHEN: For learning and testing
# 2.4 WHERE: Sklearn datasets
# 2.5 HOW: load_breast_cancer()
# 2.6 INTERNAL: Returns a Bunch object (dictionary-like)
# 2.7 OUTPUT: Data and labels
#
# 3.1-3.7 ARGUMENT: return_X_y=True
# 3.1 WHAT: Returns data (X) and target (y) separately
# 3.2 WHY: Skips extra steps of extracting from the Bunch object
# 3.3 WHEN: Just need raw data for modeling
# 3.4 WHERE: Argument in loader function
# 3.5 HOW: return_X_y=True
# 3.6 INTERNAL: Faster extraction logic
# 3.7 OUTPUT: Tuple (X, y)
# ------------------------------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)
print(f"✅ Data Loaded: {X.shape[0]} samples, {X.shape[1]} features")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Split data into 80% Train and 20% Test
# 2.2 WHY: Fair evaluation on unseen data
# 2.3 WHEN: Before training
# 2.4 WHERE: Model pipeline
# 2.5 HOW: train_test_split(X, y)
# 2.6 INTERNAL: Random shuffling and splitting
# 2.7 OUTPUT: 4 sets of data
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Scale features to a standard range
# 2.2 WHY: Models like SVM and KNN are sensitive to size of numbers
# 2.3 WHEN: Essential for distance-based models
# 2.4 WHERE: Preprocessing
# 2.5 HOW: StandardScaler().fit_transform(X)
# 2.6 INTERNAL: Subtracts mean, divides by standard deviation
# 2.7 OUTPUT: Normalized data (Z-score)
# ------------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features Scaled (Mean=0, STD=1)")

# ==============================================================================
# SECTION 2: THE MODEL ARENA (TRAINING & EVALUATION)
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create a dictionary of models to be comparison
# 2.2 WHY: Allows us to loop through them automatically
# 2.3 WHEN: Comparing multiple settings/models
# 2.4 WHERE: Model definition
# 2.5 HOW: models = {"Name": ModelObject()}
# 2.6 INTERNAL: Storing objects to pointers
# 2.7 OUTPUT: Collection of models
# ------------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (linear)": SVC(kernel='linear', probability=True),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5)
}

# ------------------------------------------------------------------------------
# List to store results
# ------------------------------------------------------------------------------
results_list = []

print("\n🚀 Training Models and Computing Scores...")

for name, model in models.items():
    # --------------------------------------------------------------------------
    # 2.1 WHAT: Fit the model on training data
    # 2.2 WHY: Teacher teaches student
    # 2.3 WHEN: During training phase
    # 2.4 WHERE: ML loop
    # 2.5 HOW: model.fit(X, y)
    # 2.6 INTERNAL: Optimizes weights/splits for the algorithm
    # 2.7 OUTPUT: Trained model
    # --------------------------------------------------------------------------
    model.fit(X_train_scaled, y_train)
    
    # --------------------------------------------------------------------------
    # 2.1 WHAT: Make predictions on test data
    # 2.2 WHY: Exam time!
    # 2.3 WHEN: Evaluation phase
    # 2.4 WHERE: Testing
    # 2.5 HOW: model.predict(X_test)
    # 2.6 INTERNAL: Applies learned logic to new samples
    # 2.7 OUTPUT: Predicted labels (0 or 1)
    # --------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save to list
    results_list.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })
    
    print(f"   ✓ {name} completed (Acc: {accuracy:.2%})")

# ==============================================================================
# SECTION 3: RESULTS & VISUALIZATION
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create a DataFrame from the results list
# 2.2 WHY: Easy to sort, format and display as a leaderboard
# 2.3 WHEN: After finishing all experiments
# 2.4 WHERE: Final reporting
# 2.5 HOW: pd.DataFrame(data_list)
# 2.6 INTERNAL: Memory table creation
# 2.7 OUTPUT: Results Table
# ------------------------------------------------------------------------------
results_df = pd.DataFrame(results_list).sort_values(by="Accuracy", ascending=False)

print("\n" + "="*60)
print("🏆 MODEL LEADERBOARD (Sorted by Accuracy)")
print("="*60)
print(results_df.to_string(index=False))

# ------------------------------------------------------------------------------
# 2.1 WHAT: Save the leaderboard to a CSV file
# 2.2 WHY: Record keeping
# 2.3 WHEN: Success!
# 2.4 WHERE: Output folder
# 2.5 HOW: df.to_csv("path")
# 2.6 INTERNAL: Writes buffer to disk
# 2.7 OUTPUT: File on disk
# ------------------------------------------------------------------------------
results_df.to_csv(f"{OUTPUT_PATH}/model_comparison.csv", index=False)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create a Bar Chart of Model Accuracies
# 2.2 WHY: Visual representation of the champion
# 2.3 WHEN: Reporting phase
# 2.4 WHERE: Visual Summary
# 2.5 HOW: sns.barplot(...)
# 2.6 INTERNAL: Maps data to bar height
# 2.7 OUTPUT: Accuracy Plot
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plot = sns.barplot(x="Accuracy", y="Model", data=results_df, palette="magma")
plt.title("Classification Model Accuracy Comparison", fontsize=15)
plt.xlim(0.8, 1.0)  # Zoom in for clarity
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels
for p in plot.patches:
    width = p.get_width()
    plt.text(width + 0.005, p.get_y() + p.get_height()/2, f'{width:.2%}', va='center')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/accuracy_leaderboard.png", dpi=150)
print(f"\n✅ Leaderboard chart saved to: {OUTPUT_PATH}/accuracy_leaderboard.png")

# Final Observation
top_model = results_df.iloc[0]["Model"]
top_acc = results_df.iloc[0]["Accuracy"]
print(f"\n🎉 THE CHAMPION: {top_model} with {top_acc:.2%} Accuracy!")
print("="*60)
