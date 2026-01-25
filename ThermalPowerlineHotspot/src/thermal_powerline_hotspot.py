"""
================================================================================
AI-Based Thermal Powerline Hotspot Detection
================================================================================

🧩 PROBLEM STATEMENT:
    Detect thermal hotspots in power lines and transmission towers using 
    drone-based thermal inspection data. Build ML model for anomaly detection
    and create spatial risk heatmaps for predictive maintenance.

🎯 EXPECTED OUTPUT:
    1. Classification metrics (Precision, Recall, F1-Score, ROC-AUC)
    2. Confusion Matrix visualization
    3. Thermal risk heatmap for inspection priority

🪜 STEPS:
    Task 1: Data Understanding
    Task 2: Machine Learning Model
    Task 3: Spatial Risk Analysis & Visualization
    Task 4: Drone Interpretation & Recommendations
    Task 5: Reflection & Limitations

Author: Teaching Project
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORT LIBRARIES
# ==============================================================================
# 2.1 WHAT: Import statements bring external code packages into our program
# 2.2 WHY: We don't need to write everything from scratch - use pre-built tools
#      Alternative: Write all math functions ourselves (very slow, error-prone)
# 2.3 WHEN: Always at the TOP of your Python file, before any other code
# 2.4 WHERE: Every Python program that uses external libraries
# 2.5 HOW: import library_name or from library import specific_thing
# 2.6 INTERNALLY: Python searches sys.path folders for the library files
# 2.7 OUTPUT: No visible output - just loads the library into memory

import pandas as pd          # For handling tabular data (like Excel)
import numpy as np           # For numerical operations (math on arrays)
import matplotlib.pyplot as plt   # For creating visualizations (charts, plots)
import seaborn as sns        # For beautiful statistical visualizations
from sklearn.model_selection import train_test_split  # Split data for training/testing
from sklearn.ensemble import RandomForestClassifier   # Our ML classification model
from sklearn.metrics import (
    classification_report,   # Summary of precision, recall, f1
    confusion_matrix,        # Shows true vs predicted labels
    roc_auc_score,          # Area under ROC curve
    roc_curve,              # Points for ROC curve plot
    accuracy_score          # Simple accuracy percentage
)
import warnings
warnings.filterwarnings('ignore')  # Hide warning messages for cleaner output

# ==============================================================================
# SECTION 2: DATASET CREATION (SIMULATING THE PROVIDED DATA)
# ==============================================================================
# 2.1 WHAT: Create a synthetic dataset that matches the Google Sheets structure
# 2.2 WHY: Ensures reproducibility without requiring internet access
# 2.3 WHEN: When developing/teaching, before actual deployment
# 2.4 WHERE: Initial data loading phase of any ML pipeline
# 2.5 HOW: Use numpy random functions to generate realistic values
# 2.6 INTERNALLY: Random number generator creates values within specified ranges
# 2.7 OUTPUT: A pandas DataFrame with 1000 rows and 9 columns

def create_thermal_dataset(n_samples=1000, random_state=42):
    """
    Create a synthetic thermal powerline inspection dataset.
    
    ⚙️ FUNCTION ARGUMENTS EXPLANATION:
    
    3.1 n_samples (int, default=1000):
        WHAT: Number of rows (spatial tiles) to generate
        WHY: Controls dataset size for training
        WHEN: Increase for more training data, decrease for quick testing
        WHERE: Any dataset generation function
        HOW: create_thermal_dataset(n_samples=500)
        INTERNALLY: Loops n_samples times to create each row
        OUTPUT: DataFrame with exactly n_samples rows
        
    3.2 random_state (int, default=42):
        WHAT: Seed for random number generator
        WHY: Makes results reproducible (same random numbers each time)
        WHEN: Always set during development/teaching
        WHERE: Any function with randomness
        HOW: create_thermal_dataset(random_state=123)
        INTERNALLY: Initializes the random generator to a known state
        OUTPUT: Same dataset every time with same seed
    
    Returns:
        pd.DataFrame: Dataset with thermal features and fault labels
    """
    # Set random seed for reproducibility
    # 2.1 WHAT: Initialize random number generator
    # 2.2 WHY: Ensures we get the same "random" numbers each run
    np.random.seed(random_state)
    
    # Generate base features with realistic ranges
    # 2.1 WHAT: Create arrays of random numbers within realistic temperature ranges
    # 2.2 WHY: Simulate real drone thermal camera readings
    # 2.6 INTERNALLY: uniform() generates numbers between low and high
    
    # temp_mean: Average temperature of tile (15-65°C range)
    # Normal tiles: 15-45°C, Hotspot tiles: 40-65°C
    temp_mean_base = np.random.uniform(15, 45, n_samples)
    
    # Introduce some anomalies (10% of data will be potential hotspots)
    anomaly_mask = np.random.random(n_samples) < 0.30  # 30% anomaly rate
    temp_mean = temp_mean_base.copy()
    temp_mean[anomaly_mask] = np.random.uniform(40, 65, anomaly_mask.sum())
    
    # temp_max: Maximum temperature (always >= temp_mean)
    temp_max = temp_mean + np.random.uniform(0, 15, n_samples)
    
    # temp_std: Temperature standard deviation (2-7 range)
    temp_std = np.random.uniform(2, 7, n_samples)
    
    # delta_to_neighbors: Difference from adjacent tiles (-12 to +18 range)
    # Higher positive values indicate localized heating
    delta_to_neighbors = np.random.uniform(-12, 18, n_samples)
    # Anomalies tend to have higher deltas
    delta_to_neighbors[anomaly_mask] += np.random.uniform(2, 8, anomaly_mask.sum())
    
    # hotspot_fraction: Fraction of tile with high temperature (0-1)
    hotspot_fraction = np.random.uniform(0, 0.8, n_samples)
    hotspot_fraction[anomaly_mask] = np.random.uniform(0.4, 0.9, anomaly_mask.sum())
    
    # edge_gradient: Temperature gradient at tile edges (0.2-1.8)
    edge_gradient = np.random.uniform(0.2, 1.8, n_samples)
    
    # ambient_temp: Environmental temperature (15-45°C)
    ambient_temp = np.random.uniform(15, 45, n_samples)
    
    # load_factor: Electrical load (0.3-1.0)
    load_factor = np.random.uniform(0.3, 1.0, n_samples)
    # Anomalies often occur at high load
    load_factor[anomaly_mask] = np.random.uniform(0.6, 1.0, anomaly_mask.sum())
    
    # Create fault label based on combination of features
    # This simulates real labeling by thermal experts
    fault_label = np.zeros(n_samples, dtype=int)
    
    # Rule-based labeling (simulating expert annotation)
    # Condition 1: High temperature + high hotspot fraction
    condition1 = (temp_mean > 45) & (hotspot_fraction > 0.5)
    # Condition 2: High delta to neighbors + high load
    condition2 = (delta_to_neighbors > 5) & (load_factor > 0.8)
    # Condition 3: Extreme max temperature
    condition3 = temp_max > 60
    
    fault_label[condition1 | condition2 | condition3] = 1
    
    # Create DataFrame
    # 2.1 WHAT: Organize all arrays into a table structure
    # 2.2 WHY: Makes data manipulation easy with pandas
    data = pd.DataFrame({
        'temp_mean': np.round(temp_mean, 2),
        'temp_max': np.round(temp_max, 2),
        'temp_std': np.round(temp_std, 2),
        'delta_to_neighbors': np.round(delta_to_neighbors, 2),
        'hotspot_fraction': np.round(hotspot_fraction, 2),
        'edge_gradient': np.round(edge_gradient, 2),
        'ambient_temp': np.round(ambient_temp, 2),
        'load_factor': np.round(load_factor, 2),
        'fault_label': fault_label
    })
    
    return data


# ==============================================================================
# TASK 1: DATA UNDERSTANDING
# ==============================================================================
def task1_data_understanding(df):
    """
    Explore the dataset and explain physical meaning of each thermal feature.
    
    ⚙️ FUNCTION ARGUMENTS EXPLANATION:
    
    3.1 df (pd.DataFrame):
        WHAT: The input dataset containing thermal features
        WHY: We need data to analyze and understand
        WHEN: At the start of any ML project
        WHERE: First step in the ML pipeline
        HOW: task1_data_understanding(thermal_data)
        INTERNALLY: Uses pandas methods to compute statistics
        OUTPUT: Prints statistics and analysis to console
    
    Returns:
        None: Prints results to console
    """
    print("=" * 80)
    print("TASK 1: DATA UNDERSTANDING")
    print("=" * 80)
    
    # 2.1 WHAT: Display first 5 rows of the dataset
    # 2.2 WHY: Get a quick peek at what the data looks like
    # 2.5 HOW: Use .head() method on DataFrame
    # 2.7 OUTPUT: Table showing first 5 rows
    print("\n📊 1.1 First 5 Rows of Dataset:")
    print("-" * 40)
    print(df.head())
    
    # 2.1 WHAT: Get basic info about the dataset
    # 2.2 WHY: Understand data types, missing values, memory usage
    # 2.7 OUTPUT: Summary of column types and memory
    print("\n📋 1.2 Dataset Information:")
    print("-" * 40)
    print(f"Number of samples (tiles): {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")
    print(f"Target column: fault_label")
    print(f"\nData Types:")
    print(df.dtypes)
    
    # 2.1 WHAT: Check for missing values
    # 2.2 WHY: Missing data can cause model errors
    # 2.7 OUTPUT: Count of missing values per column
    print("\n🔍 1.3 Missing Values Check:")
    print("-" * 40)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found - data is complete!")
    else:
        print(missing)
    
    # 2.1 WHAT: Statistical summary of numerical columns
    # 2.2 WHY: Understand range, mean, std of each feature
    # 2.7 OUTPUT: Table with min, max, mean, std, quartiles
    print("\n📈 1.4 Statistical Summary:")
    print("-" * 40)
    print(df.describe().round(2))
    
    # 2.1 WHAT: Class distribution (how many normal vs anomalies)
    # 2.2 WHY: Check for class imbalance - affects model evaluation
    # 2.7 OUTPUT: Count and percentage of each class
    print("\n⚖️ 1.5 Class Distribution (Target Variable):")
    print("-" * 40)
    class_counts = df['fault_label'].value_counts()
    class_pct = df['fault_label'].value_counts(normalize=True) * 100
    print(f"Normal (0):  {class_counts[0]:4d} samples ({class_pct[0]:.1f}%)")
    print(f"Anomaly (1): {class_counts[1]:4d} samples ({class_pct[1]:.1f}%)")
    
    # Feature explanation table
    print("\n📖 1.6 Feature Explanation:")
    print("-" * 40)
    feature_explanations = {
        'temp_mean': 'Average temperature in the tile (°C). Higher = more heat.',
        'temp_max': 'Maximum temperature in the tile (°C). Peak heat point.',
        'temp_std': 'Temperature variation. High = uneven heating.',
        'delta_to_neighbors': 'Difference from nearby tiles. High = localized hotspot.',
        'hotspot_fraction': 'Fraction of tile above threshold. High = more area is hot.',
        'edge_gradient': 'Temperature change rate at edges. Sudden changes = concern.',
        'ambient_temp': 'Environmental temperature (°C). Context for readings.',
        'load_factor': 'Electrical load (0-1). High load = more expected heat.',
        'fault_label': 'Target: 0=Normal, 1=Thermal Anomaly'
    }
    for feature, explanation in feature_explanations.items():
        print(f"• {feature:20s}: {explanation}")
    
    # Correlation analysis
    print("\n🔗 1.7 Correlation with Fault Label:")
    print("-" * 40)
    correlations = df.corr()['fault_label'].drop('fault_label').sort_values(ascending=False)
    print("Features most correlated with thermal anomalies:")
    for feature, corr in correlations.items():
        indicator = "🔥" if corr > 0.3 else "📊" if corr > 0.1 else "〰️"
        print(f"{indicator} {feature:20s}: {corr:+.3f}")


# ==============================================================================
# TASK 2: MACHINE LEARNING MODEL
# ==============================================================================
def task2_ml_model(df, output_dir='outputs'):
    """
    Train a classification model to predict thermal anomalies.
    
    ⚙️ FUNCTION ARGUMENTS EXPLANATION:
    
    3.1 df (pd.DataFrame):
        WHAT: Dataset with features and target label
        WHY: ML model needs data to learn patterns
        WHEN: After data understanding and cleaning
        WHERE: Main training phase of ML pipeline
        HOW: task2_ml_model(thermal_data)
        INTERNALLY: Extracts features (X) and target (y), splits, trains
        OUTPUT: Returns trained model and predictions
    
    3.2 output_dir (str, default='outputs'):
        WHAT: Folder path to save output plots
        WHY: Organize generated visualizations
        WHEN: Always specify to keep outputs organized
        WHERE: Any function that generates plots
        HOW: task2_ml_model(df, output_dir='my_outputs')
        INTERNALLY: Concatenates with filenames for full path
        OUTPUT: Plots saved to specified directory
    
    Returns:
        tuple: (trained_model, X_test, y_test, y_pred, y_proba)
    """
    print("\n" + "=" * 80)
    print("TASK 2: MACHINE LEARNING MODEL")
    print("=" * 80)
    
    # 2.1 WHAT: Separate features (X) from target (y)
    # 2.2 WHY: ML models need input features separate from output labels
    # 2.5 HOW: Use column selection with drop() method
    # 2.6 INTERNALLY: Creates new DataFrames referencing original data
    # 2.7 OUTPUT: X has 8 columns, y has 1 column
    print("\n🔧 2.1 Preparing Features and Target:")
    print("-" * 40)
    
    X = df.drop('fault_label', axis=1)  # All columns except target
    y = df['fault_label']                # Only the target column
    
    print(f"Feature matrix X shape: {X.shape} (samples × features)")
    print(f"Target vector y shape: {y.shape} (samples,)")
    print(f"Features used: {list(X.columns)}")
    
    # 2.1 WHAT: Split data into training and testing sets
    # 2.2 WHY: Test set is "unseen" data to evaluate real performance
    #      Alternative: Use all data for training (leads to overfitting!)
    # 2.5 HOW: train_test_split(X, y, test_size=0.2) - 80% train, 20% test
    # 2.6 INTERNALLY: Randomly shuffles and splits the data
    # 2.7 OUTPUT: 4 arrays - X_train, X_test, y_train, y_test
    print("\n📊 2.2 Splitting Data (80% Train, 20% Test):")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 20% for testing
        random_state=42,     # Reproducibility
        stratify=y           # Maintain class balance in both sets
    )
    """
    ARGUMENT-BY-ARGUMENT EXPLANATION for train_test_split:
    
    3.1 X: Feature matrix to split
    3.2 y: Target vector to split
    3.3 test_size=0.2:
        WHAT: Fraction of data for testing
        WHY: 20% is standard - enough to evaluate, leaves enough for training
        WHEN: Almost always use 0.2 or 0.3
        ALTERNATIVE: 0.3 for smaller datasets
    3.4 random_state=42:
        WHAT: Random seed for shuffling
        WHY: Same split every time for reproducibility
    3.5 stratify=y:
        WHAT: Ensure same class ratio in train and test
        WHY: With imbalanced data, random split might put all anomalies in one set
        WHEN: Always use for classification problems
    """
    
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Testing set:  {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")
    print(f"\nClass distribution in training set:")
    print(f"  Normal:  {(y_train == 0).sum()}")
    print(f"  Anomaly: {(y_train == 1).sum()}")
    
    # 2.1 WHAT: Create and configure the Random Forest model
    # 2.2 WHY: Random Forest is robust, handles imbalanced data well
    #      Alternative: Logistic Regression (simpler but less accurate for complex patterns)
    #      Alternative: XGBoost (more powerful but harder to tune)
    # 2.5 HOW: Create instance with hyperparameters
    # 2.6 INTERNALLY: Prepares to build multiple decision trees
    print("\n🤖 2.3 Creating Random Forest Classifier:")
    print("-" * 40)
    
    model = RandomForestClassifier(
        n_estimators=100,       # Number of trees in the forest
        max_depth=10,           # Maximum depth of each tree
        min_samples_split=5,    # Minimum samples to split a node
        min_samples_leaf=2,     # Minimum samples in leaf node
        class_weight='balanced', # Handle class imbalance
        random_state=42,        # Reproducibility
        n_jobs=-1               # Use all CPU cores
    )
    """
    ARGUMENT-BY-ARGUMENT EXPLANATION for RandomForestClassifier:
    
    3.1 n_estimators=100:
        WHAT: Number of decision trees to build
        WHY: More trees = more stable predictions, but slower training
        WHEN: Start with 100, increase if underfitting
        DEFAULT: 100 is good for most problems
        
    3.2 max_depth=10:
        WHAT: Maximum levels in each tree
        WHY: Limits tree complexity, prevents overfitting
        WHEN: Reduce if overfitting (high train, low test accuracy)
        DEFAULT: None (unlimited) - often leads to overfitting
        
    3.3 min_samples_split=5:
        WHAT: Minimum samples needed to split a node
        WHY: Prevents creating branches for tiny groups
        WHEN: Increase for noisy data
        DEFAULT: 2
        
    3.4 min_samples_leaf=2:
        WHAT: Minimum samples in final nodes (leaves)
        WHY: Each prediction needs enough samples to be reliable
        WHEN: Increase for more conservative predictions
        DEFAULT: 1
        
    3.5 class_weight='balanced':
        WHAT: Automatically adjust weights based on class frequency
        WHY: Gives more importance to minority class (anomalies)
        WHEN: Always use for imbalanced classification
        ALTERNATIVE: None (treats all classes equally)
        
    3.6 n_jobs=-1:
        WHAT: Number of CPU cores to use
        WHY: -1 means use all available cores for faster training
        WHEN: Always for large datasets
        DEFAULT: 1 (single core)
    """
    
    print("Model Configuration:")
    print(f"  • Number of trees:    {model.n_estimators}")
    print(f"  • Max tree depth:     {model.max_depth}")
    print(f"  • Min samples split:  {model.min_samples_split}")
    print(f"  • Min samples leaf:   {model.min_samples_leaf}")
    print(f"  • Class weight:       {model.class_weight}")
    
    # 2.1 WHAT: Train the model on training data
    # 2.2 WHY: Model learns patterns from labeled examples
    # 2.5 HOW: model.fit(X_train, y_train)
    # 2.6 INTERNALLY: Builds 100 decision trees, each on random subset
    # 2.7 OUTPUT: Model weights are updated (no visible output)
    print("\n🎓 2.4 Training the Model:")
    print("-" * 40)
    model.fit(X_train, y_train)
    print("✅ Model training complete!")
    
    # 2.1 WHAT: Make predictions on test data
    # 2.2 WHY: Evaluate how well model generalizes to unseen data
    # 2.5 HOW: model.predict(X_test) for class labels
    #          model.predict_proba(X_test) for probability scores
    # 2.6 INTERNALLY: Each tree votes, majority wins for predict()
    # 2.7 OUTPUT: Arrays of predictions
    print("\n🎯 2.5 Making Predictions on Test Set:")
    print("-" * 40)
    
    y_pred = model.predict(X_test)           # Predicted class labels (0 or 1)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    print(f"Predictions made for {len(y_pred)} test samples")
    print(f"Predicted Normal:  {(y_pred == 0).sum()}")
    print(f"Predicted Anomaly: {(y_pred == 1).sum()}")
    
    # 2.1 WHAT: Calculate and display all evaluation metrics
    # 2.2 WHY: Different metrics reveal different aspects of performance
    print("\n📈 2.6 Model Evaluation Metrics:")
    print("-" * 40)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n1. ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("   → What percentage of predictions are correct")
    print("   ⚠️ BUT: Can be misleading with imbalanced data!")
    
    # Why accuracy is not enough - explanation
    print("\n" + "=" * 60)
    print("❓ WHY ACCURACY ALONE IS INSUFFICIENT:")
    print("=" * 60)
    print("""
    Imagine 100 power line tiles where:
    - 90 are NORMAL ✅
    - 10 are FAULTY 🔥 (dangerous hotspots!)
    
    A model that predicts EVERYTHING as normal gets:
    - Accuracy = 90% (looks great!)
    - But it MISSES ALL dangerous hotspots!
    
    In power line inspection:
    - Missing a real hotspot (False Negative) → FIRE or BLACKOUT 🔥
    - False alarm (False Positive) → Extra inspection (minor cost)
    
    THEREFORE: We need metrics that penalize missing true anomalies!
    """)
    
    # Precision, Recall, F1-Score
    print("\n2. CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal (0)', 'Anomaly (1)']))
    
    print("   PRECISION: Of all predicted anomalies, how many are real?")
    print("   RECALL:    Of all real anomalies, how many did we find?")
    print("   F1-SCORE:  Balance between precision and recall")
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n3. ROC-AUC SCORE: {roc_auc:.4f}")
    print("   → Measures how well model separates classes (0.5=random, 1.0=perfect)")
    
    # Confusion Matrix
    print("\n4. CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"""
                    PREDICTED
                 Normal  Anomaly
    ACTUAL Normal   {cm[0,0]:4d}    {cm[0,1]:4d}   ← {cm[0,0]} correct, {cm[0,1]} false alarms
           Anomaly  {cm[1,0]:4d}    {cm[1,1]:4d}   ← {cm[1,0]} MISSED, {cm[1,1]} correctly caught
    
    ⚠️ {cm[1,0]} anomalies were MISSED (False Negatives) - these could cause failures!
    """)
    
    # Feature Importance
    print("\n📊 2.7 Feature Importance:")
    print("-" * 40)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Which features are most useful for detecting hotspots?")
    for _, row in feature_importance.iterrows():
        bar = '█' * int(row['Importance'] * 50)
        print(f"  {row['Feature']:20s}: {row['Importance']:.3f} {bar}")
    
    # Save plots
    print("\n🎨 2.8 Creating Visualizations:")
    print("-" * 40)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance
    colors = plt.cm.RdYlGn_r(feature_importance['Importance'] / feature_importance['Importance'].max())
    axes[2].barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
    axes[2].set_xlabel('Importance')
    axes[2].set_title('Feature Importance')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/model_evaluation.png")
    
    return model, X_test, y_test, y_pred, y_proba


# ==============================================================================
# TASK 3: SPATIAL RISK ANALYSIS & VISUALIZATION
# ==============================================================================
def task3_spatial_risk_analysis(df, model, output_dir='outputs'):
    """
    Aggregate predictions across spatial grid cells and generate risk heatmap.
    
    ⚙️ FUNCTION ARGUMENTS EXPLANATION:
    
    3.1 df (pd.DataFrame):
        WHAT: Original dataset with all features
        WHY: Need features to make predictions for all tiles
        WHEN: After model is trained
        
    3.2 model: Trained RandomForest model
        WHAT: The trained ML classifier
        WHY: Use it to predict risk probability for each tile
        
    3.3 output_dir (str): Folder to save heatmap image
    
    Returns:
        pd.DataFrame: Risk scores for each grid cell
    """
    print("\n" + "=" * 80)
    print("TASK 3: SPATIAL RISK ANALYSIS & VISUALIZATION")
    print("=" * 80)
    
    # 2.1 WHAT: Get risk probability for each tile
    # 2.2 WHY: Probability gives more info than just yes/no prediction
    # 2.5 HOW: model.predict_proba(X)[:, 1] gets probability of class 1
    # 2.7 OUTPUT: Array of probabilities (0.0 to 1.0)
    print("\n🔮 3.1 Calculating Risk Probabilities:")
    print("-" * 40)
    
    X = df.drop('fault_label', axis=1)
    risk_proba = model.predict_proba(X)[:, 1]
    
    df_with_risk = df.copy()
    df_with_risk['risk_score'] = risk_proba
    
    print(f"Calculated risk scores for {len(df)} tiles")
    print(f"Risk score range: {risk_proba.min():.3f} to {risk_proba.max():.3f}")
    print(f"Average risk: {risk_proba.mean():.3f}")
    
    # 2.1 WHAT: Create grid layout for heatmap visualization
    # 2.2 WHY: Organize tiles into a 2D grid representing physical layout
    # 2.5 HOW: Create n×n grid where n = sqrt(number of tiles)
    print("\n🗺️ 3.2 Creating Spatial Grid:")
    print("-" * 40)
    
    n_tiles = len(df)
    grid_size = int(np.ceil(np.sqrt(n_tiles)))  # E.g., 1000 tiles → 32×32 grid
    
    print(f"Tiles: {n_tiles} → Grid: {grid_size}×{grid_size} = {grid_size**2} cells")
    
    # Create 2D grid of risk scores
    risk_grid = np.zeros((grid_size, grid_size))
    for i, risk in enumerate(risk_proba):
        row = i // grid_size
        col = i % grid_size
        if row < grid_size and col < grid_size:
            risk_grid[row, col] = risk
    
    # 2.1 WHAT: Classify risk levels into categories
    # 2.2 WHY: Easier for operators to prioritize actions
    print("\n📊 3.3 Risk Level Classification:")
    print("-" * 40)
    
    def classify_risk(score):
        if score < 0.25:
            return 'Low'
        elif score < 0.50:
            return 'Medium'
        elif score < 0.75:
            return 'High'
        else:
            return 'Critical'
    
    df_with_risk['risk_level'] = df_with_risk['risk_score'].apply(classify_risk)
    
    risk_counts = df_with_risk['risk_level'].value_counts()
    print("Risk Level Distribution:")
    for level in ['Low', 'Medium', 'High', 'Critical']:
        if level in risk_counts:
            count = risk_counts[level]
            pct = count / n_tiles * 100
            emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Critical': '🔴'}[level]
            print(f"  {emoji} {level:8s}: {count:4d} tiles ({pct:5.1f}%)")
    
    # 2.1 WHAT: Create the thermal risk heatmap visualization
    # 2.2 WHY: Visual representation helps operators quickly identify problem areas
    print("\n🎨 3.4 Generating Thermal Risk Heatmap:")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    im = axes[0].imshow(risk_grid, cmap='RdYlGn_r', aspect='equal', vmin=0, vmax=1)
    axes[0].set_title('Thermal Risk Heatmap\n(Power Corridor Grid)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Grid Column (West → East)')
    axes[0].set_ylabel('Grid Row (North → South)')
    cbar = plt.colorbar(im, ax=axes[0], label='Risk Score')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low\n(0.0)', 'Medium\n(0.25)', 'Elevated\n(0.50)', 'High\n(0.75)', 'Critical\n(1.0)'])
    
    # Add grid lines
    axes[0].set_xticks(np.arange(-0.5, grid_size, 5), minor=True)
    axes[0].set_yticks(np.arange(-0.5, grid_size, 5), minor=True)
    axes[0].grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Risk distribution histogram
    axes[1].hist(risk_proba, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    axes[1].axvline(x=0.75, color='orange', linestyle='--', linewidth=2, label='Critical Threshold (0.75)')
    axes[1].set_xlabel('Risk Score')
    axes[1].set_ylabel('Number of Tiles')
    axes[1].set_title('Distribution of Risk Scores', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thermal_risk_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/thermal_risk_heatmap.png")
    
    # Identify high-risk zones
    print("\n🎯 3.5 High-Risk Zone Identification:")
    print("-" * 40)
    
    critical_zones = df_with_risk[df_with_risk['risk_level'] == 'Critical']
    high_zones = df_with_risk[df_with_risk['risk_level'] == 'High']
    
    print(f"Critical zones requiring immediate attention: {len(critical_zones)}")
    print(f"High-risk zones for scheduled maintenance: {len(high_zones)}")
    
    if len(critical_zones) > 0:
        print("\n⚠️ Top 5 Critical Tiles (Highest Risk):")
        top_critical = critical_zones.nlargest(5, 'risk_score')[['temp_mean', 'temp_max', 'hotspot_fraction', 'risk_score']]
        print(top_critical.to_string())
    
    return df_with_risk


# ==============================================================================
# TASK 4: POWER SYSTEM & DRONE INTERPRETATION
# ==============================================================================
def task4_drone_interpretation(df_with_risk):
    """
    Recommend drone inspection and maintenance actions based on hotspot severity.
    
    ⚙️ FUNCTION ARGUMENTS EXPLANATION:
    
    3.1 df_with_risk (pd.DataFrame):
        WHAT: DataFrame with risk scores and levels for each tile
        WHY: Need risk information to prioritize inspections
        WHEN: After spatial risk analysis is complete
        
    Returns:
        dict: Recommendations for each risk level
    """
    print("\n" + "=" * 80)
    print("TASK 4: POWER SYSTEM & DRONE INTERPRETATION")
    print("=" * 80)
    
    # 2.1 WHAT: Define maintenance recommendations based on risk level
    # 2.2 WHY: Different risk levels require different actions
    print("\n🚁 4.1 Drone Inspection Recommendations:")
    print("-" * 40)
    
    recommendations = {
        'Critical': {
            'priority': 'IMMEDIATE (24 hours)',
            'action': 'Deploy drone for detailed thermal inspection',
            'maintenance': 'Schedule emergency repair crew',
            'frequency': 'Daily monitoring until resolved',
            'icon': '🔴'
        },
        'High': {
            'priority': 'URGENT (72 hours)',
            'action': 'Schedule drone flyover for closer inspection',
            'maintenance': 'Plan preventive maintenance within 1 week',
            'frequency': 'Every 2 days',
            'icon': '🟠'
        },
        'Medium': {
            'priority': 'SCHEDULED (1 week)',
            'action': 'Include in regular drone patrol route',
            'maintenance': 'Add to monthly maintenance checklist',
            'frequency': 'Weekly',
            'icon': '🟡'
        },
        'Low': {
            'priority': 'ROUTINE (Monthly)',
            'action': 'Standard automated drone patrol',
            'maintenance': 'No immediate action required',
            'frequency': 'Monthly',
            'icon': '🟢'
        }
    }
    
    for level, rec in recommendations.items():
        print(f"\n{rec['icon']} {level.upper()} RISK:")
        print(f"   Priority: {rec['priority']}")
        print(f"   Action: {rec['action']}")
        print(f"   Maintenance: {rec['maintenance']}")
        print(f"   Monitoring: {rec['frequency']}")
    
    # 2.1 WHAT: Analyze spatial clustering of high-risk tiles
    # 2.2 WHY: Clustered hotspots may indicate systemic issues
    print("\n📍 4.2 Spatial Clustering Analysis:")
    print("-" * 40)
    
    critical_count = (df_with_risk['risk_level'] == 'Critical').sum()
    high_count = (df_with_risk['risk_level'] == 'High').sum()
    total_high_risk = critical_count + high_count
    
    print(f"Total high-risk areas: {total_high_risk} tiles")
    print(f"  - Critical: {critical_count}")
    print(f"  - High: {high_count}")
    
    if total_high_risk > 0:
        # Analyze feature patterns in high-risk areas
        high_risk_data = df_with_risk[df_with_risk['risk_level'].isin(['Critical', 'High'])]
        
        print("\n📊 High-Risk Area Characteristics:")
        print(f"   Avg Temperature: {high_risk_data['temp_mean'].mean():.1f}°C")
        print(f"   Avg Max Temp:    {high_risk_data['temp_max'].mean():.1f}°C")
        print(f"   Avg Load Factor: {high_risk_data['load_factor'].mean():.2f}")
        print(f"   Avg Hotspot %:   {high_risk_data['hotspot_fraction'].mean()*100:.1f}%")
    
    # 2.1 WHAT: Provide operational recommendations
    # 2.2 WHY: Help maintenance teams take appropriate action
    print("\n📋 4.3 Operational Recommendations:")
    print("-" * 40)
    print("""
    1. IMMEDIATE ACTIONS:
       • Deploy drones to Critical zones within 24 hours
       • Alert on-call maintenance crew for potential emergency
       • Prepare replacement parts inventory for common failure points
    
    2. SHORT-TERM (This Week):
       • Schedule preventive maintenance for High-risk zones
       • Review historical data for recurring hotspot patterns
       • Update drone flight paths to increase coverage of problem areas
    
    3. LONG-TERM PLANNING:
       • Analyze correlation between load patterns and hotspots
       • Consider infrastructure upgrades for chronic problem areas
       • Implement predictive maintenance scheduling based on AI model
    """)
    
    # 2.1 WHAT: Cost-benefit analysis
    # 2.2 WHY: Help management justify AI-based inspection investment
    print("\n💰 4.4 Cost-Benefit Insight:")
    print("-" * 40)
    print(f"""
    Without AI Detection:
    • Rely on scheduled inspections (may miss developing hotspots)
    • Higher risk of unexpected failures
    • Estimated outage cost: $50,000 - $500,000 per incident
    
    With AI Detection:
    • Early warning for {total_high_risk} potential problem areas
    • Prioritized inspection reduces unnecessary checks
    • Estimated prevention value: ${total_high_risk * 10000:,}+ in avoided damage
    """)
    
    return recommendations


# ==============================================================================
# TASK 5: REFLECTION & LIMITATIONS
# ==============================================================================
def task5_reflection():
    """
    Discuss dataset limitations and propose improvements.
    
    Returns:
        dict: Limitations and proposed improvements
    """
    print("\n" + "=" * 80)
    print("TASK 5: REFLECTION & LIMITATIONS")
    print("=" * 80)
    
    # 2.1 WHAT: Document dataset and methodology limitations
    # 2.2 WHY: Honest assessment helps improve future iterations
    print("\n⚠️ 5.1 Dataset Limitations:")
    print("-" * 40)
    
    limitations = [
        {
            'issue': 'Synthetic Data',
            'impact': 'Model trained on simulated features, not real thermal imagery',
            'consequence': 'May not capture real-world thermal patterns'
        },
        {
            'issue': 'No Temporal Information',
            'impact': 'Dataset is a snapshot, no time-series data',
            'consequence': 'Cannot detect developing hotspots over time'
        },
        {
            'issue': 'No Spatial Coordinates',
            'impact': 'Tiles lack real GPS coordinates',
            'consequence': 'Cannot map results to actual tower locations'
        },
        {
            'issue': 'Feature Abstraction',
            'impact': 'Using pre-extracted features, not raw thermal images',
            'consequence': 'Limited ability to discover new patterns'
        },
        {
            'issue': 'Weather Context',
            'impact': 'Limited ambient condition information',
            'consequence': 'Hard to account for seasonal/weather effects'
        }
    ]
    
    for i, lim in enumerate(limitations, 1):
        print(f"\n{i}. {lim['issue']}")
        print(f"   Impact: {lim['impact']}")
        print(f"   Consequence: {lim['consequence']}")
    
    # 2.1 WHAT: Propose improvements for real-world deployment
    # 2.2 WHY: Guide future development of the system
    print("\n💡 5.2 Proposed Improvements:")
    print("-" * 40)
    
    improvements = [
        {
            'suggestion': 'Use Real Thermal Images',
            'benefit': 'Deep learning (CNN) can extract richer features from raw imagery',
            'implementation': 'Collect labeled thermal images from actual drone flights'
        },
        {
            'suggestion': 'Add Temporal Monitoring',
            'benefit': 'Track hotspot evolution over time for predictive alerts',
            'implementation': 'Store historical data, use time-series analysis (LSTM)'
        },
        {
            'suggestion': 'Integrate GPS Coordinates',
            'benefit': 'Map predictions to exact tower/segment locations',
            'implementation': 'Tag each tile with lat/long from drone metadata'
        },
        {
            'suggestion': 'Multi-Modal Fusion',
            'benefit': 'Combine thermal with visible imagery for richer context',
            'implementation': 'Use multi-input deep learning models'
        },
        {
            'suggestion': 'Real-Time Edge Processing',
            'benefit': 'On-drone analysis for immediate alerts',
            'implementation': 'Deploy lightweight ML models on edge devices'
        },
        {
            'suggestion': 'Feedback Loop',
            'benefit': 'Continuous improvement from maintenance outcomes',
            'implementation': 'Track which predictions led to actual failures'
        }
    ]
    
    for i, imp in enumerate(improvements, 1):
        print(f"\n{i}. {imp['suggestion']}")
        print(f"   Benefit: {imp['benefit']}")
        print(f"   How: {imp['implementation']}")
    
    # 2.1 WHAT: Summary and conclusion
    # 2.2 WHY: Provide final takeaways
    print("\n📝 5.3 Conclusion:")
    print("-" * 40)
    print("""
    This capstone demonstrated an end-to-end AI pipeline for thermal hotspot
    detection in power infrastructure. Key achievements:
    
    ✅ Built a Random Forest classifier for anomaly detection
    ✅ Evaluated model using appropriate metrics (not just accuracy!)
    ✅ Created spatial risk heatmaps for prioritized inspection
    ✅ Provided actionable maintenance recommendations
    
    The methodology, while using synthetic data, establishes a framework that
    can be extended to real thermal imagery from drone inspections. The focus
    on explainable metrics and practical recommendations makes this approach
    suitable for integration into existing utility maintenance workflows.
    
    Next Steps:
    1. Collect and label real drone thermal data
    2. Integrate with GIS systems for spatial mapping
    3. Develop real-time monitoring dashboard
    4. Pilot test with utility company for validation
    """)
    
    return {'limitations': limitations, 'improvements': improvements}


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Main function to run all capstone tasks.
    
    ⚙️ FUNCTION ARGUMENTS EXPLANATION:
    This function takes no arguments. It orchestrates the entire pipeline.
    
    2.1 WHAT: Entry point that coordinates all tasks
    2.2 WHY: Provides clean, organized execution flow
    2.5 HOW: Called when script runs directly (not imported)
    """
    print("=" * 80)
    print("🔌 AI-BASED THERMAL POWERLINE HOTSPOT DETECTION")
    print("   Drone-Based Thermal Inspection for Predictive Maintenance")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Load/Create Dataset
    print("\n📥 Loading Thermal Dataset...")
    df = create_thermal_dataset(n_samples=1000, random_state=42)
    print(f"✅ Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
    
    # Step 2: Run all tasks
    task1_data_understanding(df)
    model, X_test, y_test, y_pred, y_proba = task2_ml_model(df)
    df_with_risk = task3_spatial_risk_analysis(df, model)
    task4_drone_interpretation(df_with_risk)
    task5_reflection()
    
    print("\n" + "=" * 80)
    print("✅ CAPSTONE PROJECT COMPLETE!")
    print("=" * 80)
    print("\nGenerated Outputs:")
    print("  📊 outputs/model_evaluation.png")
    print("  🗺️ outputs/thermal_risk_heatmap.png")
    print("\nThank you for learning about AI-based thermal hotspot detection!")


# 2.1 WHAT: Python idiom to run main() only when script is executed directly
# 2.2 WHY: Allows file to be imported without running main()
# 2.5 HOW: if __name__ == "__main__" checks if this is the main script
if __name__ == "__main__":
    main()
