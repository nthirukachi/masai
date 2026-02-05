"""
AI-Based Crop Health Monitoring Using Drone Multispectral Data
===============================================================

This script builds an end-to-end AI pipeline to detect crop stress using 
vegetation indices from drone multispectral imagery.

WHAT IT DOES:
- Loads drone-captured vegetation data
- Trains and compares 5 ML classification models
- Generates a spatial stress heatmap
- Provides drone inspection recommendations

WHY DO WE NEED THIS?
- Farmers can't manually check every plant in a huge farm
- Drones can fly and take special photos of entire fields
- AI can analyze these photos and find stressed crops
- Early detection saves time, money, and increases yield

REAL-LIFE ANALOGY:
Think of it like a doctor's check-up for plants:
- Doctor uses a thermometer → Drone uses special cameras
- Doctor checks blood pressure → Drone checks plant color/moisture
- Doctor says "take medicine" → AI says "water this area"
"""

# =============================================================================
# FIX FOR WINDOWS UNICODE ENCODING
# =============================================================================
import sys
import io
# Set stdout to UTF-8 encoding for emoji support on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# SECTION 1: IMPORTING LIBRARIES
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 What: Import pandas for data handling
# 1.2 Why: Pandas reads CSV files and handles data like Excel spreadsheets
#          It's the BEST tool for tabular data because:
#          - Easy to read/write files
#          - Fast data manipulation
#          - Works well with other ML libraries
# 1.3 When: Whenever you work with tables/spreadsheets
# 1.4 Where: Data science, ML, analytics, finance, research
# 1.5 How: import pandas as pd, then use pd.read_csv(), pd.DataFrame()
# 1.6 Internal: Pandas uses NumPy arrays under the hood for fast computation
# 1.7 Output: Creates a DataFrame object (like a smart table)
# -----------------------------------------------------------------------------
import pandas as pd

# -----------------------------------------------------------------------------
# 1.1 What: Import numpy for numerical operations
# 1.2 Why: NumPy handles numbers and arrays MUCH faster than Python lists
#          It's the FOUNDATION of scientific computing in Python
# 1.3 When: Math operations, array manipulation, ML preprocessing
# 1.4 Where: Every ML project, image processing, scientific computing
# 1.5 How: import numpy as np, then use np.array(), np.mean(), etc.
# 1.6 Internal: Written in C for speed, uses contiguous memory blocks
# 1.7 Output: NumPy arrays that support vectorized operations
# -----------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------
# 1.1 What: Import matplotlib for creating visualizations
# 1.2 Why: Creates charts, plots, and images. It's like paint for data!
#          Most other plotting libraries are built on top of matplotlib
# 1.3 When: Need to visualize data, create charts, save images
# 1.4 Where: Reports, dashboards, scientific papers, presentations
# 1.5 How: import matplotlib.pyplot as plt, then plt.plot(), plt.show()
# 1.6 Internal: Renders graphics using backend engines (GTK, Qt, etc.)
# 1.7 Output: Figures and axes objects that can be displayed/saved
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1.1 What: Import seaborn for beautiful statistical visualizations
# 1.2 Why: Seaborn makes matplotlib prettier with less code
#          It's designed specifically for statistical graphics
# 1.3 When: Heatmaps, distribution plots, categorical plots
# 1.4 Where: Data exploration, presentations, publications
# 1.5 How: import seaborn as sns, then sns.heatmap(), sns.barplot()
# 1.6 Internal: Builds on matplotlib, adds professional styling
# 1.7 Output: Beautiful, publication-ready visualizations
# -----------------------------------------------------------------------------
import seaborn as sns

# -----------------------------------------------------------------------------
# 1.1 What: Import warnings to suppress unnecessary messages
# 1.2 Why: Some libraries print warnings that clutter output
#          We want clean, readable output for teaching
# 1.3 When: In production code or teaching notebooks
# 1.4 Where: Any Python script where warnings are distracting
# 1.5 How: warnings.filterwarnings('ignore')
# 1.6 Internal: Intercepts Python's warning system
# 1.7 Output: No output - just prevents warnings from showing
# -----------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1.1 What: Import train_test_split for splitting data
# 1.2 Why: We need to test our model on data it hasn't seen before
#          It's like studying for an exam, then taking the actual exam
#          CRITICAL: Never test on training data (that's cheating!)
# 1.3 When: Before training any ML model
# 1.4 Where: Every supervised ML project
# 1.5 How: X_train, X_test, y_train, y_test = train_test_split(X, y)
# 1.6 Internal: Randomly shuffles and splits data into parts
# 1.7 Output: Four datasets: training features, test features, 
#             training labels, test labels
# -----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1.1 What: Import StandardScaler for feature scaling
# 1.2 Why: ML algorithms work better when features are on same scale
#          Like comparing height in cm vs weight in kg - confusing!
#          StandardScaler converts everything to "standard" units
# 1.3 When: Before training most ML models (except tree-based)
# 1.4 Where: Regression, SVM, KNN, Neural Networks
# 1.5 How: scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
# 1.6 Internal: Subtracts mean and divides by standard deviation
# 1.7 Output: Transformed data with mean=0 and variance=1
# -----------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1.1 What: Import LabelEncoder for encoding text labels
# 1.2 Why: Computers understand numbers, not words like "Healthy"/"Stressed"
#          LabelEncoder converts text to numbers: Healthy=0, Stressed=1
# 1.3 When: Target variable is text/categorical
# 1.4 Where: Classification problems with text labels
# 1.5 How: encoder = LabelEncoder(); y_encoded = encoder.fit_transform(y)
# 1.6 Internal: Creates a mapping dictionary: {label: number}
# 1.7 Output: Numeric array representing original labels
# -----------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# Importing Classification Models
# =============================================================================

# -----------------------------------------------------------------------------
# LOGISTIC REGRESSION
# 1.1 What: A model that predicts probability of belonging to a class
# 1.2 Why: Simple, fast, interpretable - great baseline model
#          Despite the name, it's for CLASSIFICATION, not regression!
# 1.3 When: Binary classification, need interpretability
# 1.4 Where: Medical diagnosis, spam detection, credit scoring
# 1.5 How: model = LogisticRegression(); model.fit(X, y)
# 1.6 Internal: Uses sigmoid function to convert numbers to probabilities
# 1.7 Output: Predictions (0 or 1) and probability estimates
# -----------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# DECISION TREE
# 1.1 What: A tree that makes decisions based on feature thresholds
# 1.2 Why: Easy to understand and visualize - like a flowchart!
#          Doesn't need scaling, handles non-linear relationships
# 1.3 When: Need interpretability, rules extraction
# 1.4 Where: Credit approval, medical diagnosis, game AI
# 1.5 How: model = DecisionTreeClassifier(); model.fit(X, y)
# 1.6 Internal: Recursively splits data to minimize impurity (Gini/entropy)
# 1.7 Output: Predictions based on learned decision rules
# -----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------
# RANDOM FOREST
# 1.1 What: An ensemble of many decision trees working together
# 1.2 Why: More accurate than single trees, reduces overfitting
#          Like asking many experts and taking a vote!
# 1.3 When: Need high accuracy, handling complex patterns
# 1.4 Where: Fraud detection, recommendation systems, genetics
# 1.5 How: model = RandomForestClassifier(); model.fit(X, y)
# 1.6 Internal: Trains many trees on random subsets, averages predictions
# 1.7 Output: Robust predictions, feature importance scores
# -----------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# SUPPORT VECTOR MACHINE (SVM)
# 1.1 What: Finds the best boundary (hyperplane) between classes
# 1.2 Why: Powerful for high-dimensional data, works well with clear margins
#          Can handle non-linear boundaries using "kernel trick"
# 1.3 When: Complex classification, image recognition
# 1.4 Where: Face recognition, text classification, bioinformatics
# 1.5 How: model = SVC(); model.fit(X, y)
# 1.6 Internal: Maximizes margin between support vectors
# 1.7 Output: Class predictions, can output probabilities if enabled
# -----------------------------------------------------------------------------
from sklearn.svm import SVC

# -----------------------------------------------------------------------------
# K-NEAREST NEIGHBORS (KNN)
# 1.1 What: Classifies based on the majority class of nearest neighbors
# 1.2 Why: No training needed (lazy learner), intuitive
#          Like asking your 5 closest friends for their opinion!
# 1.3 When: Small-medium datasets, need simplicity
# 1.4 Where: Recommendation systems, anomaly detection
# 1.5 How: model = KNeighborsClassifier(); model.fit(X, y)
# 1.6 Internal: Calculates distances to all training points, takes vote
# 1.7 Output: Class label based on neighbor majority
# -----------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# Importing Evaluation Metrics
# =============================================================================

# -----------------------------------------------------------------------------
# EVALUATION METRICS
# 1.1 What: Tools to measure how good our model is
# 1.2 Why: We need objective ways to compare models
#          Different metrics for different goals:
#          - accuracy_score: Overall correctness (correct/total)
#          - precision_score: Of predicted positive, how many correct?
#          - recall_score: Of actual positive, how many did we find?
#          - f1_score: Balance between precision and recall
#          - roc_auc_score: Area under ROC curve (overall ranking quality)
#          - confusion_matrix: Shows TP, TN, FP, FN in a table
#          - classification_report: Summary of all metrics
# 1.3 When: After training, to evaluate model performance
# 1.4 Where: Every ML classification project
# 1.5 How: accuracy = accuracy_score(y_true, y_pred)
# 1.6 Internal: Compares predictions with actual labels
# 1.7 Output: Numeric scores (0 to 1 scale, higher is better)
# -----------------------------------------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -----------------------------------------------------------------------------
# OS module for file path handling
# 1.1 What: Provides functions for interacting with the operating system
# 1.2 Why: We need to create folders, save files, handle paths
# 1.3 When: Saving outputs, creating directories
# 1.4 Where: Any script that reads/writes files
# 1.5 How: os.makedirs(), os.path.join()
# 1.6 Internal: Interfaces with OS kernel for file operations
# 1.7 Output: File operations success/failure
# -----------------------------------------------------------------------------
import os

# =============================================================================
# SECTION 2: CONFIGURATION AND SETUP
# =============================================================================

print("=" * 70)
print("🌾 AI-Based Crop Health Monitoring Using Drone Multispectral Data 🚁")
print("=" * 70)
print()

# -----------------------------------------------------------------------------
# 2.1 What: Define the dataset file path
# 2.2 Why: We load data from a local CSV file (downloaded from Google Sheets)
#          This avoids SSL/network issues and ensures consistent data
# 2.3 When: At the start of the script
# 2.4 Where: Data loading section
# 2.5 How: Use pandas read_csv with the local file path
# 2.6 Internal: Reads CSV file from disk, parses into DataFrame
# 2.7 Output: Complete file path for data loading
# -----------------------------------------------------------------------------
DATASET_PATH = "data/crop_health_data.csv"

# -----------------------------------------------------------------------------
# 2.1 What: Create output directory for saving visualizations
# 2.2 Why: We need a place to store generated images
#          Using os.makedirs with exist_ok=True is safe (won't error if exists)
# 2.3 When: Before saving any files
# 2.4 Where: Project setup section
# 2.5 How: os.makedirs(path, exist_ok=True)
# 2.6 Internal: Creates directory tree on filesystem
# 2.7 Output: Directory created (or already exists)
# -----------------------------------------------------------------------------
OUTPUT_DIR = "outputs/sample_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 3: TASK 1 - DATA UNDERSTANDING
# =============================================================================

print("📊 TASK 1: DATA UNDERSTANDING")
print("-" * 40)
print()

# -----------------------------------------------------------------------------
# 3.1 What: Load the dataset from Google Sheets URL
# 3.2 Why: We need to get the drone field data into Python
#          pandas.read_csv() can read directly from URLs!
# 3.3 When: At the start of any data science project
# 3.4 Where: Data loading phase
# 3.5 How: df = pd.read_csv(url)
# 3.6 Internal: Downloads data, parses CSV format, creates DataFrame
# 3.7 Output: DataFrame with 16 columns of vegetation data
#
# ARGUMENTS EXPLANATION:
# - DATASET_URL (str): The web address where the CSV file is located
#   3.1 What: The URL pointing to our dataset
#   3.2 Why: read_csv can fetch files from web URLs, not just local files
#   3.3 When: When data is hosted online (cloud, Google Sheets, etc.)
#   3.4 Where: Cloud-based data science workflows
#   3.5 How: Pass as first argument to read_csv
#   3.6 Internal: Uses urllib to make HTTP request, streams response
#   3.7 Output: DataFrame loaded from remote source
# -----------------------------------------------------------------------------
print("Loading dataset from local file...")
df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset loaded successfully from {DATASET_PATH}!")
print()

# -----------------------------------------------------------------------------
# 3.1 What: Display basic dataset information
# 3.2 Why: First step is always to understand what data looks like
#          Shape tells us: (rows, columns) = (samples, features)
# 3.3 When: Immediately after loading data
# 3.4 Where: Exploratory Data Analysis (EDA) phase
# 3.5 How: df.shape, df.columns, df.info()
# 3.6 Internal: Accesses DataFrame metadata stored in memory
# 3.7 Output: Tuple (n_rows, n_cols), list of column names
# -----------------------------------------------------------------------------
print("📋 Dataset Information:")
print(f"   • Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   • Each row = one spatial observation in the field")
print(f"   • Each column = one measurement or metadata")
print()

# -----------------------------------------------------------------------------
# 3.1 What: List all columns with their data types
# 3.2 Why: Understanding features is crucial for ML
#          Different data types need different handling
# 3.3 When: During EDA
# 3.4 Where: Any data science project
# 3.5 How: df.columns to list, df.dtypes for types
# 3.6 Internal: Returns Index object with column labels
# 3.7 Output: List of feature names
# -----------------------------------------------------------------------------
print("📊 Columns (Features and Target):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")
print()

# -----------------------------------------------------------------------------
# 3.1 What: Display first 5 rows of data
# 3.2 Why: Visual check to see actual data values
#          head() shows first n rows (default=5)
# 3.3 When: During data exploration
# 3.4 Where: EDA phase
# 3.5 How: df.head(n) where n is number of rows
# 3.6 Internal: Slices DataFrame to first n rows
# 3.7 Output: DataFrame with only first n rows
#
# ARGUMENTS EXPLANATION:
# - n (int): Number of rows to display
#   3.1 What: How many rows to show from the top
#   3.2 Why: We don't want to print thousands of rows!
#   3.3 When: Quick data inspection
#   3.4 Where: Any tabular data exploration
#   3.5 How: Pass integer value (default is 5)
#   3.6 Internal: Uses integer indexing [:n]
#   3.7 Output: Subset DataFrame
# -----------------------------------------------------------------------------
print("🔍 First 5 rows of data:")
print(df.head())
print()

# -----------------------------------------------------------------------------
# 3.1 What: Display statistical summary of the data
# 3.2 Why: describe() shows mean, std, min, max, quartiles
#          Helps identify outliers, scaling needs, data quality
# 3.3 When: During EDA
# 3.4 Where: Every data science project
# 3.5 How: df.describe()
# 3.6 Internal: Calculates statistics for each numeric column
# 3.7 Output: DataFrame with statistics as rows
# -----------------------------------------------------------------------------
print("📈 Statistical Summary:")
print(df.describe().round(3))
print()

# -----------------------------------------------------------------------------
# 3.1 What: Check for missing values
# 3.2 Why: Missing values can break ML models or cause errors
#          isnull().sum() counts NaN values per column
# 3.3 When: Data cleaning phase
# 3.4 Where: Every data project
# 3.5 How: df.isnull().sum()
# 3.6 Internal: Creates boolean mask, sums True values
# 3.7 Output: Series with count of missing values per column
# -----------------------------------------------------------------------------
print("🔎 Missing Values Check:")
missing = df.isnull().sum().sum()
if missing == 0:
    print("   ✅ No missing values found! Data is complete.")
else:
    print(f"   ⚠️ Found {missing} missing values. Need cleaning!")
print()

# -----------------------------------------------------------------------------
# 3.1 What: Check target variable distribution
# 3.2 Why: Imbalanced classes need special handling
#          value_counts() shows how many of each class we have
# 3.3 When: Classification problem setup
# 3.4 Where: Binary/multi-class classification projects
# 3.5 How: df['column'].value_counts()
# 3.6 Internal: Groups by unique values, counts occurrences
# 3.7 Output: Series with counts per unique value
# -----------------------------------------------------------------------------
print("🎯 Target Variable Distribution (crop_health_label):")
target_counts = df['crop_health_label'].value_counts()
for label, count in target_counts.items():
    percentage = count / len(df) * 100
    emoji = "🟢" if label == "Healthy" else "🔴"
    print(f"   {emoji} {label}: {count} samples ({percentage:.1f}%)")
print()

# -----------------------------------------------------------------------------
# VEGETATION INDEX EXPLANATION
# 3.1 What: Explain what each vegetation index measures
# 3.2 Why: Domain knowledge is essential for interpretation
#          These indices are derived from multispectral drone imagery
# 3.3 When: Data understanding phase
# 3.4 Where: Agricultural AI projects
# 3.5 How: Print educational information
# 3.6 Internal: String output to console
# 3.7 Output: Educational text
# -----------------------------------------------------------------------------
print("=" * 70)
print("📚 VEGETATION INDEX EXPLANATIONS")
print("=" * 70)
print("""
🌿 NDVI (Normalized Difference Vegetation Index):
   • Most popular index for plant health
   • Formula: (NIR - Red) / (NIR + Red)
   • Range: -1 to +1 (higher = healthier vegetation)
   • Like checking how green and alive a plant looks

🌱 GNDVI (Green NDVI):
   • Similar to NDVI but uses green light instead of red
   • Better for detecting chlorophyll content variations
   • Useful for assessing nitrogen levels

🏜️ SAVI (Soil-Adjusted Vegetation Index):
   • NDVI adjusted for soil brightness
   • Better in areas with sparse vegetation
   • Reduces soil interference in measurements

📊 EVI (Enhanced Vegetation Index):
   • More sensitive in high biomass regions
   • Corrects for atmospheric and canopy effects
   • Used in MODIS satellite products

🔴 Red-Edge Bands:
   • Special wavelengths where vegetation reflectance changes rapidly
   • Very sensitive to chlorophyll content
   • Early indicator of plant stress

☀️ NIR Reflectance:
   • Near-infrared light reflected by plants
   • Healthy plants reflect more NIR
   • Related to cell structure and water content

🌍 Soil Brightness:
   • How bright the soil appears
   • Affected by moisture, organic content
   • Helps separate plant from soil signal

🌳 Canopy Density:
   • How much of ground is covered by leaves
   • Higher density = more biomass
   • Indicates crop growth stage

💧 Moisture Index:
   • Water content in vegetation
   • Stressed plants often show lower moisture
   • Early indicator of drought stress
""")

# =============================================================================
# SECTION 4: TASK 2 - MACHINE LEARNING MODEL COMPARISON
# =============================================================================

print("=" * 70)
print("🤖 TASK 2: MACHINE LEARNING MODEL COMPARISON")
print("=" * 70)
print()

# -----------------------------------------------------------------------------
# 4.1 What: Separate features (X) from target (y)
# 4.2 Why: ML models need separate input (X) and output (y)
#          Features = what we use to predict
#          Target = what we want to predict
# 4.3 When: Before any ML training
# 4.4 Where: Every supervised learning project
# 4.5 How: X = df.drop('target', axis=1), y = df['target']
# 4.6 Internal: Creates new DataFrame without target column
# 4.7 Output: X (features DataFrame), y (target Series)
#
# ARGUMENTS EXPLANATION:
# - 'crop_health_label' (str): Column name to drop
#   What: The target variable we want to predict
#   Why: We don't want the model to "cheat" by seeing the answer
# - axis=1: 
#   What: Specifies we're dropping a column, not a row
#   Why: axis=0 is rows, axis=1 is columns
#   How: Use axis=1 for column operations
# -----------------------------------------------------------------------------
print("🔧 Preparing data for machine learning...")
print()

# Define feature columns (everything except grid coordinates and target)
feature_columns = ['ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max', 'gndvi', 
                   'savi', 'evi', 'red_edge_1', 'red_edge_2', 'nir_reflectance',
                   'soil_brightness', 'canopy_density', 'moisture_index']

# Keep grid coordinates separate for spatial analysis later
grid_columns = ['grid_x', 'grid_y']

# Features for ML model (vegetation indices only)
X = df[feature_columns]

# Target variable
y = df['crop_health_label']

print(f"   • Feature columns: {len(feature_columns)}")
print(f"   • Feature names: {', '.join(feature_columns[:5])}... (+{len(feature_columns)-5} more)")
print(f"   • Target column: crop_health_label")
print()

# -----------------------------------------------------------------------------
# 4.1 What: Encode target labels from text to numbers
# 4.2 Why: ML models need numeric targets, not text
#          LabelEncoder: "Healthy" → 0, "Stressed" → 1
# 4.3 When: When target is categorical (text)
# 4.4 Where: Classification problems
# 4.5 How: encoder.fit_transform(y)
# 4.6 Internal: Creates mapping, applies to all values
# 4.7 Output: NumPy array of integers
#
# FIT vs TRANSFORM:
# - fit(): Learn the mapping from data
# - transform(): Apply the mapping
# - fit_transform(): Do both at once (convenience method)
# -----------------------------------------------------------------------------
print("🏷️ Encoding target labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"   Original labels: {list(label_encoder.classes_)}")
print(f"   Encoded as: {list(range(len(label_encoder.classes_)))}")
print(f"   Example: 'Healthy' → 0, 'Stressed' → 1")
print()

# -----------------------------------------------------------------------------
# 4.1 What: Split data into training and testing sets
# 4.2 Why: We MUST evaluate on unseen data to check generalization
#          Like studying for exam (training) then taking exam (testing)
#          Prevents OVERFITTING (memorizing instead of learning)
# 4.3 When: Before training any ML model
# 4.4 Where: Every supervised ML project
# 4.5 How: train_test_split(X, y, test_size=0.2, random_state=42)
# 4.6 Internal: Randomly shuffles data, splits by percentage
# 4.7 Output: 4 datasets: X_train, X_test, y_train, y_test
#
# ARGUMENTS EXPLANATION:
# - test_size=0.2:
#   What: 20% of data for testing, 80% for training
#   Why: Standard split; enough training data, enough for valid testing
#   When: Adjust based on dataset size (small data: use cross-validation)
#
# - random_state=42:
#   What: Seed for random number generator
#   Why: Makes results REPRODUCIBLE (same split every time)
#   When: Always use in research/teaching for consistency
#   Why 42?: It's a famous number from "Hitchhiker's Guide to the Galaxy"
#
# - stratify=y_encoded:
#   What: Ensures train/test have same class proportions as original
#   Why: Prevents accidentally putting all "Stressed" in test set
#   When: Important for imbalanced datasets
# -----------------------------------------------------------------------------
print("✂️ Splitting data into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)
print(f"   • Training samples: {len(X_train)}")
print(f"   • Testing samples: {len(X_test)}")
print()

# -----------------------------------------------------------------------------
# 4.1 What: Scale features to same range
# 4.2 Why: Features have different scales (NDVI: 0-1, NIR: 0.2-0.9)
#          Many algorithms are sensitive to scale (SVM, KNN, Logistic)
#          StandardScaler: transforms to mean=0, std=1
# 4.3 When: Before training scale-sensitive models
# 4.4 Where: Most ML pipelines (except tree-based models)
# 4.5 How: scaler.fit_transform(X_train), scaler.transform(X_test)
# 4.6 Internal: z = (x - mean) / std for each feature
# 4.7 Output: Scaled arrays with standardized values
#
# IMPORTANT: fit on TRAIN only, transform BOTH
# Why? Test data should be treated as "unseen" - we can't use its statistics!
# Leaking test data info = DATA LEAKAGE = invalid results
# -----------------------------------------------------------------------------
print("📏 Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit AND transform on training
X_test_scaled = scaler.transform(X_test)         # Only transform on testing
print("   ✅ Features scaled to zero mean and unit variance")
print()

# -----------------------------------------------------------------------------
# 4.1 What: Define models to compare
# 4.2 Why: Different models have different strengths
#          We'll test 5 popular classification algorithms
# 4.3 When: During model selection phase
# 4.4 Where: ML experiments
# 4.5 How: Create dictionary of model_name: model_object
# 4.6 Internal: Each model has different learning algorithms
# 4.7 Output: Dictionary for easy iteration
#
# MODEL PARAMETERS EXPLANATION:
#
# LogisticRegression(max_iter=1000):
#   - max_iter: Maximum iterations for solver convergence
#   - Default 100 sometimes not enough, 1000 is safer
#
# DecisionTreeClassifier(random_state=42):
#   - random_state: Reproducibility for tree building
#
# RandomForestClassifier(n_estimators=100, random_state=42):
#   - n_estimators: Number of trees in the forest (more = better but slower)
#   - 100 is a good default balance
#
# SVC(probability=True, random_state=42):
#   - probability=True: Enable probability estimation for ROC-AUC
#   - Slower training but needed for AUC calculation
#
# KNeighborsClassifier(n_neighbors=5):
#   - n_neighbors: How many neighbors to consider (k in KNN)
#   - 5 is common default; odd number avoids ties
# -----------------------------------------------------------------------------
print("🏭 Initializing Classification Models:")
print()

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

for name in models:
    print(f"   ✓ {name}")
print()

# -----------------------------------------------------------------------------
# 4.1 What: Train and evaluate each model
# 4.2 Why: Compare performance to find the best model
# 4.3 When: Model comparison phase
# 4.4 Where: ML experiments
# 4.5 How: Loop through models, fit, predict, calculate metrics
# 4.6 Internal: Each model learns patterns from training data
# 4.7 Output: Dictionary of results for comparison
# -----------------------------------------------------------------------------
print("=" * 70)
print("📊 TRAINING AND EVALUATION RESULTS")
print("=" * 70)
print()

results = []
best_model = None
best_model_name = None
best_f1 = 0

for name, model in models.items():
    print(f"🔄 Training: {name}...")
    
    # Train the model on scaled training data
    # fit() is the learning step where model finds patterns
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on test data
    # predict() uses learned patterns to classify new data
    y_pred = model.predict(X_test_scaled)
    
    # Get probability predictions for ROC-AUC
    # predict_proba() gives probability of each class
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = y_pred
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    # Track best model (by F1-Score)
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name
    
    # Print results
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    print()

# -----------------------------------------------------------------------------
# 4.1 What: Create comparison table
# 4.2 Why: Easy visual comparison of all models
# 4.3 When: After all models are trained
# 4.4 Where: Model selection documentation
# 4.5 How: Create DataFrame from results list
# 4.6 Internal: DataFrame organizes data in tabular format
# 4.7 Output: Formatted comparison table
# -----------------------------------------------------------------------------
print("=" * 70)
print("📋 MODEL COMPARISON TABLE")
print("=" * 70)
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)
print(results_df.to_string(index=False))
print()

print(f"🏆 BEST MODEL: {best_model_name} (F1-Score: {best_f1:.4f})")
print()

# -----------------------------------------------------------------------------
# 4.1 What: Print detailed classification report for best model
# 4.2 Why: Shows precision/recall per class
# 4.3 When: After selecting best model
# 4.4 Where: Model evaluation section
# 4.5 How: classification_report(y_true, y_pred)
# 4.6 Internal: Calculates per-class and weighted metrics
# 4.7 Output: String with formatted report
# -----------------------------------------------------------------------------
print("=" * 70)
print(f"📊 DETAILED REPORT: {best_model_name}")
print("=" * 70)
print()

y_pred_best = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best, target_names=['Healthy', 'Stressed']))

# -----------------------------------------------------------------------------
# 4.1 What: Create confusion matrix visualization
# 4.2 Why: Shows true vs predicted labels visually
# 4.3 When: Model evaluation
# 4.4 Where: Classification projects
# 4.5 How: confusion_matrix() + seaborn heatmap
# 4.6 Internal: Counts TP, TN, FP, FN
# 4.7 Output: Heatmap image saved to file
# -----------------------------------------------------------------------------
print("📊 Creating confusion matrix visualization...")
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Stressed'],
            yticklabels=['Healthy', 'Stressed'])
plt.title(f'Confusion Matrix: {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print(f"   ✅ Saved: {OUTPUT_DIR}/confusion_matrix.png")
print()

# -----------------------------------------------------------------------------
# 4.1 What: Create model comparison bar chart
# 4.2 Why: Visual comparison is intuitive
# 4.3 When: After model comparison
# 4.4 Where: Presentations, reports
# 4.5 How: matplotlib bar chart
# 4.6 Internal: Plots rectangles with heights = metric values
# 4.7 Output: Bar chart image
# -----------------------------------------------------------------------------
print("📊 Creating model comparison chart...")
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(results_df))
width = 0.15

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i])

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
plt.close()
print(f"   ✅ Saved: {OUTPUT_DIR}/model_comparison.png")
print()

# =============================================================================
# SECTION 5: TASK 3 - SPATIAL ANALYSIS & VISUALIZATION
# =============================================================================

print("=" * 70)
print("🗺️ TASK 3: SPATIAL ANALYSIS & VISUALIZATION")
print("=" * 70)
print()

# -----------------------------------------------------------------------------
# 5.1 What: Make predictions on entire dataset
# 5.2 Why: We need predictions for all grid cells to create heatmap
# 5.3 When: After model is trained
# 5.4 Where: Spatial analysis phase
# 5.5 How: Scale all features, predict with best model
# 5.6 Internal: Model applies learned patterns to all samples
# 5.7 Output: Prediction for each grid cell
# -----------------------------------------------------------------------------
print("🔮 Generating predictions for all grid cells...")

# Scale all features (not just train/test)
X_all_scaled = scaler.transform(X)

# Predict on all data
y_all_pred = best_model.predict(X_all_scaled)
y_all_proba = best_model.predict_proba(X_all_scaled)[:, 1]  # Probability of being stressed

# Add predictions to dataframe
df['predicted_label'] = label_encoder.inverse_transform(y_all_pred)
df['stress_probability'] = y_all_proba

print(f"   ✅ Predictions generated for {len(df)} grid cells")
print()

# -----------------------------------------------------------------------------
# 5.1 What: Create spatial stress heatmap
# 5.2 Why: Visual representation of stress across the field
#          Red areas = stressed, Green areas = healthy
# 5.3 When: After predictions are made
# 5.4 Where: Field visualization
# 5.5 How: Create pivot table, plot as heatmap
# 5.6 Internal: Reorganizes data into grid format
# 5.7 Output: Heatmap image
# -----------------------------------------------------------------------------
print("🗺️ Creating field stress heatmap...")

# Create pivot table for heatmap (grid_x vs grid_y with stress probability)
# pivot_table aggregates if multiple observations per cell
heatmap_data = df.pivot_table(
    values='stress_probability',
    index='grid_y',
    columns='grid_x',
    aggfunc='mean'
)

# Create the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    heatmap_data,
    cmap='RdYlGn_r',  # Red-Yellow-Green reversed (Red = high stress)
    annot=False,
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Stress Probability'}
)
plt.title('🌾 Field Stress Heatmap\n(Red = Stressed, Green = Healthy)', fontsize=14)
plt.xlabel('Grid X (Column)')
plt.ylabel('Grid Y (Row)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stress_heatmap.png'), dpi=150)
plt.close()
print(f"   ✅ Saved: {OUTPUT_DIR}/stress_heatmap.png")
print()

# -----------------------------------------------------------------------------
# 5.1 What: Calculate stress statistics by zone
# 5.2 Why: Aggregate data for area-level insights
# 5.3 When: After predictions
# 5.4 Where: Spatial analysis
# 5.5 How: Group by grid position, calculate statistics
# 5.6 Internal: SQL-like GROUP BY operation
# 5.7 Output: Summary statistics
# -----------------------------------------------------------------------------
print("📊 Stress Distribution Summary:")
stress_counts = df['predicted_label'].value_counts()
for label, count in stress_counts.items():
    percentage = count / len(df) * 100
    emoji = "🟢" if label == "Healthy" else "🔴"
    print(f"   {emoji} {label}: {count} cells ({percentage:.1f}%)")
print()

# Calculate high-stress zones (stress probability > 0.7)
high_stress_cells = df[df['stress_probability'] > 0.7]
print(f"⚠️ High-Stress Zones (probability > 0.7): {len(high_stress_cells)} cells")
print()

# =============================================================================
# SECTION 6: TASK 4 - DRONE & AGRONOMY INTERPRETATION
# =============================================================================

print("=" * 70)
print("🚁 TASK 4: DRONE INSPECTION RECOMMENDATIONS")
print("=" * 70)
print()

# -----------------------------------------------------------------------------
# 6.1 What: Categorize stress levels
# 6.2 Why: Prioritize inspection based on severity
# 6.3 When: Interpretation phase
# 6.4 Where: Agricultural decision support
# 6.5 How: Define thresholds, categorize each cell
# 6.6 Internal: Apply conditional logic
# 6.7 Output: Priority categories
# -----------------------------------------------------------------------------
def categorize_stress(prob):
    """
    Categorize stress probability into priority levels.
    
    ARGUMENTS:
    - prob (float): Stress probability from 0 to 1
      What: The model's confidence that the cell is stressed
      Why: Higher probability = higher priority for inspection
      When: After predictions are made
      Where: Each grid cell
      How: Compare against thresholds
      Internal: Simple conditional comparison
      Output: String category label
    
    RETURNS:
    - str: Priority level ('CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'HEALTHY')
    """
    if prob >= 0.8:
        return 'CRITICAL'
    elif prob >= 0.6:
        return 'HIGH'
    elif prob >= 0.4:
        return 'MODERATE'
    elif prob >= 0.2:
        return 'LOW'
    else:
        return 'HEALTHY'

df['stress_priority'] = df['stress_probability'].apply(categorize_stress)

print("📋 INSPECTION PRIORITY ZONES:")
print()

priority_counts = df['stress_priority'].value_counts()
priority_order = ['CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'HEALTHY']
priority_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MODERATE': '🟡', 'LOW': '🟢', 'HEALTHY': '✅'}

for priority in priority_order:
    if priority in priority_counts.index:
        count = priority_counts[priority]
        percentage = count / len(df) * 100
        emoji = priority_emoji[priority]
        print(f"   {emoji} {priority}: {count} cells ({percentage:.1f}%)")
print()

# -----------------------------------------------------------------------------
# 6.1 What: Generate specific inspection recommendations
# 6.2 Why: Actionable insights for farmers
# 6.3 When: After analysis
# 6.4 Where: Decision support system
# 6.5 How: Filter high-priority cells, list coordinates
# 6.6 Internal: DataFrame filtering
# 6.7 Output: List of recommended inspection locations
# -----------------------------------------------------------------------------
print("🚁 RECOMMENDED DRONE FLIGHT PATH:")
print()

critical_zones = df[df['stress_priority'] == 'CRITICAL'][['grid_x', 'grid_y', 'stress_probability']]
high_zones = df[df['stress_priority'] == 'HIGH'][['grid_x', 'grid_y', 'stress_probability']]

if len(critical_zones) > 0:
    print("⚠️ CRITICAL ZONES - Inspect IMMEDIATELY:")
    for _, row in critical_zones.head(10).iterrows():
        print(f"   📍 Grid ({int(row['grid_x'])}, {int(row['grid_y'])}) - Stress: {row['stress_probability']:.1%}")
    if len(critical_zones) > 10:
        print(f"   ... and {len(critical_zones)-10} more critical zones")
    print()

if len(high_zones) > 0:
    print("🟠 HIGH PRIORITY - Inspect within 24 hours:")
    for _, row in high_zones.head(5).iterrows():
        print(f"   📍 Grid ({int(row['grid_x'])}, {int(row['grid_y'])}) - Stress: {row['stress_probability']:.1%}")
    if len(high_zones) > 5:
        print(f"   ... and {len(high_zones)-5} more high priority zones")
    print()

print("💡 INTERPRETATION GUIDELINES:")
print("""
   1. CRITICAL zones require immediate ground inspection
   2. High stress may indicate:
      - Pest infestation
      - Disease outbreak
      - Water stress (drought or flooding)
      - Nutrient deficiency
   3. Consider sending follow-up drones with higher resolution
   4. Compare with historical data for trend analysis
""")

# =============================================================================
# SECTION 7: TASK 5 - REFLECTION
# =============================================================================

print("=" * 70)
print("📝 TASK 5: REFLECTION - LIMITATIONS & IMPROVEMENTS")
print("=" * 70)
print()

print("⚠️ CURRENT LIMITATIONS:")
print("""
   1. 📊 Dataset Size:
      - Current dataset is synthetic/limited
      - Real-world farms have millions of data points
      - More data would improve model accuracy

   2. 🌦️ Temporal Data Missing:
      - We have single snapshot, not time series
      - Plant stress develops over time
      - Tracking trends would improve prediction

   3. 🌡️ Weather Data Not Included:
      - Temperature, rainfall affect plant health
      - Integrating weather would improve accuracy
      - Historical weather patterns important

   4. 🦠 No Disease/Pest Classification:
      - Current model only detects "stress"
      - Doesn't tell us WHAT is causing stress
      - Multi-class classification would be better

   5. 🛰️ Single Resolution:
      - Fixed grid resolution may miss details
      - Multi-scale analysis would be beneficial
""")

print("🚀 PROPOSED IMPROVEMENTS:")
print("""
   1. 🔄 Time Series Analysis:
      - Collect data over growing season
      - Use LSTM or temporal models
      - Predict future stress emergence

   2. 🧠 Deep Learning:
      - Use CNNs on raw imagery
      - Transfer learning from agricultural datasets
      - Semantic segmentation for precise boundaries

   3. 🌐 Multi-Source Data Fusion:
      - Integrate weather stations
      - Add soil sensors
      - Include farmer observations

   4. 📱 Real-Time Processing:
      - Edge computing on drones
      - Instant alerts during flight
      - Adaptive flight paths

   5. 🤝 Farmer Feedback Loop:
      - Validate predictions with ground truth
      - Continuous model improvement
      - Domain expert knowledge integration
""")

# =============================================================================
# SECTION 8: FINAL SUMMARY
# =============================================================================

print("=" * 70)
print("✅ PROJECT EXECUTION COMPLETE")
print("=" * 70)
print()

print("📁 OUTPUT FILES GENERATED:")
print(f"   • {OUTPUT_DIR}/confusion_matrix.png")
print(f"   • {OUTPUT_DIR}/model_comparison.png")
print(f"   • {OUTPUT_DIR}/stress_heatmap.png")
print()

print("🏆 KEY FINDINGS:")
print(f"   • Best Model: {best_model_name}")
print(f"   • F1-Score: {best_f1:.4f}")
print(f"   • Total Grid Cells Analyzed: {len(df)}")
print(f"   • Stressed Cells Identified: {len(df[df['predicted_label'] == 'Stressed'])}")
print(f"   • Critical Zones: {len(df[df['stress_priority'] == 'CRITICAL'])}")
print()

print("🎯 RECOMMENDATIONS:")
print("   1. Deploy drone to critical zones first")
print("   2. Take close-up images of stressed plants")
print("   3. Consult agronomist for treatment plan")
print("   4. Schedule re-scan after treatment")
print()

print("=" * 70)
print("🌾 Thank you for using AI Crop Health Monitoring! 🚁")
print("=" * 70)
