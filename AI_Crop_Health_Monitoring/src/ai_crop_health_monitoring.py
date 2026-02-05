"""
================================================================================
AI-BASED CROP HEALTH MONITORING USING DRONE MULTISPECTRAL DATA
================================================================================

🌾 PROJECT OVERVIEW:
This script builds an end-to-end AI pipeline to detect crop stress using 
multispectral vegetation indices derived from drone imagery.

📋 WHAT THIS SCRIPT DOES:
1. Loads and explores vegetation index data
2. Trains a Random Forest classifier to predict crop health
3. Evaluates model using Precision, Recall, F1-Score, ROC-AUC
4. Creates a spatial stress heatmap
5. Provides drone inspection recommendations

👶 SIMPLE EXPLANATION:
Imagine you have a huge farm and want to find which plants are sick.
A drone flies over and takes special photos. This script teaches a computer
to look at those photos and say "Healthy" or "Stressed" for each area.

================================================================================
"""

# ==============================================================================
# SECTION 0: CONFIGURE UTF-8 ENCODING (Windows Fix)
# ==============================================================================
# 
# 📘 WHAT IS THIS?
# Windows console may not support UTF-8 by default, causing emoji errors.
# This fixes Unicode encoding issues on Windows.
#
# ==============================================================================

import sys
import io

# Fix Windows console UTF-8 encoding for emoji support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==============================================================================
# SECTION 1: IMPORT LIBRARIES
# ==============================================================================
# 
# 📘 WHAT ARE IMPORTS?
# Imports bring tools (code written by others) into our program.
# Like borrowing a calculator instead of building one from scratch.
#
# 📘 WHY DO WE NEED THEM?
# - pandas: To work with data tables (like Excel in Python)
# - numpy: For mathematical operations on numbers
# - sklearn: For machine learning algorithms
# - matplotlib/seaborn: For creating charts and visualizations
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import pandas library and give it a nickname 'pd'
# 2.2 WHY: pandas helps us read CSV files and work with data tables
#          Alternative: Pure Python lists (harder), SQL databases (overkill)
#          pandas is BEST because it's fast and has many helpful functions
# 2.3 WHEN: Always when working with tabular data (rows and columns)
# 2.4 WHERE: Data science, analytics, ML preprocessing
# 2.5 HOW: import pandas as pd, then use pd.read_csv(), pd.DataFrame()
# 2.6 INTERNAL: pandas loads C-optimized code for fast data operations
# 2.7 OUTPUT: No visible output, but 'pd' is now available to use
# ------------------------------------------------------------------------------
import pandas as pd

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import numpy library and give it a nickname 'np'
# 2.2 WHY: numpy provides fast mathematical operations on arrays
#          Alternative: Python lists (10-100x slower)
#          numpy is BEST for numerical computing
# 2.3 WHEN: Whenever doing math on collections of numbers
# 2.4 WHERE: Scientific computing, ML, data analysis
# 2.5 HOW: import numpy as np, then use np.array(), np.mean()
# 2.6 INTERNAL: Uses C/Fortran code for blazing fast calculations
# 2.7 OUTPUT: No visible output, but 'np' is now available
# ------------------------------------------------------------------------------
import numpy as np

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import train_test_split function from sklearn
# 2.2 WHY: Splits data into training (to learn) and testing (to evaluate)
#          Alternative: Manual splitting (error-prone)
#          This function ensures random, balanced splitting
# 2.3 WHEN: Before training any ML model
# 2.4 WHERE: Every ML project needs this
# 2.5 HOW: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 2.6 INTERNAL: Randomly shuffles and divides data by percentage
# 2.7 OUTPUT: 4 separate datasets
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import RandomForestClassifier - our ML model
# 2.2 WHY: Random Forest is robust, handles many features well, and gives
#          good accuracy without much tuning
#          Alternative: Decision Tree (less accurate), SVM (slower)
#          Random Forest is BEST for tabular data classification
# 2.3 WHEN: For classification problems with structured data
# 2.4 WHERE: Healthcare, finance, agriculture, fraud detection
# 2.5 HOW: model = RandomForestClassifier(), model.fit(X, y), model.predict(X)
# 2.6 INTERNAL: Creates many decision trees and combines their votes
# 2.7 OUTPUT: A trained model object
# ------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import evaluation metrics for classification
# 2.2 WHY: To measure how good our model is at predictions
#          - classification_report: Precision, Recall, F1 in one table
#          - confusion_matrix: Shows correct vs wrong predictions
#          - roc_auc_score: Area under ROC curve (0-1, higher better)
# 2.3 WHEN: After training, to evaluate model performance
# 2.4 WHERE: Every ML classification project
# 2.5 HOW: print(classification_report(y_true, y_pred))
# 2.6 INTERNAL: Compares predicted labels with actual labels
# 2.7 OUTPUT: Performance numbers and tables
# ------------------------------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import LabelEncoder to convert text labels to numbers
# 2.2 WHY: ML models need numbers, not text like "Healthy"/"Stressed"
#          Alternative: Manual mapping (tedious), OneHotEncoder (for multi-class)
#          LabelEncoder is BEST for binary classification
# 2.3 WHEN: When target variable has text labels
# 2.4 WHERE: Preprocessing step in ML pipeline
# 2.5 HOW: encoder = LabelEncoder(), y_encoded = encoder.fit_transform(y)
# 2.6 INTERNAL: Maps each unique label to an integer (0, 1, 2, ...)
# 2.7 OUTPUT: Array of integers instead of strings
# ------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import matplotlib for creating visualizations
# 2.2 WHY: To create charts, graphs, and heatmaps
#          Alternative: Plotly (interactive), Bokeh (web-based)
#          matplotlib is BEST for static, publication-quality figures
# 2.3 WHEN: Whenever you need to visualize data
# 2.4 WHERE: Data analysis, reporting, presentations
# 2.5 HOW: plt.figure(), plt.plot(), plt.show()
# 2.6 INTERNAL: Uses rendering engines to draw graphics
# 2.7 OUTPUT: Images/plots displayed or saved
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import seaborn for statistical visualizations
# 2.2 WHY: Makes beautiful charts with less code than matplotlib
#          Built on top of matplotlib but prettier
# 2.3 WHEN: For statistical plots like heatmaps, distributions
# 2.4 WHERE: EDA, presentations, reports
# 2.5 HOW: sns.heatmap(data), sns.boxplot(x, y)
# 2.6 INTERNAL: Wraps matplotlib with better defaults
# 2.7 OUTPUT: Professional-looking plots
# ------------------------------------------------------------------------------
import seaborn as sns

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import os module for file/folder operations
# 2.2 WHY: To create output directories, check if files exist
# 2.3 WHEN: When saving files or checking paths
# 2.4 WHERE: Any project that reads/writes files
# 2.5 HOW: os.makedirs('folder'), os.path.exists('file')
# 2.6 INTERNAL: Interfaces with operating system
# 2.7 OUTPUT: Created folders, path checks
# ------------------------------------------------------------------------------
import os

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import warnings module to control warning messages
# 2.2 WHY: To hide unnecessary warning messages that clutter output
# 2.3 WHEN: When libraries produce many warnings
# 2.4 WHERE: Production code, notebooks
# 2.5 HOW: warnings.filterwarnings('ignore')
# 2.6 INTERNAL: Filters Python's warning system
# 2.7 OUTPUT: Cleaner console output
# ------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 2.1 WHAT: Import ssl to handle secure connections
# 2.2 WHY: Fixes "certificate verify failed" errors when downloading data
#          Common on Windows or restricted networks
# 2.3 WHEN: Downloading data from HTTPS URLs
# 2.4 WHERE: Global setup
# 2.5 HOW: ssl._create_default_https_context = ssl._create_unverified_context
# 2.6 INTERNAL: Sets global default context to unverified
# 2.7 OUTPUT: Prevents URLError
# ------------------------------------------------------------------------------
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 60)
print("🌾 AI-BASED CROP HEALTH MONITORING SYSTEM")
print("=" * 60)


# ==============================================================================
# SECTION 2: CONFIGURATION AND SETUP
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Define the dataset URL
# 2.2 WHY: Store the data source location for easy reference
# 2.3 WHEN: At the start to configure data source
# 2.4 WHERE: Configuration section
# 2.5 HOW: Use this URL with pandas to download data
# 2.6 INTERNAL: String stored in memory
# 2.7 OUTPUT: No visible output, variable created
# ------------------------------------------------------------------------------
DATA_URL = "https://docs.google.com/spreadsheets/d/1wPL7_G65NBY7801PfKhbsM7ujANoID6DIzb2zmcJ1yM/export?format=csv"

# ------------------------------------------------------------------------------
# 2.1 WHAT: Define output directory path
# 2.2 WHY: All visualizations will be saved here
# 2.3 WHEN: Before creating any plots
# 2.4 WHERE: Configuration section
# 2.5 HOW: Use os.makedirs() to create this folder
# 2.6 INTERNAL: String path stored in memory
# 2.7 OUTPUT: No visible output
# ------------------------------------------------------------------------------
OUTPUT_DIR = "outputs"

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create output directory if it doesn't exist
# 2.2 WHY: To ensure we can save files without errors
# 2.3 WHEN: Before saving any files
# 2.4 WHERE: Setup phase
# 2.5 HOW: os.makedirs(path, exist_ok=True)
# 2.6 INTERNAL: Creates folder in file system
# 2.7 OUTPUT: Folder created (or nothing if exists)
#
# 3.1-3.7 ARGUMENT: exist_ok=True
# 3.1 WHAT: Prevents error if folder already exists
# 3.2 WHY: Script won't crash on second run
# 3.3 WHEN: Always use this for safety
# 3.4 WHERE: Any makedirs call
# 3.5 HOW: exist_ok=True (boolean parameter)
# 3.6 INTERNAL: Skips creation if folder exists
# 3.7 OUTPUT: No error even if folder present
# ------------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# SECTION 3: DATA LOADING (TASK 1 - Part A)
# ==============================================================================

print("\n" + "=" * 60)
print("📥 TASK 1: DATA UNDERSTANDING")
print("=" * 60)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Load data from Google Sheets CSV export
# 2.2 WHY: Get the vegetation index data into Python for analysis
#          Alternative: Download manually, use APIs
#          pd.read_csv() is BEST for simple CSV access
# 2.3 WHEN: At the start of any data project
# 2.4 WHERE: Data loading phase
# 2.5 HOW: df = pd.read_csv(url_or_filepath)
# 2.6 INTERNAL: Downloads CSV, parses rows/columns into DataFrame
# 2.7 OUTPUT: DataFrame object with all data
#
# 3.1-3.7 ARGUMENT: DATA_URL
# 3.1 WHAT: The URL of the CSV file to read
# 3.2 WHY: Tells pandas where to get the data
# 3.3 WHEN: Always need a source for read_csv
# 3.4 WHERE: First argument to read_csv()
# 3.5 HOW: Can be URL or file path string
# 3.6 INTERNAL: pandas downloads from URL
# 3.7 OUTPUT: Data loaded into DataFrame
# ------------------------------------------------------------------------------
print("\n📂 Loading dataset from Google Sheets...")
df = pd.read_csv(DATA_URL)
print(f"✅ Dataset loaded successfully!")
print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")


# ==============================================================================
# SECTION 4: DATA EXPLORATION (TASK 1 - Part B)
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Display first 5 rows of data
# 2.2 WHY: To see sample data and verify loading worked
# 2.3 WHEN: Right after loading data
# 2.4 WHERE: EDA phase
# 2.5 HOW: df.head(n) returns first n rows
# 2.6 INTERNAL: Slices DataFrame from top
# 2.7 OUTPUT: Table showing first 5 rows
# ------------------------------------------------------------------------------
print("\n📊 First 5 rows of data:")
print(df.head())

# ------------------------------------------------------------------------------
# 2.1 WHAT: Display column names and data types
# 2.2 WHY: To understand what features we have
# 2.3 WHEN: During EDA
# 2.4 WHERE: After loading data
# 2.5 HOW: df.info() prints summary
# 2.6 INTERNAL: Analyzes each column
# 2.7 OUTPUT: Column names, types, non-null counts
# ------------------------------------------------------------------------------
print("\n📋 Dataset Information:")
print(df.info())

# ------------------------------------------------------------------------------
# 2.1 WHAT: Display statistical summary of numerical columns
# 2.2 WHY: To see min, max, mean, std of each feature
# 2.3 WHEN: During EDA
# 2.4 WHERE: Understanding data distributions
# 2.5 HOW: df.describe() calculates stats
# 2.6 INTERNAL: Computes statistics for each column
# 2.7 OUTPUT: Table with count, mean, std, min, 25%, 50%, 75%, max
# ------------------------------------------------------------------------------
print("\n📈 Statistical Summary:")
print(df.describe())

# ------------------------------------------------------------------------------
# 2.1 WHAT: Check distribution of target variable
# 2.2 WHY: To see if data is balanced (similar Healthy vs Stressed counts)
# 2.3 WHEN: Before training - important for model evaluation
# 2.4 WHERE: EDA phase
# 2.5 HOW: df['column'].value_counts()
# 2.6 INTERNAL: Counts unique values
# 2.7 OUTPUT: Count of each class
# ------------------------------------------------------------------------------
print("\n🏷️ Target Variable Distribution:")
print(df['crop_health_label'].value_counts())


# ==============================================================================
# SECTION 5: VEGETATION INDEX EXPLANATION
# ==============================================================================
# 
# 📘 WHAT ARE VEGETATION INDICES?
# Numbers calculated from different light colors that plants reflect.
# Like taking a plant's temperature, but with light!
#
# 📘 LIST OF INDICES IN OUR DATA:
# 
# | Index | Full Name | Simple Explanation |
# |-------|-----------|-------------------|
# | NDVI | Normalized Difference Vegetation Index | "Greenness score" - higher = healthier |
# | GNDVI | Green NDVI | Uses green light instead of red |
# | SAVI | Soil Adjusted Vegetation Index | NDVI corrected for bare soil |
# | EVI | Enhanced Vegetation Index | Better for dense vegetation |
# | Red Edge | Red Edge Reflectance | Early stress detector |
# | NIR | Near Infrared Reflectance | Invisible light plants reflect |
# | Moisture | Moisture Index | Water content in leaves |
#
# ==============================================================================

print("\n" + "-" * 60)
print("🌿 VEGETATION INDICES EXPLANATION")
print("-" * 60)
print("""
📗 NDVI (Normalized Difference Vegetation Index):
   - Range: -1 to +1
   - Higher values (0.6-0.9) = Healthy green vegetation
   - Lower values (0.2-0.4) = Stressed or sparse vegetation
   - Simple analogy: Like a "health score" for plants

📗 GNDVI (Green NDVI):
   - Similar to NDVI but uses green light
   - Better at detecting chlorophyll content
   - Sensitive to nitrogen deficiency

📗 SAVI (Soil Adjusted Vegetation Index):
   - NDVI corrected for soil brightness
   - Better in areas with visible soil
   - Range: -1 to +1

📗 EVI (Enhanced Vegetation Index):
   - Improved NDVI for dense canopies
   - Less affected by atmospheric conditions
   - Better for forests and high-vegetation areas

📗 Red Edge (red_edge_1, red_edge_2):
   - Detects stress before visible symptoms
   - Early warning system for plant problems
   - Sensitive to chlorophyll changes

📗 NIR Reflectance (nir_reflectance):
   - Near-infrared light reflection
   - Healthy leaves reflect more NIR
   - Related to cell structure

📗 Moisture Index:
   - Water content in vegetation
   - Lower values = water stress
   - Important for irrigation decisions
""")


# ==============================================================================
# SECTION 6: DATA PREPROCESSING (TASK 2 - Part A)
# ==============================================================================

print("\n" + "=" * 60)
print("🧹 DATA PREPROCESSING")
print("=" * 60)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Check for missing values in the dataset
# 2.2 WHY: ML models can't handle missing data properly
# 2.3 WHEN: Before training
# 2.4 WHERE: Preprocessing phase
# 2.5 HOW: df.isnull().sum() counts NaN per column
# 2.6 INTERNAL: Checks each cell for null
# 2.7 OUTPUT: Count of missing values per column
# ------------------------------------------------------------------------------
print("\n🔍 Checking for missing values:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("✅ No missing values found!")
else:
    print(f"⚠️ Found {missing_values.sum()} missing values")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Define feature columns (X) - input variables
# 2.2 WHY: Separate what model will learn FROM (features)
#          from what it will learn TO PREDICT (target)
# 2.3 WHEN: Before model training
# 2.4 WHERE: Preprocessing
# 2.5 HOW: List all column names except target and spatial coords
# 2.6 INTERNAL: Creates list of strings
# 2.7 OUTPUT: List of feature names
# ------------------------------------------------------------------------------
FEATURE_COLUMNS = [
    'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
    'gndvi', 'savi', 'evi',
    'red_edge_1', 'red_edge_2',
    'nir_reflectance', 'soil_brightness',
    'canopy_density', 'moisture_index'
]

print(f"\n📊 Features used for prediction: {len(FEATURE_COLUMNS)}")
for i, col in enumerate(FEATURE_COLUMNS, 1):
    print(f"   {i}. {col}")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Extract feature matrix X
# 2.2 WHY: Create the input data for ML model
# 2.3 WHEN: Preprocessing
# 2.4 WHERE: Before train/test split
# 2.5 HOW: df[list_of_columns] selects those columns
# 2.6 INTERNAL: Creates new DataFrame with selected columns
# 2.7 OUTPUT: DataFrame with only feature columns
# ------------------------------------------------------------------------------
X = df[FEATURE_COLUMNS]
print(f"\n✅ Feature matrix X shape: {X.shape}")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Extract target variable y
# 2.2 WHY: This is what we want the model to predict
# 2.3 WHEN: Preprocessing
# 2.4 WHERE: Before encoding
# 2.5 HOW: df['column_name'] selects single column
# 2.6 INTERNAL: Returns pandas Series
# 2.7 OUTPUT: Series with target values
# ------------------------------------------------------------------------------
y = df['crop_health_label']
print(f"✅ Target variable y shape: {y.shape}")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Encode target labels to numbers
# 2.2 WHY: ML models need numbers (0, 1), not text ("Healthy", "Stressed")
# 2.3 WHEN: After extracting target, before training
# 2.4 WHERE: Preprocessing
# 2.5 HOW: LabelEncoder().fit_transform(y)
# 2.6 INTERNAL: Creates mapping {label: number}, applies to all values
# 2.7 OUTPUT: Array of 0s and 1s
#
# 3.1-3.7 ARGUMENT: y (target variable)
# 3.1 WHAT: The array of labels to encode
# 3.2 WHY: Input for the encoder
# 3.3 WHEN: When calling fit_transform
# 3.4 WHERE: First argument to fit_transform
# 3.5 HOW: Pass the Series or array
# 3.6 INTERNAL: Learns unique values, assigns numbers
# 3.7 OUTPUT: Encoded array
# ------------------------------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n🏷️ Label Encoding:")
print(f"   Classes: {label_encoder.classes_}")
print(f"   Mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")


# ==============================================================================
# SECTION 7: TRAIN-TEST SPLIT
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Split data into training and testing sets
# 2.2 WHY: Use training data to learn, testing data to evaluate
#          This prevents "cheating" - model can't memorize test answers
# 2.3 WHEN: After preprocessing, before training
# 2.4 WHERE: Every ML project
# 2.5 HOW: train_test_split(X, y, test_size=0.2, random_state=42)
# 2.6 INTERNAL: Randomly shuffles, then splits by percentage
# 2.7 OUTPUT: 4 arrays: X_train, X_test, y_train, y_test
#
# 3.1-3.7 ARGUMENT: test_size=0.2
# 3.1 WHAT: Fraction of data for testing
# 3.2 WHY: 20% testing is common, enough to evaluate
# 3.3 WHEN: Always specify this
# 3.4 WHERE: In train_test_split call
# 3.5 HOW: Float between 0 and 1
# 3.6 INTERNAL: Calculates number of test samples
# 3.7 OUTPUT: 80% train, 20% test
#
# 3.1-3.7 ARGUMENT: random_state=42
# 3.1 WHAT: Seed for random number generator
# 3.2 WHY: Makes split reproducible - same split every run
# 3.3 WHEN: Always use for reproducibility
# 3.4 WHERE: In train_test_split call
# 3.5 HOW: Any integer (42 is tradition)
# 3.6 INTERNAL: Initializes random generator
# 3.7 OUTPUT: Same split each time
#
# 3.1-3.7 ARGUMENT: stratify=y_encoded
# 3.1 WHAT: Ensures same class proportions in train/test
# 3.2 WHY: If 70% Healthy in data, both train/test get ~70%
# 3.3 WHEN: For classification with imbalanced classes
# 3.4 WHERE: In train_test_split call
# 3.5 HOW: Pass the target variable
# 3.6 INTERNAL: Splits each class proportionally
# 3.7 OUTPUT: Balanced class distribution
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print("\n✂️ Train-Test Split:")
print(f"   Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")


# ==============================================================================
# SECTION 8: MODEL TRAINING (TASK 2 - Part B)
# ==============================================================================

print("\n" + "=" * 60)
print("🤖 TASK 2: MACHINE LEARNING MODEL")
print("=" * 60)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create Random Forest Classifier model
# 2.2 WHY: Random Forest is accurate, handles many features, resistant to overfitting
#          It creates many decision trees and combines their predictions
# 2.3 WHEN: For classification tasks with tabular data
# 2.4 WHERE: ML modeling phase
# 2.5 HOW: RandomForestClassifier(n_estimators=100, random_state=42)
# 2.6 INTERNAL: Initializes model with hyperparameters
# 2.7 OUTPUT: Untrained model object
#
# 3.1-3.7 ARGUMENT: n_estimators=100
# 3.1 WHAT: Number of trees in the forest
# 3.2 WHY: More trees = more stable predictions, but slower
# 3.3 WHEN: Always specify (default is 100)
# 3.4 WHERE: Model creation
# 3.5 HOW: Integer, typically 100-500
# 3.6 INTERNAL: Creates 100 separate decision trees
# 3.7 OUTPUT: Model with 100 trees
#
# 3.1-3.7 ARGUMENT: random_state=42
# 3.1 WHAT: Seed for reproducibility
# 3.2 WHY: Same model each run
# 3.3 WHEN: Always for reproducibility
# 3.4 WHERE: Model creation
# 3.5 HOW: Any integer
# 3.6 INTERNAL: Controls random tree building
# 3.7 OUTPUT: Reproducible model
#
# 3.1-3.7 ARGUMENT: n_jobs=-1
# 3.1 WHAT: Number of CPU cores to use
# 3.2 WHY: -1 means use ALL cores = faster training
# 3.3 WHEN: For large datasets
# 3.4 WHERE: Model creation
# 3.5 HOW: -1 for all cores, or specific number
# 3.6 INTERNAL: Parallel processing
# 3.7 OUTPUT: Faster training
# ------------------------------------------------------------------------------
print("\n🌲 Creating Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=10,
    min_samples_split=5
)
print("✅ Model created!")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Train the model on training data
# 2.2 WHY: Model learns patterns that distinguish Healthy from Stressed
# 2.3 WHEN: After creating model, before evaluation
# 2.4 WHERE: Training phase
# 2.5 HOW: model.fit(X_train, y_train)
# 2.6 INTERNAL: Builds 100 decision trees, each learns from random data subset
# 2.7 OUTPUT: Trained model (same object, now with learned patterns)
#
# 3.1-3.7 ARGUMENT: X_train
# 3.1 WHAT: Training feature data
# 3.2 WHY: Input data for model to learn from
# 3.3 WHEN: During training
# 3.4 WHERE: First argument to fit()
# 3.5 HOW: DataFrame or 2D array
# 3.6 INTERNAL: Each tree learns from bootstrap sample
# 3.7 OUTPUT: Patterns learned
#
# 3.1-3.7 ARGUMENT: y_train
# 3.1 WHAT: Training target labels
# 3.2 WHY: Correct answers for model to learn
# 3.3 WHEN: During training
# 3.4 WHERE: Second argument to fit()
# 3.5 HOW: 1D array of labels
# 3.6 INTERNAL: Each tree learns to predict these
# 3.7 OUTPUT: Model knows patterns for each class
# ------------------------------------------------------------------------------
print("🎓 Training model...")
model.fit(X_train, y_train)
print("✅ Model trained successfully!")


# ==============================================================================
# SECTION 9: MODEL EVALUATION (TASK 2 - Part C)
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Make predictions on test data
# 2.2 WHY: To see how well model performs on unseen data
# 2.3 WHEN: After training
# 2.4 WHERE: Evaluation phase
# 2.5 HOW: model.predict(X_test)
# 2.6 INTERNAL: Each tree votes, majority wins
# 2.7 OUTPUT: Array of predicted labels (0 or 1)
# ------------------------------------------------------------------------------
print("\n📊 Evaluating model on test data...")
y_pred = model.predict(X_test)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Get prediction probabilities
# 2.2 WHY: Needed for ROC-AUC calculation
#          Shows model's confidence, not just final answer
# 2.3 WHEN: For ROC-AUC and probability-based actions
# 2.4 WHERE: Evaluation phase
# 2.5 HOW: model.predict_proba(X_test)
# 2.6 INTERNAL: Returns probability for each class
# 2.7 OUTPUT: 2D array with probabilities
# ------------------------------------------------------------------------------
y_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------------------------
# 2.1 WHAT: Print classification report with all metrics
# 2.2 WHY: Shows Precision, Recall, F1-Score for each class
# 2.3 WHEN: After predictions
# 2.4 WHERE: Evaluation phase
# 2.5 HOW: classification_report(y_true, y_pred, target_names=classes)
# 2.6 INTERNAL: Computes TP, TN, FP, FN, then metrics
# 2.7 OUTPUT: Formatted table of metrics
# ------------------------------------------------------------------------------
print("\n" + "-" * 60)
print("📋 CLASSIFICATION REPORT")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ------------------------------------------------------------------------------
# 2.1 WHAT: Calculate and print ROC-AUC score
# 2.2 WHY: Single number summarizing model quality (0-1, higher better)
#          ROC-AUC = Area Under Receiver Operating Characteristic Curve
# 2.3 WHEN: For binary classification evaluation
# 2.4 WHERE: Evaluation phase
# 2.5 HOW: roc_auc_score(y_true, y_proba)
# 2.6 INTERNAL: Calculates area under TPR vs FPR curve
# 2.7 OUTPUT: Float between 0 and 1
# ------------------------------------------------------------------------------
roc_auc = roc_auc_score(y_test, y_proba)
print(f"📈 ROC-AUC Score: {roc_auc:.4f}")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create and display confusion matrix
# 2.2 WHY: Shows exactly where model makes mistakes
#          True Positives, True Negatives, False Positives, False Negatives
# 2.3 WHEN: After predictions
# 2.4 WHERE: Evaluation
# 2.5 HOW: confusion_matrix(y_true, y_pred)
# 2.6 INTERNAL: Counts predictions in each category
# 2.7 OUTPUT: 2x2 matrix for binary classification
# ------------------------------------------------------------------------------
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# ==============================================================================
# SECTION 10: FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 WHAT: Extract feature importance scores from model
# 2.2 WHY: To understand which vegetation indices matter most
#          Helps agronomists know what to focus on
# 2.3 WHEN: After training
# 2.4 WHERE: Model interpretation
# 2.5 HOW: model.feature_importances_
# 2.6 INTERNAL: Averages importance across all trees
# 2.7 OUTPUT: Array of importance values (sum to 1)
# ------------------------------------------------------------------------------
print("\n" + "-" * 60)
print("🔑 FEATURE IMPORTANCE")
print("-" * 60)

feature_importance = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# Save feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance for Crop Health Prediction', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Vegetation Index')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
plt.close()
print(f"\n✅ Feature importance plot saved to {OUTPUT_DIR}/feature_importance.png")


# ==============================================================================
# SECTION 11: SPATIAL ANALYSIS & HEATMAP (TASK 3)
# ==============================================================================

print("\n" + "=" * 60)
print("🗺️ TASK 3: SPATIAL ANALYSIS & VISUALIZATION")
print("=" * 60)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Make predictions for entire dataset
# 2.2 WHY: Need predictions for all grid cells to create heatmap
# 2.3 WHEN: After model is trained
# 2.4 WHERE: Spatial analysis phase
# 2.5 HOW: model.predict(X)
# 2.6 INTERNAL: Applies model to all rows
# 2.7 OUTPUT: Predictions for every spatial location
# ------------------------------------------------------------------------------
print("\n🔮 Generating predictions for all grid cells...")
all_predictions = model.predict(X)
all_proba = model.predict_proba(X)[:, 1]

# ------------------------------------------------------------------------------
# 2.1 WHAT: Add predictions to original DataFrame
# 2.2 WHY: Combine predictions with spatial coordinates for mapping
# 2.3 WHEN: Before creating heatmap
# 2.4 WHERE: Spatial analysis
# 2.5 HOW: df['new_column'] = values
# 2.6 INTERNAL: Adds new column to DataFrame
# 2.7 OUTPUT: DataFrame with predictions
# ------------------------------------------------------------------------------
df['predicted_label'] = label_encoder.inverse_transform(all_predictions)
df['stress_probability'] = all_proba

print(f"✅ Predictions generated for {len(df)} grid cells")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Calculate stress statistics
# 2.2 WHY: Summarize overall field health
# 2.3 WHEN: After predictions
# 2.4 WHERE: Spatial analysis
# 2.5 HOW: Count values, calculate percentages
# 2.6 INTERNAL: Simple math on arrays
# 2.7 OUTPUT: Summary statistics
# ------------------------------------------------------------------------------
stress_count = (df['predicted_label'] == 'Stressed').sum()
healthy_count = (df['predicted_label'] == 'Healthy').sum()
total_count = len(df)

print(f"\n📊 Field Health Summary:")
print(f"   Total grid cells: {total_count}")
print(f"   🟢 Healthy: {healthy_count} ({healthy_count/total_count*100:.1f}%)")
print(f"   🔴 Stressed: {stress_count} ({stress_count/total_count*100:.1f}%)")

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create pivot table for heatmap
# 2.2 WHY: Need 2D grid of stress probabilities for visualization
# 2.3 WHEN: Before creating heatmap
# 2.4 WHERE: Spatial analysis
# 2.5 HOW: df.pivot_table(values, index, columns, aggfunc)
# 2.6 INTERNAL: Reorganizes data into grid format
# 2.7 OUTPUT: 2D matrix of values
#
# 3.1-3.7 ARGUMENT: values='stress_probability'
# 3.1 WHAT: Column to use for cell values
# 3.2 WHY: We want to visualize stress probability
# 3.3 WHEN: For pivot table creation
# 3.4 WHERE: First key argument
# 3.5 HOW: Column name as string
# 3.6 INTERNAL: Fills grid with these values
# 3.7 OUTPUT: Probability values in grid
#
# 3.1-3.7 ARGUMENT: index='grid_y'
# 3.1 WHAT: Column for row indices
# 3.2 WHY: Y coordinate becomes vertical axis
# 3.3 WHEN: Pivot table creation
# 3.4 WHERE: Index argument
# 3.5 HOW: Column name as string
# 3.6 INTERNAL: Unique values become row labels
# 3.7 OUTPUT: Rows organized by Y
#
# 3.1-3.7 ARGUMENT: columns='grid_x'
# 3.1 WHAT: Column for column indices
# 3.2 WHY: X coordinate becomes horizontal axis
# 3.3 WHEN: Pivot table creation
# 3.4 WHERE: Columns argument
# 3.5 HOW: Column name as string
# 3.6 INTERNAL: Unique values become column labels
# 3.7 OUTPUT: Columns organized by X
#
# 3.1-3.7 ARGUMENT: aggfunc='mean'
# 3.1 WHAT: How to aggregate if multiple values per cell
# 3.2 WHY: Average probability if duplicates
# 3.3 WHEN: Always specify for clarity
# 3.4 WHERE: Pivot table creation
# 3.5 HOW: String like 'mean', 'sum', 'count'
# 3.6 INTERNAL: Applies function to combine values
# 3.7 OUTPUT: Single value per cell
# ------------------------------------------------------------------------------
print("\n🗺️ Creating stress heatmap...")

heatmap_data = df.pivot_table(
    values='stress_probability',
    index='grid_y',
    columns='grid_x',
    aggfunc='mean'
)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create field stress heatmap visualization
# 2.2 WHY: Visual map helps farmers see stress patterns at a glance
# 2.3 WHEN: After creating pivot table
# 2.4 WHERE: Visualization phase
# 2.5 HOW: seaborn.heatmap(data, cmap, cbar_kws)
# 2.6 INTERNAL: Maps values to colors, draws grid
# 2.7 OUTPUT: Heatmap image
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_data,
    cmap='RdYlGn_r',  # Red=stressed, Green=healthy
    annot=False,
    cbar_kws={'label': 'Stress Probability (0=Healthy, 1=Stressed)'},
    square=True
)
plt.title('Field-Level Crop Stress Heatmap', fontsize=16)
plt.xlabel('Grid X (West → East)')
plt.ylabel('Grid Y (South → North)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stress_heatmap.png'), dpi=150)
plt.close()

print(f"✅ Stress heatmap saved to {OUTPUT_DIR}/stress_heatmap.png")


# ==============================================================================
# SECTION 12: DRONE INSPECTION RECOMMENDATIONS (TASK 4)
# ==============================================================================

print("\n" + "=" * 60)
print("🚁 TASK 4: DRONE & AGRONOMY INTERPRETATION")
print("=" * 60)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Categorize areas by stress severity
# 2.2 WHY: Different stress levels need different drone actions
# 2.3 WHEN: After predictions
# 2.4 WHERE: Recommendation generation
# 2.5 HOW: Use thresholds on stress probability
# 2.6 INTERNAL: Simple comparisons
# 2.7 OUTPUT: Categories for each area
# ------------------------------------------------------------------------------

def categorize_stress(prob):
    """
    Categorize stress level based on probability.
    
    3.1 WHAT: Function to assign category based on probability
    3.2 WHY: Makes recommendation logic reusable
    3.3 WHEN: For each grid cell
    3.4 WHERE: Called in apply()
    3.5 HOW: if-elif-else checks
    3.6 INTERNAL: Simple comparison logic
    3.7 OUTPUT: String category
    
    Parameters:
    -----------
    prob : float
        3.1 WHAT: Stress probability (0-1)
        3.2 WHY: Input for categorization
        3.3 WHEN: For each prediction
        3.4 WHERE: Function argument
        3.5 HOW: Float between 0 and 1
        3.6 INTERNAL: Compared against thresholds
        3.7 OUTPUT: Determines which category
    
    Returns:
    --------
    str : Stress category ('Low', 'Medium', 'High', 'Critical')
    """
    if prob < 0.25:
        return 'Low (Healthy)'
    elif prob < 0.50:
        return 'Medium'
    elif prob < 0.75:
        return 'High'
    else:
        return 'Critical'

df['stress_category'] = df['stress_probability'].apply(categorize_stress)

# Count areas by category
category_counts = df['stress_category'].value_counts()
print("\n📊 Stress Category Distribution:")
print(category_counts)

# Generate recommendations
print("\n" + "-" * 60)
print("🚁 DRONE INSPECTION RECOMMENDATIONS")
print("-" * 60)

recommendations = """
Based on the stress analysis, here are the recommended drone actions:

🔴 CRITICAL STRESS AREAS (Probability > 75%):
   - Immediate detailed inspection required
   - Collect close-up imagery for diagnosis
   - Priority: HIGH - inspect within 24 hours
   - Action: Low-altitude multispectral + RGB capture

🟠 HIGH STRESS AREAS (Probability 50-75%):
   - Schedule inspection within 3 days
   - Monitor for progression
   - Priority: MEDIUM-HIGH
   - Action: Standard multispectral survey

🟡 MEDIUM STRESS AREAS (Probability 25-50%):
   - Include in routine weekly surveys
   - Mark for continued monitoring
   - Priority: MEDIUM
   - Action: Regular monitoring flight

🟢 LOW STRESS / HEALTHY AREAS (Probability < 25%):
   - Standard bi-weekly monitoring
   - No immediate action needed
   - Priority: LOW
   - Action: Routine surveillance only
"""

print(recommendations)

# Save recommendations to file
with open(os.path.join(OUTPUT_DIR, 'drone_recommendations.txt'), 'w', encoding='utf-8') as f:
    f.write("DRONE INSPECTION STRATEGY\n")
    f.write("=" * 50 + "\n\n")
    f.write("STRESS CATEGORY DISTRIBUTION:\n")
    f.write(category_counts.to_string() + "\n\n")
    f.write(recommendations)
    
print(f"✅ Recommendations saved to {OUTPUT_DIR}/drone_recommendations.txt")


# ==============================================================================
# SECTION 13: REFLECTION AND LIMITATIONS (TASK 5)
# ==============================================================================
# 
# 📘 WHAT IS THIS SECTION?
# This section provides a comprehensive reflection on the AI Crop Health
# Monitoring project, discussing:
# - Current approach limitations
# - Real-world data challenges
# - Proposed improvements with implementation examples
# - Quantitative metrics for measuring improvement
# - Future roadmap for production deployment
#
# ==============================================================================

print("\n" + "=" * 60)
print("📝 TASK 5: REFLECTION")
print("==" * 30)
print("🎯 Comprehensive Analysis of Limitations & Proposed Improvements")
print("=" * 60)

# ------------------------------------------------------------------------------
# PART A: DETAILED LIMITATIONS ANALYSIS
# ------------------------------------------------------------------------------

limitations_detailed = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           LIMITATIONS ANALYSIS                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ 📊 LIMITATION 1: DATASET SIZE & REPRESENTATIVENESS                          │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE:
- Our dataset contains a limited number of samples
- May not capture rare stress patterns (e.g., localized pest infestations)
- Class imbalance may affect model performance on minority class

REAL-WORLD IMPACT:
- Model may miss edge cases that occur in < 5% of fields
- False negatives on rare but serious diseases could be costly
- Example: Late blight in potatoes can destroy 100% yield in 2 weeks

QUANTITATIVE CONCERN:
┌────────────────────────┬─────────────────────────────────────────────────┐
│ Metric                 │ Concern                                          │
├────────────────────────┼─────────────────────────────────────────────────┤
│ Sample size            │ May need 10,000+ samples for robust model       │
│ Geographic coverage    │ Currently single field - need regional data     │
│ Stress type coverage   │ Only 2 classes - real world has 10+ conditions  │
└────────────────────────┴─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🕐 LIMITATION 2: TEMPORAL DIMENSION MISSING                                  │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE:
- Single time-point snapshot
- No seasonal variation captured
- Stress progression patterns not modeled

REAL-WORLD IMPACT:
- Cannot detect EARLY stress before visual symptoms
- Miss growth-stage specific problems
- Example: Wheat rust detection window is only 48-72 hours for intervention

WHAT WE'RE MISSING:
┌────────────────────────┬─────────────────────────────────────────────────┐
│ Temporal Feature       │ Agricultural Importance                          │
├────────────────────────┼─────────────────────────────────────────────────┤
│ NDVI trend (3-week)    │ Detects early declining health before critical  │
│ Phenological stage     │ Flowering stress vs vegetative stress differ    │
│ Weather correlation    │ Stress often follows 5-7 days after heat event  │
│ Recovery patterns      │ Some stress is temporary (wilting mid-day)      │
└────────────────────────┴─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🌍 LIMITATION 3: GEOGRAPHIC & ENVIRONMENTAL SPECIFICITY                     │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE:
- Trained on single geographic location
- Specific soil type and climate zone
- May not generalize to other regions

TRANSFER LEARNING CHALLENGE:
- A model trained in Iowa (USA) may fail in Punjab (India)
- Soil color, background, and vegetation vary globally

REGIONAL VARIATIONS EXAMPLE:
┌────────────────────────┬─────────────────────────────────────────────────┐
│ Region                 │ NDVI Threshold for "Healthy"                    │
├────────────────────────┼─────────────────────────────────────────────────┤
│ Midwest USA (Corn)     │ > 0.65 during peak season                       │
│ Tropical Asia (Rice)   │ > 0.55 (different canopy structure)             │
│ Mediterranean (Grape)  │ > 0.45 (sparse training, row crops)             │
│ Arid regions (Wheat)   │ > 0.40 (water stress is normal baseline)        │
└────────────────────────┴─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔬 LIMITATION 4: GROUND TRUTH UNCERTAINTY                                    │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE:
- Labels may have been assigned with subjective criteria
- No standardized stress severity scale used
- Inter-annotator agreement not measured

LABEL QUALITY CONCERNS:
┌────────────────────────┬─────────────────────────────────────────────────┐
│ Issue                  │ Impact                                          │
├────────────────────────┼─────────────────────────────────────────────────┤
│ Subjective labeling    │ Same plant rated "Stressed" by one, "Healthy"  │
│                        │ by another annotator                            │
│ Temporal mismatch      │ Satellite captured at 10am, field visit at 3pm │
│ Spatial resolution     │ 10m pixel may contain mixed health zones        │
│ Annotation bias        │ Annotators may focus on obvious symptoms only  │
└────────────────────────┴─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🌱 LIMITATION 5: SINGLE CROP TYPE ASSUMPTION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE:
- Model assumes homogeneous crop type
- Index thresholds are crop-specific
- Mixed cropping systems not handled

CROP-SPECIFIC CHALLENGES:
┌────────────────────────┬─────────────────────────────────────────────────┐
│ Crop Type              │ Unique Characteristics                          │
├────────────────────────┼─────────────────────────────────────────────────┤
│ Cereals (wheat, rice)  │ Dense canopy, high NDVI in tillering stage     │
│ Legumes (soybean)      │ Yellow leaves in maturity are NORMAL           │
│ Vegetables (tomato)    │ Sparse spacing, soil visible through canopy    │
│ Orchards (apple)       │ Permanent crop with annual variation           │
│ Cotton                 │ NDVI drops during boll opening (normal)        │
└────────────────────────┴─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🛰️ LIMITATION 6: SATELLITE DATA RESOLUTION CONSTRAINTS                     │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE:
- Using derived indices from moderate resolution imagery
- Cloud cover can cause data gaps
- Atmospheric effects not fully corrected

RESOLUTION TRADE-OFFS:
┌────────────────────────┬────────────────┬───────────────────────────────────┐
│ Resolution             │ Use Case       │ Limitation                        │
├────────────────────────┼────────────────┼───────────────────────────────────┤
│ 10m (Sentinel-2)       │ Field-level    │ Cannot detect individual plants   │
│ 3m (PlanetScope)       │ Plot-level     │ Expensive, lower spectral bands   │
│ 0.3m (Drone)           │ Plant-level    │ Limited coverage, weather depend  │
└────────────────────────┴────────────────┴───────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ⚙️ LIMITATION 7: MODEL INTERPRETABILITY                                     │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE:
- Random Forest provides feature importance
- But doesn't explain WHY a specific prediction was made
- Farmers need actionable explanations

EXPLAINABILITY GAP:
- "Your crop is stressed" → NOT ACTIONABLE
- "Your crop is stressed due to low moisture in zone A" → ACTIONABLE
- Need: SHAP values, LIME explanations for individual predictions

┌─────────────────────────────────────────────────────────────────────────────┐
│ 📡 LIMITATION 8: REAL-TIME PROCESSING NOT IMPLEMENTED                       │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT STATE:
- Batch processing only
- No integration with live satellite feeds
- Manual data download required

PRODUCTION REQUIREMENTS:
- Auto-ingest new satellite data every 2-5 days
- Trigger alerts when stress detected
- Update recommendations based on latest imagery
"""

print(limitations_detailed)

# ------------------------------------------------------------------------------
# PART B: PROPOSED IMPROVEMENTS WITH REAL-WORLD DATA
# ------------------------------------------------------------------------------

improvements_detailed = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PROPOSED IMPROVEMENTS WITH REAL-WORLD DATA               ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ 📈 IMPROVEMENT 1: EXPAND DATASET WITH REAL-WORLD DATA SOURCES               │
└─────────────────────────────────────────────────────────────────────────────┘

DATA SOURCES TO INTEGRATE:
┌────────────────────────┬─────────────────────────────────────────────────────┐
│ Source                 │ Data Type & Access                                  │
├────────────────────────┼─────────────────────────────────────────────────────┤
│ Sentinel-2 (ESA)       │ Free 10m multispectral, 5-day revisit              │
│ Landsat (USGS)         │ Free 30m, 40+ years historical                      │
│ PlanetScope            │ Paid 3m daily, excellent for time-series           │
│ MODIS (NASA)           │ Free 250m-1km, daily for NDVI trends               │
│ OpenWeather API        │ Weather data for correlation                        │
│ ISRIC/SoilGrids        │ Global soil property maps                          │
│ FAO GAUL               │ Administrative boundaries for regional models      │
└────────────────────────┴─────────────────────────────────────────────────────┘

IMPLEMENTATION CODE EXAMPLE:
```python
# Example: Loading real Sentinel-2 data via Google Earth Engine
import ee
ee.Initialize()

# Define field boundary
field = ee.Geometry.Rectangle([77.0, 28.0, 77.1, 28.1])

# Get Sentinel-2 Surface Reflectance
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \\
    .filterBounds(field) \\
    .filterDate('2024-01-01', '2024-12-31') \\
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# Calculate NDVI time series
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

s2_ndvi = s2.map(add_ndvi)
```

EXPECTED IMPROVEMENT:
- 10x more samples → Accuracy improvement of 5-10%
- Multi-site data → Better generalization
- Historical data → Weather-stress correlation

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🕐 IMPROVEMENT 2: IMPLEMENT MULTI-TEMPORAL ANALYSIS                          │
└─────────────────────────────────────────────────────────────────────────────┘

APPROACH:
- Collect time-series of 10-20 images per growing season
- Calculate temporal features: trends, volatility, phenological markers
- Detect EARLY stress before it becomes severe

NEW TEMPORAL FEATURES:
┌────────────────────────┬─────────────────────────────────────────────────────┐
│ Feature                │ Calculation                                         │
├────────────────────────┼─────────────────────────────────────────────────────┤
│ ndvi_slope_7day        │ Linear regression slope over past 7 days           │
│ ndvi_deviation         │ Current NDVI - historical average for this date    │
│ recovery_rate          │ Speed of NDVI increase after stress event          │
│ peak_ndvi_date         │ Day of year when NDVI reaches maximum              │
│ growing_degree_days    │ Accumulated heat units from weather data           │
└────────────────────────┴─────────────────────────────────────────────────────┘

IMPLEMENTATION CONCEPT:
```python
# Time-series feature engineering
import pandas as pd

# Sample multi-temporal data structure
temporal_data = {
    'grid_id': ['A1'] * 10,
    'date': pd.date_range('2024-06-01', periods=10, freq='5D'),
    'ndvi': [0.45, 0.48, 0.52, 0.55, 0.53, 0.48, 0.42, 0.38, 0.35, 0.32]
}

df_temp = pd.DataFrame(temporal_data)

# Calculate 7-day NDVI slope (stress indicator)
df_temp['ndvi_slope'] = df_temp['ndvi'].rolling(3).apply(
    lambda x: (x.iloc[-1] - x.iloc[0]) / 2
)

# Negative slope = EARLY WARNING of stress developing
# Trigger alert when slope < -0.02 for 2 consecutive periods
```

EXPECTED IMPROVEMENT:
- Early detection 2-3 weeks before visual symptoms
- 90%+ accuracy for stress progression prediction
- Reduced false positives from temporary stress

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🌦️ IMPROVEMENT 3: INTEGRATE WEATHER & ENVIRONMENTAL DATA                    │
└─────────────────────────────────────────────────────────────────────────────┘

WEATHER FEATURES TO ADD:
┌────────────────────────┬─────────────────────────────────────────────────────┐
│ Feature                │ Agricultural Significance                           │
├────────────────────────┼─────────────────────────────────────────────────────┤
│ temp_max_7day          │ Heat stress accumulation                            │
│ precip_total_7day      │ Recent rainfall (drought/flood risk)               │
│ vapor_pressure_deficit │ Evapotranspiration stress indicator                │
│ humidity_min           │ Disease risk (fungal infections)                   │
│ wind_speed_max         │ Physical damage / lodging risk                     │
│ frost_nights_count     │ Cold damage in sensitive growth stages             │
└────────────────────────┴─────────────────────────────────────────────────────┘

REAL-WORLD DATA SOURCE - OpenWeather API:
```python
import requests
import pandas as pd

API_KEY = 'your_api_key'
LAT, LON = 28.6139, 77.2090  # Example: Delhi

# Get historical weather for past 7 days
url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={LAT}&lon={LON}&dt={{timestamp}}&appid={API_KEY}'

# Create weather features
weather_features = {
    'temp_max_7day': 38.5,       # Celsius
    'precip_total_7day': 5.2,    # mm
    'humidity_avg': 65,          # %
}

# Combine with vegetation indices
# When temp_max_7day > 35 and precip_7day < 10mm → HIGH STRESS RISK
```

EXPECTED IMPROVEMENT:
- Better explanation of stress causes
- Predict stress 5-7 days in advance
- Reduce false positives by 30%

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🤖 IMPROVEMENT 4: ADVANCED MACHINE LEARNING MODELS                           │
└─────────────────────────────────────────────────────────────────────────────┘

MODEL COMPARISON ROADMAP:
┌─────────────────────────┬────────────┬───────────┬─────────────────────────┐
│ Model                   │ Complexity │ Speed     │ Best Use Case           │
├─────────────────────────┼────────────┼───────────┼─────────────────────────┤
│ Random Forest (current) │ Medium     │ Fast      │ Baseline, explainable   │
│ XGBoost                 │ Medium     │ Fast      │ Higher accuracy         │
│ LightGBM                │ Medium     │ Very Fast │ Large datasets          │
│ CatBoost                │ Medium     │ Fast      │ Categorical features    │
│ CNN (image input)       │ High       │ Slow      │ Raw satellite imagery   │
│ LSTM (time series)      │ High       │ Medium    │ Temporal predictions    │
│ Transformer             │ Very High  │ Slow      │ State-of-the-art        │
└─────────────────────────┴────────────┴───────────┴─────────────────────────┘

IMPLEMENTATION - XGBoost COMPARISON:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# XGBoost model for comparison
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    random_state=42
)

# Compare with Random Forest
rf_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')

print(f"Random Forest ROC-AUC: {np.mean(rf_scores):.4f}")
print(f"XGBoost ROC-AUC: {np.mean(xgb_scores):.4f}")
```

EXPECTED IMPROVEMENT:
- 3-5% accuracy improvement with XGBoost
- 10-15% improvement with ensemble methods
- CNNs could enable raw image input (no feature engineering)

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔍 IMPROVEMENT 5: EXPLAINABLE AI (XAI) FOR FARMERS                           │
└─────────────────────────────────────────────────────────────────────────────┘

CURRENT GAP:
- Model says "Stressed" but doesn't explain WHY
- Farmers need actionable insights

SOLUTION - SHAP VALUES:
```python
import shap

# Create SHAP explainer for Random Forest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For a single prediction, show which features contributed
# Example output: "This area is stressed because:
#   - Moisture Index is 0.15 (critically low) - contributing +0.35
#   - NDVI is 0.28 (below healthy threshold) - contributing +0.25
#   - SAVI is 0.31 (low vegetation) - contributing +0.15

# Visualize
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLUMNS)
```

FARMER-FRIENDLY OUTPUT:
```
📍 Grid Location: X=15, Y=23
🔴 Prediction: STRESSED (89% confidence)

📊 STRESS CAUSES:
┌────────────────────────┬────────────────────────────────────────────────┐
│ Factor                 │ Explanation                                     │
├────────────────────────┼────────────────────────────────────────────────┤
│ 💧 Moisture (0.15)     │ CRITICAL: 70% below healthy level              │
│ 🌿 NDVI (0.28)         │ WARNING: Vegetation vigor declining            │
│ ☀️ Recent Temperature  │ Heat wave (38°C for 5 days) preceded stress   │
└────────────────────────┴────────────────────────────────────────────────┘

🎯 RECOMMENDED ACTION: Immediate irrigation (20mm)
```

EXPECTED IMPROVEMENT:
- Farmer trust and adoption increased
- Precision interventions (right action at right place)
- Reduced chemical usage through targeted application

┌─────────────────────────────────────────────────────────────────────────────┐
│ ✅ IMPROVEMENT 6: PRODUCTION-GRADE VALIDATION FRAMEWORK                      │
└─────────────────────────────────────────────────────────────────────────────┘

VALIDATION REQUIREMENTS FOR REAL DEPLOYMENT:
┌────────────────────────┬─────────────────────────────────────────────────────┐
│ Validation Type        │ Implementation                                      │
├────────────────────────┼─────────────────────────────────────────────────────┤
│ Spatial cross-val      │ Train on fields A,B,C → Test on field D            │
│ Temporal cross-val     │ Train on 2022-2023 → Test on 2024                  │
│ Field verification     │ Ground truth from agronomist field visits          │
│ A/B testing            │ Compare AI vs traditional disease detection        │
│ Continuous monitoring  │ Track model performance over seasons              │
└────────────────────────┴─────────────────────────────────────────────────────┘

METRICS DASHBOARD CONCEPT:
```python
production_metrics = {
    'accuracy_overall': 0.85,
    'precision_stressed': 0.78,
    'recall_stressed': 0.92,  # Critical - don't miss stressed areas
    'false_positive_rate': 0.15,
    'early_detection_rate': 0.75,
    'farmer_adoption_rate': 0.45,
    'intervention_success_rate': 0.82
}
```

┌─────────────────────────────────────────────────────────────────────────────┐
│ 📡 IMPROVEMENT 7: REAL-TIME PROCESSING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

PRODUCTION ARCHITECTURE:
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│ Satellite   │────▶│ Cloud        │────▶│ ML Pipeline │────▶│ Alert       │
│ Data Feed   │     │ Processing   │     │ (Model)     │     │ System      │
│ (Sentinel)  │     │ (GEE/AWS)    │     │             │     │ (SMS/App)   │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │                    │
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
   Auto-ingest         Preprocess          Predict            Farmer gets
   every 5 days        NDVI, EVI           stress             SMS alert
```

IMPLEMENTATION STEPS:
1. Google Earth Engine for satellite data processing
2. AWS/GCP for model hosting
3. API backend for mobile app
4. SMS/WhatsApp integration for farmer alerts
"""

print(improvements_detailed)

# ------------------------------------------------------------------------------
# PART C: IMPLEMENTATION ROADMAP
# ------------------------------------------------------------------------------

roadmap = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         IMPLEMENTATION ROADMAP                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 1: SHORT-TERM (1-3 MONTHS)                    │
└─────────────────────────────────────────────────────────────────────────────┘

✅ COMPLETED:
[x] Random Forest model with vegetation indices
[x] Stress heatmap visualization
[x] Drone inspection recommendations
[x] Basic feature importance analysis

🔄 NEXT STEPS:
[ ] Add XGBoost/LightGBM model comparison
[ ] Implement SHAP explanations
[ ] Integrate weather data API
[ ] Create validation sampling framework

ESTIMATED EFFORT: 40-60 hours

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 2: MEDIUM-TERM (3-6 MONTHS)                   │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 GOALS:
[ ] Multi-temporal analysis with 10+ time steps
[ ] Expand to 5+ fields for regional model
[ ] Field validation with agronomist team
[ ] Mobile app prototype for farmer alerts

KEY MILESTONES:
- Month 4: Multi-field model trained and validated
- Month 5: Time-series features integrated
- Month 6: Farmer pilot program launch

ESTIMATED EFFORT: 200-300 hours

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 3: LONG-TERM (6-12 MONTHS)                    │
└─────────────────────────────────────────────────────────────────────────────┘

🚀 PRODUCTION DEPLOYMENT:
[ ] Real-time satellite data pipeline
[ ] Multi-crop, multi-region models
[ ] Deep learning integration (CNN/LSTM)
[ ] Commercial API for agribusiness

SUCCESS METRICS:
┌────────────────────────┬─────────────────────────────────────────────────────┐
│ KPI                    │ Target                                              │
├────────────────────────┼─────────────────────────────────────────────────────┤
│ Model accuracy         │ > 90% on held-out test set                         │
│ Early detection rate   │ > 80% (2+ weeks before visual symptoms)            │
│ Farmer adoption        │ > 500 farmers using the system                     │
│ Yield improvement      │ > 10% compared to baseline                         │
│ False positive rate    │ < 10% to maintain farmer trust                     │
└────────────────────────┴─────────────────────────────────────────────────────┘
"""

print(roadmap)

# ------------------------------------------------------------------------------
# PART D: QUANTITATIVE METRICS FOR IMPROVEMENT TRACKING
# ------------------------------------------------------------------------------

print("\n" + "-" * 60)
print("📊 QUANTITATIVE METRICS FOR IMPROVEMENT TRACKING")
print("-" * 60)

# Create metrics dataframe for improvement tracking
improvement_metrics = pd.DataFrame({
    'Improvement Area': [
        'Expanded Dataset (10x)',
        'Multi-temporal Features',
        'Weather Integration',
        'XGBoost Model',
        'Ensemble Methods',
        'SHAP Explanations',
        'Field Validation'
    ],
    'Current ROC-AUC': [roc_auc, roc_auc, roc_auc, roc_auc, roc_auc, roc_auc, roc_auc],
    'Expected ROC-AUC': [
        min(roc_auc + 0.08, 0.98),  # More data
        min(roc_auc + 0.10, 0.98),  # Temporal features
        min(roc_auc + 0.05, 0.98),  # Weather
        min(roc_auc + 0.05, 0.98),  # XGBoost
        min(roc_auc + 0.10, 0.99),  # Ensemble
        roc_auc,                      # SHAP doesn't improve accuracy
        min(roc_auc + 0.15, 0.99),  # Validated + cleaned labels
    ],
    'Implementation Effort': [
        'Medium (40 hrs)',
        'High (80 hrs)',
        'Low (20 hrs)',
        'Low (10 hrs)',
        'Medium (40 hrs)',
        'Low (15 hrs)',
        'High (100 hrs)'
    ],
    'Business Impact': [
        'High - Better generalization',
        'Very High - Early detection',
        'Medium - Stress explanation',
        'Medium - Better accuracy',
        'High - Robust predictions',
        'Very High - Farmer trust',
        'Critical - Production ready'
    ]
})

print("\n📈 IMPROVEMENT IMPACT MATRIX:")
print(improvement_metrics.to_string(index=False))

# Save comprehensive reflection to file
reflection_comprehensive = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           AI CROP HEALTH MONITORING - COMPREHENSIVE REFLECTION               ║
║                         Task 5: Limitations & Improvements                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
MODEL PERFORMANCE: ROC-AUC = {roc_auc:.4f}

{limitations_detailed}

{improvements_detailed}

{roadmap}

╔══════════════════════════════════════════════════════════════════════════════╗
║                              FINAL SUMMARY                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY TAKEAWAYS:
==============
1. CURRENT MODEL STRENGTHS:
   - Good baseline performance with Random Forest
   - Feature importance provides interpretability
   - Successfully identifies stressed areas

2. CRITICAL GAPS TO ADDRESS:
   - Temporal dimension missing (early detection)
   - Single field limits generalization
   - No real-time processing pipeline

3. HIGHEST IMPACT IMPROVEMENTS:
   a) Multi-temporal analysis → Early stress detection
   b) Weather integration → Stress cause explanation
   c) Field validation → Production-ready confidence

4. RECOMMENDED NEXT ACTIONS:
   - Priority 1: Integrate 5+ time steps of imagery
   - Priority 2: Add weather API data
   - Priority 3: Implement SHAP for farmer explanations
   - Priority 4: Expand to multiple fields for validation

CONCLUSION:
===========
This project demonstrates the potential of AI for crop health monitoring.
However, production deployment requires addressing temporal, spatial, and
validation gaps. The proposed improvements can potentially increase 
accuracy from {roc_auc:.4f} to >0.95 while providing explainable, 
actionable recommendations to farmers.

The key to success is not just model accuracy, but:
- TIMELINESS: Detecting stress before yield loss
- EXPLAINABILITY: Telling farmers WHY and WHAT to do
- TRUST: Validated predictions that farmers can rely on
"""

with open(os.path.join(OUTPUT_DIR, 'reflection_comprehensive.txt'), 'w', encoding='utf-8') as f:
    f.write(reflection_comprehensive)

print(f"\n✅ Comprehensive reflection saved to {OUTPUT_DIR}/reflection_comprehensive.txt")

# Also save improvement metrics to CSV for tracking
improvement_metrics.to_csv(os.path.join(OUTPUT_DIR, 'improvement_metrics.csv'), index=False)
print(f"✅ Improvement metrics saved to {OUTPUT_DIR}/improvement_metrics.csv")


# ==============================================================================
# SECTION 14: SAVE FINAL OUTPUTS
# ==============================================================================

print("\n" + "=" * 60)
print(" SAVING FINAL OUTPUTS")
print("=" * 60)

# Save predictions to CSV
output_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
df.to_csv(output_path, index=False)
print(f" Predictions saved to {output_path}")

# Create confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print(f" Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")


# ==============================================================================
# SECTION 16: NEXT STEP 1 & 2 - PRIORITY DRONE INSPECTION LIST
# ==============================================================================
# 
# 📘 WHAT IS THIS?
# Automates Steps 1 & 2 of NEXT STEPS:
# - Identifies areas that need immediate drone inspection
# - Creates prioritized list sorted by stress probability
#
# ==============================================================================

print("\n" + "=" * 60)
print(" NEXT STEP IMPLEMENTATION: PRIORITY DRONE INSPECTION LIST")
print("=" * 60)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Filter high-priority areas for drone inspection
# 2.2 WHY: Critical and High stress areas need immediate attention
#          Saves time by focusing drone flights on problem areas
# 2.3 WHEN: After stress categorization
# 2.4 WHERE: Operational planning phase
# 2.5 HOW: Filter DataFrame by stress category
# 2.6 INTERNAL: Boolean indexing on DataFrame
# 2.7 OUTPUT: DataFrame with only priority areas
# ------------------------------------------------------------------------------

# Get Critical and High stress areas
critical_areas = df[df['stress_category'] == 'Critical'][
    ['grid_x', 'grid_y', 'stress_probability', 'ndvi_mean', 'moisture_index', 'stress_category']
].copy()
high_stress_areas = df[df['stress_category'] == 'High'][
    ['grid_x', 'grid_y', 'stress_probability', 'ndvi_mean', 'moisture_index', 'stress_category']
].copy()

# Combine and sort by stress probability (highest first)
priority_inspection = pd.concat([critical_areas, high_stress_areas])
priority_inspection = priority_inspection.sort_values('stress_probability', ascending=False)

# Add priority ranking
priority_inspection['priority_rank'] = range(1, len(priority_inspection) + 1)

print(f"\n Priority Inspection Summary:")
print(f"    Critical areas: {len(critical_areas)}")
print(f"    High stress areas: {len(high_stress_areas)}")
print(f"    Total priority locations: {len(priority_inspection)}")

if len(priority_inspection) > 0:
    print(f"\n TOP 10 PRIORITY INSPECTION LOCATIONS:")
    print("-" * 60)
    print(priority_inspection.head(10).to_string(index=False))
    
    # Save priority inspection list to CSV
    priority_csv_path = os.path.join(OUTPUT_DIR, 'priority_inspection_list.csv')
    priority_inspection.to_csv(priority_csv_path, index=False)
    print(f"\n Priority inspection list saved to {priority_csv_path}")
else:
    print("\n No high-priority areas found! Field is in good health.")


# ==============================================================================
# SECTION 17: NEXT STEP 3 - FIELD VALIDATION SAMPLING PLAN
# ==============================================================================
# 
# 📘 WHAT IS THIS?
# Creates a sampling plan for ground truth validation:
# - Stratified random sample from each stress category
# - Field team can visit these locations to verify predictions
# - Helps measure model accuracy in real-world conditions
#
# ==============================================================================

print("\n" + "=" * 60)
print("📋 NEXT STEP IMPLEMENTATION: FIELD VALIDATION SAMPLING PLAN")
print("=" * 60)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Create stratified sample for field validation
# 2.2 WHY: Need ground truth data to validate model predictions
#          Stratified sampling ensures all categories are represented
# 2.3 WHEN: Before field deployment
# 2.4 WHERE: Model validation phase
# 2.5 HOW: Sample fixed percentage from each stress category
# 2.6 INTERNAL: Group by category, sample from each group
# 2.7 OUTPUT: DataFrame with sample points for field validation
#
# 3.1-3.7 ARGUMENT: n (sample size per category)
# 3.1 WHAT: Number of samples to select
# 3.2 WHY: Balance between coverage and field work effort
# 3.3 WHEN: Sampling
# 3.4 WHERE: sample() method argument
# 3.5 HOW: Integer, typically 10-20 per category
# 3.6 INTERNAL: Random selection without replacement
# 3.7 OUTPUT: Specified number of samples
# ------------------------------------------------------------------------------

validation_samples = pd.DataFrame()

print("\n🎯 Stratified Sampling Plan (10 samples per category or max available):\n")

for category in df['stress_category'].unique():
    category_data = df[df['stress_category'] == category].copy()
    
    # Sample 10 points or maximum available (whichever is smaller)
    sample_size = min(10, len(category_data))
    
    if sample_size > 0:
        sampled = category_data.sample(n=sample_size, random_state=42)
        sampled['validation_purpose'] = 'Ground truth verification'
        validation_samples = pd.concat([validation_samples, sampled])
        print(f"   {category}: {sample_size} samples selected (from {len(category_data)} total)")

# Select relevant columns for field team
validation_columns = ['grid_x', 'grid_y', 'stress_category', 'stress_probability', 
                      'ndvi_mean', 'moisture_index', 'predicted_label', 'validation_purpose']
validation_export = validation_samples[validation_columns].copy()

# Add field validation columns (to be filled by field team)
validation_export['actual_health_status'] = ''  # Field team fills this
validation_export['notes'] = ''                  # Field observations
validation_export['photo_id'] = ''               # Reference to field photos

print(f"\n📍 Total validation samples: {len(validation_export)}")

# Save validation sampling plan
validation_csv_path = os.path.join(OUTPUT_DIR, 'field_validation_samples.csv')
validation_export.to_csv(validation_csv_path, index=False)
print(f" Field validation samples saved to {validation_csv_path}")

# Create field validation instructions
validation_instructions = """
FIELD VALIDATION GUIDE
======================

PURPOSE:
This file contains sample locations for ground truth validation.
Visit each location and record the actual crop health status.

INSTRUCTIONS FOR FIELD TEAM:
1. Use GPS to navigate to each (grid_x, grid_y) location
2. Visually assess the crop health
3. Fill in the 'actual_health_status' column:
   - 'Healthy' if plants appear vigorous and green
   - 'Stressed' if plants show wilting, discoloration, or disease

4. Add notes about what you observe (pest damage, water stress, etc.)
5. Take photos and record the photo ID

VALIDATION METRICS:
After completing field work, compare:
- Model's 'predicted_label' vs your 'actual_health_status'
- Calculate: Accuracy = (matching predictions / total samples) × 100
"""

validation_guide_path = os.path.join(OUTPUT_DIR, 'field_validation_guide.txt')
with open(validation_guide_path, 'w', encoding='utf-8') as f:
    f.write(validation_instructions)
print(f" Field validation guide saved to {validation_guide_path}")


# ==============================================================================
# SECTION 18: NEXT STEP 4 - INTERVENTION STRATEGY PLANNING
# ==============================================================================
# 
# 📘 WHAT IS THIS?
# Generates specific intervention recommendations based on:
# - Stress probability level
# - Vegetation index values (NDVI, moisture, etc.)
# - Creates actionable plan for farm managers
#
# ==============================================================================

print("\n" + "=" * 60)
print(" NEXT STEP IMPLEMENTATION: INTERVENTION STRATEGY PLANNING")
print("=" * 60)

# ------------------------------------------------------------------------------
# 2.1 WHAT: Function to suggest intervention based on feature values
# 2.2 WHY: Different stress causes need different solutions
#          Low moisture → irrigation, Low NDVI → fertilizer, etc.
# 2.3 WHEN: For each stressed area
# 2.4 WHERE: Intervention planning
# 2.5 HOW: if-elif logic based on thresholds
# 2.6 INTERNAL: Compares values against agronomic thresholds
# 2.7 OUTPUT: String with recommended action
# ------------------------------------------------------------------------------

def suggest_intervention(row):
    """
    Suggest intervention strategy based on vegetation indices.
    
    3.1 WHAT: Analyzes row data to recommend action
    3.2 WHY: Automates intervention planning
    3.3 WHEN: Called for each stressed area
    3.4 WHERE: apply() on DataFrame
    3.5 HOW: Pass row, returns recommendation string
    3.6 INTERNAL: Threshold-based decision logic
    3.7 OUTPUT: Intervention recommendation
    
    Parameters:
    -----------
    row : pd.Series
        3.1 WHAT: Single row from DataFrame
        3.2 WHY: Contains all feature values for one location
        3.3 WHEN: During apply()
        3.4 WHERE: Function argument
        3.5 HOW: Passed automatically by apply()
        3.6 INTERNAL: Access columns via row['column_name']
        3.7 OUTPUT: Used to determine recommendation
    """
    recommendations = []
    
    # Check moisture stress
    if row['moisture_index'] < 0.3:
        recommendations.append(' Irrigation needed - Low moisture detected')
    
    # Check NDVI (vegetation health)
    if row['ndvi_mean'] < 0.4:
        recommendations.append(' Fertilizer application - Low vegetation vigor')
    
    # Check canopy density
    if 'canopy_density' in row and row['canopy_density'] < 0.5:
        recommendations.append(' Check for pest/disease - Thin canopy detected')
    
    # If still stressed but no specific issue found
    if len(recommendations) == 0 and row['stress_probability'] > 0.5:
        recommendations.append(' Detailed inspection needed - Cause unclear from indices')
    
    return ' | '.join(recommendations) if recommendations else ' Monitor only'


# Apply intervention suggestions to stressed areas
stressed_areas = df[df['predicted_label'] == 'Stressed'].copy()

if len(stressed_areas) > 0:
    stressed_areas['recommended_intervention'] = stressed_areas.apply(suggest_intervention, axis=1)
    
    # Create intervention summary
    intervention_summary = stressed_areas.groupby('stress_category').agg({
        'grid_x': 'count',
        'stress_probability': 'mean',
        'ndvi_mean': 'mean',
        'moisture_index': 'mean'
    }).rename(columns={'grid_x': 'area_count'})
    
    print("\n INTERVENTION SUMMARY BY STRESS CATEGORY:")
    print("-" * 60)
    print(intervention_summary.to_string())
    
    # Count intervention types
    print("\n RECOMMENDED INTERVENTIONS:")
    print("-" * 60)
    
    irrigation_needed = stressed_areas['recommended_intervention'].str.contains('Irrigation').sum()
    fertilizer_needed = stressed_areas['recommended_intervention'].str.contains('Fertilizer').sum()
    inspection_needed = stressed_areas['recommended_intervention'].str.contains('pest|inspection', case=False).sum()
    
    print(f"    Areas needing irrigation: {irrigation_needed}")
    print(f"    Areas needing fertilizer: {fertilizer_needed}")
    print(f"    Areas needing detailed inspection: {inspection_needed}")
    
    # Save intervention plan
    intervention_columns = ['grid_x', 'grid_y', 'stress_category', 'stress_probability',
                           'ndvi_mean', 'moisture_index', 'recommended_intervention']
    intervention_plan = stressed_areas[intervention_columns].sort_values(
        'stress_probability', ascending=False
    )
    
    intervention_csv_path = os.path.join(OUTPUT_DIR, 'intervention_plan.csv')
    intervention_plan.to_csv(intervention_csv_path, index=False)
    print(f"\n Intervention plan saved to {intervention_csv_path}")
    
    # Show top 10 intervention actions
    print("\n TOP 10 PRIORITY INTERVENTIONS:")
    print("-" * 60)
    print(intervention_plan[['grid_x', 'grid_y', 'stress_category', 'recommended_intervention']].head(10).to_string(index=False))

else:
    print("\n No stressed areas found! No interventions needed.")


# ==============================================================================
# SECTION 19: NEXT STEPS SUMMARY REPORT
# ==============================================================================

print("\n" + "=" * 60)
print("📋 NEXT STEPS IMPLEMENTATION COMPLETE!")
print("=" * 60)

next_steps_report = f"""
🎯 NEXT STEPS IMPLEMENTATION REPORT
{'=' * 50}

✅ STEP 1 & 2: DRONE INSPECTION PRIORITIZATION
   - Priority inspection list generated
   - {len(priority_inspection) if len(priority_inspection) > 0 else 0} locations flagged for immediate drone survey
   - File: priority_inspection_list.csv

✅ STEP 3: FIELD VALIDATION SAMPLING
   - Stratified sample created for ground truth validation
   - {len(validation_export)} sample locations selected
   - Field guide included for validation team
   - Files: field_validation_samples.csv, field_validation_guide.txt

✅ STEP 4: INTERVENTION STRATEGY
   - Automated recommendations based on vegetation indices
   - {len(stressed_areas) if len(stressed_areas) > 0 else 0} stressed areas analyzed
   - Specific actions assigned (irrigation, fertilizer, inspection)
   - File: intervention_plan.csv

📁 NEW OUTPUT FILES:
   7. priority_inspection_list.csv
   8. field_validation_samples.csv
   9. field_validation_guide.txt
   10. intervention_plan.csv
"""

print(next_steps_report)

# Save next steps report
with open(os.path.join(OUTPUT_DIR, 'next_steps_report.txt'), 'w', encoding='utf-8') as f:
    f.write(next_steps_report)
print(f"✅ Next steps report saved to {OUTPUT_DIR}/next_steps_report.txt")


# ==============================================================================
# SECTION 20: FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("🏆 PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"""
📁 OUTPUT FILES GENERATED:
   1. {OUTPUT_DIR}/feature_importance.png
   2. {OUTPUT_DIR}/stress_heatmap.png
   3. {OUTPUT_DIR}/confusion_matrix.png
   4. {OUTPUT_DIR}/drone_recommendations.txt
   5. {OUTPUT_DIR}/reflection.txt
   6. {OUTPUT_DIR}/predictions.csv
   7. {OUTPUT_DIR}/priority_inspection_list.csv
   8. {OUTPUT_DIR}/field_validation_samples.csv
   9. {OUTPUT_DIR}/field_validation_guide.txt
   10. {OUTPUT_DIR}/intervention_plan.csv
   11. {OUTPUT_DIR}/next_steps_report.txt

📊 MODEL PERFORMANCE:
   - ROC-AUC Score: {roc_auc:.4f}

🌾 FIELD HEALTH STATUS:
   - 🟢 Healthy: {healthy_count} areas ({healthy_count/total_count*100:.1f}%)
   - 🔴 Stressed: {stress_count} areas ({stress_count/total_count*100:.1f}%)

🎯 NEXT STEPS IMPLEMENTED:
   ✅ Step 1 & 2: Priority drone inspection list created
   ✅ Step 3: Field validation samples exported
   ✅ Step 4: Intervention strategies planned
""")

print("=" * 60)
print("🌱 Thank you for using AI Crop Health Monitoring!")
print("=" * 60)
