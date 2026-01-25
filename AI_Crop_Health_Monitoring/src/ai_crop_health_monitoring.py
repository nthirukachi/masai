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
with open(os.path.join(OUTPUT_DIR, 'drone_recommendations.txt'), 'w') as f:
    f.write("DRONE INSPECTION STRATEGY\n")
    f.write("=" * 50 + "\n\n")
    f.write("STRESS CATEGORY DISTRIBUTION:\n")
    f.write(category_counts.to_string() + "\n\n")
    f.write(recommendations)
    
print(f"✅ Recommendations saved to {OUTPUT_DIR}/drone_recommendations.txt")


# ==============================================================================
# SECTION 13: REFLECTION AND LIMITATIONS (TASK 5)
# ==============================================================================

print("\n" + "=" * 60)
print("📝 TASK 5: REFLECTION")
print("=" * 60)

reflection = """
LIMITATIONS OF THIS APPROACH:
-----------------------------
1. 📊 Dataset Size:
   - Small dataset may not capture all stress patterns
   - More data would improve model generalization

2. 🕐 Temporal Aspects:
   - Single time snapshot - no seasonal variation
   - Stress changes over time not captured

3. 🌍 Geographic Specificity:
   - Model trained on one field/region
   - May not transfer to different climates or crops

4. 🔬 Ground Truth:
   - Labels may have some uncertainty
   - Field validation not included

5. 🌱 Crop Type:
   - Single crop type assumed
   - Different crops have different index thresholds

PROPOSED IMPROVEMENTS:
----------------------
1. 📈 More Data:
   - Collect data across multiple seasons
   - Include multiple fields and crop types

2. 🛰️ Multi-temporal Analysis:
   - Track changes over time
   - Detect stress progression

3. 🔍 Additional Features:
   - Weather data integration
   - Soil sensor data
   - Historical yield data

4. 🤖 Advanced Models:
   - Try XGBoost, LightGBM for comparison
   - Deep learning for texture features
   - Ensemble methods for robustness

5. ✅ Validation:
   - Field verification of predictions
   - Agronomist expert review
   - A/B testing of recommendations
"""

print(reflection)

# Save reflection to file
with open(os.path.join(OUTPUT_DIR, 'reflection.txt'), 'w') as f:
    f.write("PROJECT REFLECTION AND LIMITATIONS\n")
    f.write("=" * 50 + "\n")
    f.write(reflection)

print(f"✅ Reflection saved to {OUTPUT_DIR}/reflection.txt")


# ==============================================================================
# SECTION 14: SAVE FINAL OUTPUTS
# ==============================================================================

print("\n" + "=" * 60)
print("💾 SAVING FINAL OUTPUTS")
print("=" * 60)

# Save predictions to CSV
output_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")

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
print(f"✅ Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")


# ==============================================================================
# SECTION 15: FINAL SUMMARY
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

📊 MODEL PERFORMANCE:
   - ROC-AUC Score: {roc_auc:.4f}

🌾 FIELD HEALTH STATUS:
   - 🟢 Healthy: {healthy_count} areas ({healthy_count/total_count*100:.1f}%)
   - 🔴 Stressed: {stress_count} areas ({stress_count/total_count*100:.1f}%)

🎯 NEXT STEPS:
   1. Review the stress heatmap
   2. Prioritize drone inspections based on recommendations
   3. Collect ground truth data for validation
   4. Plan intervention strategies for stressed areas
""")

print("=" * 60)
print("🌱 Thank you for using AI Crop Health Monitoring!")
print("=" * 60)
