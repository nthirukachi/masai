"""
===============================================================================
K-Means Feature Augmentation + Perceptron Cross-Validation
===============================================================================

PROJECT: MQ5_KMeans_Perceptron_CrossVal
DATASET: scikit-learn's load_wine (178 samples, 13 features, 3 classes → binary)

OBJECTIVE:
    Determine if augmenting features with K-Means cluster information
    (one-hot membership + centroid distances) improves Perceptron classification.

APPROACH:
    1. Load Wine dataset and create binary labels (class 0 = positive)
    2. Use stratified 5-fold cross-validation
    3. For each fold:
       - Standardize features (fit on train only)
       - Fit K-Means k=4 (on train only) → prevents data leakage
       - Augment features: original + one-hot clusters + centroid distances
       - Train baseline Perceptron on original features
       - Train enhanced Perceptron on augmented features
    4. Compare metrics: Accuracy, F1-score, Average Precision
    5. Generate executive summary with statistical significance

REAL-LIFE ANALOGY:
    Think of wine classification like sorting students into groups:
    - Original features = grades in 13 subjects
    - K-Means = finding 4 natural "types" of students (nerds, athletes, artists, etc.)
    - Feature augmentation = telling the classifier "this student is most like a nerd,
      but also somewhat artistic" (distances to each group center)
    - The enhanced classifier has MORE information to make better decisions!

===============================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 NumPy - Numerical Python
# -----------------------------------------------------------------------------
# WHAT: NumPy is the fundamental package for scientific computing in Python.
# WHY: We need it for:
#      - Array operations (faster than Python lists)
#      - Mathematical functions (mean, std, sqrt)
#      - Distance calculations (Euclidean distance to centroids)
# WHEN: Use NumPy when working with numbers, matrices, or scientific data.
# WHERE: Used everywhere in data science and machine learning.
# HOW: Import as 'np' (standard convention).
# INTERNALLY: NumPy uses C code for speed (100x faster than Python loops).
import numpy as np

# -----------------------------------------------------------------------------
# 1.2 Pandas - Data Analysis Library
# -----------------------------------------------------------------------------
# WHAT: Pandas provides DataFrames (like Excel spreadsheets in Python).
# WHY: We need it for:
#      - Creating nice tables for metric results
#      - Saving results to CSV files
#      - Easy data manipulation and display
# WHEN: Use Pandas when you need tabular data with labels.
# WHERE: Used in data analysis, data cleaning, and reporting.
# HOW: Import as 'pd' (standard convention).
# INTERNALLY: Built on NumPy, adds row/column labels.
import pandas as pd

# -----------------------------------------------------------------------------
# 1.3 Matplotlib - Plotting Library
# -----------------------------------------------------------------------------
# WHAT: Matplotlib is the most popular Python plotting library.
# WHY: We need it for:
#      - Creating comparison bar plots
#      - Visualizing baseline vs enhanced metrics
#      - Saving plots as PNG files
# WHEN: Use when you need any kind of visualization.
# WHERE: Used in all data science projects for visualization.
# HOW: Import pyplot module as 'plt' (standard convention).
# INTERNALLY: Creates vector graphics that can be saved in many formats.
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1.4 Scikit-learn Imports
# -----------------------------------------------------------------------------

# 1.4.1 load_wine - Wine Dataset Loader
# WHAT: Function that loads the famous Wine recognition dataset.
# WHY: This is our dataset! 178 samples, 13 features, 3 classes.
# WHEN: Use at the start to get our data.
# WHERE: Built into scikit-learn, no download needed.
# HOW: Call load_wine() to get a Bunch object with data and target.
# INTERNALLY: Returns a dictionary-like object with keys: data, target, feature_names, etc.
from sklearn.datasets import load_wine

# 1.4.2 StandardScaler - Feature Normalization
# WHAT: Transforms features to have mean=0 and standard deviation=1.
# WHY: K-Means and Perceptron work better when all features are on the same scale.
#      Without scaling, features with large values (like "proline" in wine) 
#      would dominate the distance calculations.
# WHEN: Almost always before using distance-based algorithms (K-Means, KNN, SVM).
# WHERE: Used in 90% of machine learning pipelines.
# HOW: Create scaler → fit on training data → transform train and test.
# INTERNALLY: For each feature: (x - mean) / std
# ALTERNATIVE: MinMaxScaler scales to [0,1], but StandardScaler is preferred for
#              K-Means because it doesn't bound the data.
from sklearn.preprocessing import StandardScaler

# 1.4.3 KMeans - Clustering Algorithm
# WHAT: Unsupervised algorithm that groups data into k clusters.
# WHY: We use it to find natural groupings in the wine data, then use these
#      groupings as additional features for the Perceptron.
# WHEN: When you want to find structure/groups in unlabeled data.
# WHERE: Customer segmentation, image compression, feature engineering.
# HOW: Create model → fit on data → predict cluster assignments.
# INTERNALLY: 
#      1. Randomly place k centroids
#      2. Assign each point to nearest centroid
#      3. Move centroids to mean of their assigned points
#      4. Repeat until convergence
from sklearn.cluster import KMeans

# 1.4.4 Perceptron - Linear Classifier
# WHAT: The simplest neural network - a single layer with linear decision boundary.
# WHY: We use it as our classifier to compare baseline vs enhanced features.
#      It's simple enough to clearly show the effect of feature augmentation.
# WHEN: For linearly separable problems, or as a baseline classifier.
# WHERE: Text classification, simple pattern recognition, educational purposes.
# HOW: Create model → fit on X, y → predict on new data.
# INTERNALLY: 
#      1. Start with random weights
#      2. For each misclassified sample: weights += learning_rate * y * x
#      3. Repeat until no misclassifications or max iterations
# ALTERNATIVE: Logistic Regression is more common in practice, but Perceptron
#              is specified in the problem statement.
from sklearn.linear_model import Perceptron

# 1.4.5 StratifiedKFold - Cross-Validation Splitter
# WHAT: Splits data into k folds, preserving class proportions in each fold.
# WHY: Our data is imbalanced (59 class 0, 71 class 1, 48 class 2).
#      Regular KFold might create folds with very few positive samples.
#      Stratified ensures each fold has ~33% class 0 samples.
# WHEN: ALWAYS use for classification with imbalanced classes.
# WHERE: Standard practice in all classification evaluations.
# HOW: Create splitter → call split(X, y) to get train/test indices.
# INTERNALLY: Groups samples by class, then distributes each class evenly across folds.
# ALTERNATIVE: Regular KFold doesn't preserve class proportions - DON'T use for
#              imbalanced classification!
from sklearn.model_selection import StratifiedKFold

# 1.4.6 Evaluation Metrics
# WHAT: Functions to measure how well our classifier performs.
# WHY: We need multiple metrics because accuracy alone is misleading for imbalanced data.
#      - accuracy_score: % of correct predictions
#      - f1_score: Balance between precision and recall
#      - average_precision_score: Area under precision-recall curve
# WHEN: After training, to evaluate model performance.
# WHERE: Used in every classification project.
# HOW: Call with (y_true, y_pred) or (y_true, y_proba) for AP.
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

# 1.4.7 Statistical Testing
# WHAT: ttest_rel performs paired t-test on two related samples.
# WHY: We need to know if the difference between baseline and enhanced is
#      statistically significant, or just due to random chance.
#      - p-value < 0.05 means the difference is significant
#      - p-value >= 0.05 means we can't claim a real difference
# WHEN: When comparing two methods on the same data.
# WHERE: Scientific research, A/B testing, model comparison.
# HOW: ttest_rel(scores1, scores2) returns (statistic, p-value).
# INTERNALLY: Tests if the mean difference is significantly different from 0.
from scipy.stats import ttest_rel

# -----------------------------------------------------------------------------
# 1.5 Warnings - Suppress Convergence Warnings
# -----------------------------------------------------------------------------
# WHAT: Python module to control warning messages.
# WHY: Perceptron may not converge in 1000 iterations, which triggers warnings.
#      These warnings clutter the output but aren't critical for learning.
# WHEN: When you want clean output without non-critical warnings.
# WHERE: Common in production code and demonstrations.
# HOW: filterwarnings("ignore") suppresses all warnings.
# CAUTION: In production, be careful about suppressing warnings!
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1.6 OS - Operating System Interface
# -----------------------------------------------------------------------------
# WHAT: Module for interacting with the operating system.
# WHY: We need it to:
#      - Create output directories if they don't exist
#      - Construct file paths that work on Windows/Mac/Linux
# WHEN: When saving files or checking file locations.
# WHERE: Used in almost all Python programs that handle files.
# HOW: os.makedirs() creates directories, os.path.join() builds paths.
import os


# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 Random Seed
# -----------------------------------------------------------------------------
# WHAT: A number that initializes the random number generator.
# WHY: Makes the experiment REPRODUCIBLE. Same seed = same random numbers = same results.
#      Without this, every run would give different fold splits, different K-Means
#      initialization, and different Perceptron results.
# WHEN: ALWAYS set at the start of any ML experiment.
# WHERE: Used in research, competitions, and debugging.
# HOW: Pass to StratifiedKFold, KMeans, and Perceptron.
# VALUE: 42 is the "answer to everything" in The Hitchhiker's Guide to the Galaxy.
#        It's a popular choice but any number works.
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# 2.2 Number of Clusters for K-Means
# -----------------------------------------------------------------------------
# WHAT: The number of groups K-Means will create.
# WHY: Specified in the problem statement as k=4.
#      4 clusters means 4 one-hot columns + 4 distance columns = 8 new features.
# WHEN: Must be decided before running K-Means.
# WHERE: Value depends on the problem. Here it's given.
# HOW: Pass as n_clusters parameter to KMeans.
# IMPACT: More clusters = more features, but also more complexity.
N_CLUSTERS = 4

# -----------------------------------------------------------------------------
# 2.3 Number of Cross-Validation Folds
# -----------------------------------------------------------------------------
# WHAT: How many times to split the data for training/testing.
# WHY: 5-fold CV is the industry standard. Balances:
#      - Enough folds to get stable estimates
#      - Not too many folds (would make training sets too similar)
# WHEN: Standard practice for model evaluation.
# WHERE: Used in almost all ML experiments.
# HOW: Pass to StratifiedKFold(n_splits=5).
# ALTERNATIVE: 10-fold is sometimes used but 5-fold is sufficient for 178 samples.
N_FOLDS = 5

# -----------------------------------------------------------------------------
# 2.4 Output Directory
# -----------------------------------------------------------------------------
# WHAT: Where to save results (CSV, PNG, text files).
# WHY: Keeps outputs organized and separate from code.
# WHEN: Set at the start, used when saving files.
# WHERE: Standard practice to have dedicated output folders.
# HOW: Use os.path.join() to build paths.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================

def create_binary_labels(y):
    """
    Convert multi-class labels to binary: class 0 = positive (1), others = negative (0).
    
    -------------------------------------------------------------------------
    WHAT THIS FUNCTION DOES:
    -------------------------------------------------------------------------
    Takes the original 3-class wine labels (0, 1, 2) and converts them to binary:
        - Class 0 → 1 (positive, the wine type we're trying to detect)
        - Class 1 → 0 (negative)
        - Class 2 → 0 (negative)
    
    -------------------------------------------------------------------------
    WHY THIS IS NEEDED:
    -------------------------------------------------------------------------
    The problem asks us to create a binary classification task.
    Binary classification is simpler and lets us use metrics like F1-score
    and Average Precision more directly.
    
    -------------------------------------------------------------------------
    WHEN TO USE:
    -------------------------------------------------------------------------
    When you need to convert a multi-class problem to binary, or when you
    want to detect "one class vs all others" (One-vs-Rest approach).
    
    -------------------------------------------------------------------------
    WHERE IT'S USED:
    -------------------------------------------------------------------------
    - Medical diagnosis: Sick vs Healthy
    - Fraud detection: Fraud vs Legitimate
    - Spam detection: Spam vs Not Spam
    
    -------------------------------------------------------------------------
    HOW IT WORKS INTERNALLY:
    -------------------------------------------------------------------------
    1. Creates an array of zeros same length as y
    2. Wherever y == 0, sets the value to 1
    3. Returns the new binary array
    
    -------------------------------------------------------------------------
    PARAMETERS:
    -------------------------------------------------------------------------
    y : numpy.ndarray
        Original labels with 3 classes (0, 1, 2).
        Shape: (n_samples,)
        
        WHAT: The target variable from load_wine().
        WHY: We need to convert this to binary.
        HOW: Just pass wine.target directly.
    
    -------------------------------------------------------------------------
    RETURNS:
    -------------------------------------------------------------------------
    numpy.ndarray
        Binary labels: 1 for class 0, 0 for others.
        Shape: (n_samples,)
    
    -------------------------------------------------------------------------
    EXAMPLE:
    -------------------------------------------------------------------------
    >>> y = np.array([0, 1, 2, 0, 1])
    >>> create_binary_labels(y)
    array([1, 0, 0, 1, 0])
    """
    # np.where(condition, value_if_true, value_if_false)
    # If y == 0, return 1 (positive class)
    # Otherwise, return 0 (negative class)
    return np.where(y == 0, 1, 0)


def augment_features(X, kmeans_model):
    """
    Augment features with one-hot cluster membership and centroid distances.
    
    -------------------------------------------------------------------------
    WHAT THIS FUNCTION DOES:
    -------------------------------------------------------------------------
    Takes original features and adds two types of new features:
    
    1. ONE-HOT CLUSTER MEMBERSHIP (4 new columns for k=4):
       - If sample belongs to cluster 2: [0, 0, 1, 0]
       - This tells the model "which group" the sample is in
    
    2. CENTROID DISTANCES (4 new columns for k=4):
       - Distance from sample to centroid 0, 1, 2, 3
       - This tells the model "how similar" the sample is to each group
    
    Total: Original 13 features + 4 one-hot + 4 distances = 21 features
    
    -------------------------------------------------------------------------
    WHY THIS IS NEEDED:
    -------------------------------------------------------------------------
    The Perceptron only sees raw chemical measurements. By adding cluster info:
    
    - ONE-HOT tells it: "This wine is a Type-2 wine"
    - DISTANCES tell it: "It's very close to Type-2, but also somewhat like Type-0"
    
    This gives the Perceptron MORE INFORMATION to make better decisions!
    
    -------------------------------------------------------------------------
    REAL-LIFE ANALOGY:
    -------------------------------------------------------------------------
    Imagine classifying students as "likely to pass" or "likely to fail":
    
    BASELINE (Original features only):
        - Math grade: 85
        - English grade: 72
        - ... 11 more subjects
    
    ENHANCED (Augmented features):
        - All original grades PLUS:
        - Student type: "Nerd" (one-hot: [1, 0, 0, 0])
        - Similarity to Nerd: 0.2 (very close)
        - Similarity to Athlete: 2.5 (far)
        - Similarity to Artist: 1.8 (medium)
        - Similarity to Socialite: 3.1 (very far)
    
    Now the classifier knows not just the grades, but also the "type" of student!
    
    -------------------------------------------------------------------------
    WHEN TO USE:
    -------------------------------------------------------------------------
    When you believe the data has natural clusters/groups that could help
    the classifier. This is called FEATURE ENGINEERING.
    
    -------------------------------------------------------------------------
    WHERE IT'S USED:
    -------------------------------------------------------------------------
    - Customer segmentation + churn prediction
    - Image clustering + classification
    - Gene expression clustering + disease prediction
    
    -------------------------------------------------------------------------
    HOW IT WORKS INTERNALLY:
    -------------------------------------------------------------------------
    1. Predict cluster assignments using the trained K-Means model
    2. Create one-hot encoding: np.eye(k)[cluster_labels]
       - np.eye(k) creates a k×k identity matrix
       - Using cluster labels as indices selects the right row
    3. Calculate Euclidean distance from each sample to each centroid
       - For sample x and centroid c: distance = sqrt(sum((x - c)^2))
    4. Concatenate: original X + one-hot + distances
    
    -------------------------------------------------------------------------
    PARAMETERS:
    -------------------------------------------------------------------------
    X : numpy.ndarray
        Original feature matrix.
        Shape: (n_samples, n_features) = (n, 13) for wine data
        
        WHAT: The standardized wine features.
        WHY: We need to augment these with cluster info.
        HOW: Pass the output of scaler.transform().
        
    kmeans_model : sklearn.cluster.KMeans
        A FITTED K-Means model (already trained on data).
        
        WHAT: The clustering model that assigns samples to clusters.
        WHY: We need it to:
              1. Predict cluster assignments (.predict())
              2. Get centroid locations (.cluster_centers_)
        HOW: Pass a model that was already fit with kmeans.fit(X_train).
        
        IMPORTANT: The model must be fit on TRAINING data only!
        If fit on test data, we have DATA LEAKAGE (cheating).
    
    -------------------------------------------------------------------------
    RETURNS:
    -------------------------------------------------------------------------
    numpy.ndarray
        Augmented feature matrix.
        Shape: (n_samples, n_features + 2*n_clusters) = (n, 13 + 4 + 4) = (n, 21)
        
        Columns 0-12:  Original 13 features
        Columns 13-16: One-hot cluster membership (4 columns)
        Columns 17-20: Distances to 4 centroids (4 columns)
    
    -------------------------------------------------------------------------
    EXAMPLE:
    -------------------------------------------------------------------------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
    >>> kmeans = KMeans(n_clusters=2).fit(X)
    >>> X_aug = augment_features(X, kmeans)
    >>> X_aug.shape
    (3, 6)  # 2 original + 2 one-hot + 2 distances
    """
    # -------------------------------------------------------------------------
    # Step 1: Get cluster assignments for each sample
    # -------------------------------------------------------------------------
    # kmeans_model.predict(X) returns an array of cluster labels
    # Example: [2, 0, 1, 2, 0, ...] means first sample is in cluster 2, etc.
    cluster_labels = kmeans_model.predict(X)
    
    # -------------------------------------------------------------------------
    # Step 2: Create one-hot encoding of cluster assignments
    # -------------------------------------------------------------------------
    # np.eye(k) creates a k×k identity matrix:
    # For k=4: [[1,0,0,0],
    #           [0,1,0,0],
    #           [0,0,1,0],
    #           [0,0,0,1]]
    #
    # Using cluster_labels as indices:
    # If cluster_labels[i] = 2, np.eye(4)[2] = [0, 0, 1, 0]
    #
    # This is a clever trick to avoid writing a loop!
    n_clusters = kmeans_model.n_clusters
    one_hot = np.eye(n_clusters)[cluster_labels]
    
    # -------------------------------------------------------------------------
    # Step 3: Calculate distances to each centroid
    # -------------------------------------------------------------------------
    # kmeans_model.cluster_centers_ has shape (n_clusters, n_features)
    # For each sample, compute Euclidean distance to each centroid
    #
    # Euclidean distance formula: sqrt(sum((x - c)^2))
    #
    # We use a loop for clarity, but this could be vectorized.
    centroids = kmeans_model.cluster_centers_
    n_samples = X.shape[0]
    
    # Initialize distance matrix: n_samples rows × n_clusters columns
    distances = np.zeros((n_samples, n_clusters))
    
    # For each cluster centroid
    for i in range(n_clusters):
        # Subtract centroid from all samples: broadcast (n, 13) - (13,)
        diff = X - centroids[i]
        
        # Square each element
        diff_squared = diff ** 2
        
        # Sum across features (axis=1), then square root
        distances[:, i] = np.sqrt(np.sum(diff_squared, axis=1))
    
    # -------------------------------------------------------------------------
    # Step 4: Concatenate all features
    # -------------------------------------------------------------------------
    # np.hstack() horizontally stacks arrays
    # Result shape: (n_samples, 13 + 4 + 4) = (n_samples, 21)
    X_augmented = np.hstack([X, one_hot, distances])
    
    return X_augmented


def run_cross_validation(X, y, n_clusters=N_CLUSTERS, n_folds=N_FOLDS, random_seed=RANDOM_SEED):
    """
    Run stratified k-fold cross-validation comparing baseline and enhanced pipelines.
    
    -------------------------------------------------------------------------
    WHAT THIS FUNCTION DOES:
    -------------------------------------------------------------------------
    This is the MAIN EXPERIMENT function. It:
    1. Splits data into 5 stratified folds
    2. For each fold:
       - Fits StandardScaler on training data
       - Fits K-Means on training data  
       - Creates augmented features for train and test
       - Trains baseline Perceptron on original features
       - Trains enhanced Perceptron on augmented features
       - Calculates metrics for both
    3. Returns all fold-wise metrics
    
    -------------------------------------------------------------------------
    WHY THIS IS NEEDED:
    -------------------------------------------------------------------------
    We need a FAIR comparison between baseline and enhanced pipelines.
    Cross-validation ensures:
    - Every sample is tested once
    - Results aren't dependent on a single train/test split
    - We can compute confidence intervals
    
    -------------------------------------------------------------------------
    CRITICAL: PREVENTING DATA LEAKAGE
    -------------------------------------------------------------------------
    The K-Means model is fit ONLY on training data within each fold.
    
    WHY THIS MATTERS:
    - If we fit K-Means on ALL data (including test), the test samples
      would influence where centroids are placed
    - This would make the test results overly optimistic (cheating!)
    - In production, we won't have access to test data during training
    
    ANALOGY:
    - WRONG: A teacher creates exam questions after seeing all student essays
    - RIGHT: A teacher creates questions using only practice essays, not final submissions
    
    -------------------------------------------------------------------------
    WHEN TO USE:
    -------------------------------------------------------------------------
    Use stratified k-fold CV for any classification evaluation, especially
    with imbalanced classes.
    
    -------------------------------------------------------------------------
    WHERE IT'S USED:
    -------------------------------------------------------------------------
    - Model comparison studies
    - Hyperparameter tuning (with nested CV)
    - Research paper experiments
    - Kaggle competitions
    
    -------------------------------------------------------------------------
    HOW IT WORKS INTERNALLY:
    -------------------------------------------------------------------------
    1. StratifiedKFold ensures each fold has ~same class proportions
    2. For fold i:
       - Training set: all samples NOT in fold i (~80%)
       - Test set: all samples IN fold i (~20%)
    3. Fit scaler and K-Means on training set only
    4. Transform/predict on both train and test
    5. Train two Perceptrons, evaluate both
    6. Store metrics for this fold
    7. Repeat for all 5 folds
    
    -------------------------------------------------------------------------
    PARAMETERS:
    -------------------------------------------------------------------------
    X : numpy.ndarray
        Feature matrix, shape (n_samples, n_features).
        
        WHAT: The wine features (13 chemical measurements).
        WHY: These are the inputs to our classifiers.
        HOW: Pass the .data attribute from load_wine().
        
    y : numpy.ndarray
        BINARY labels, shape (n_samples,).
        
        WHAT: The target variable (1 for class 0, 0 for others).
        WHY: This is what we're trying to predict.
        HOW: Pass the output of create_binary_labels().
        
        IMPORTANT: Must be BINARY, not the original 3-class labels!
        
    n_clusters : int, default=4
        Number of clusters for K-Means.
        
        WHAT: The 'k' in K-Means.
        WHY: Specified in problem statement as k=4.
        HOW: Use default or specify explicitly.
        
        IMPACT ON OUTPUT:
        - More clusters = more augmented features
        - 4 clusters = 4 one-hot + 4 distances = 8 new features
        
    n_folds : int, default=5
        Number of cross-validation folds.
        
        WHAT: How many times to split data.
        WHY: 5-fold is industry standard (80/20 split each time).
        HOW: Use default for standard evaluation.
        
        IMPACT:
        - More folds = more reliable estimates but longer runtime
        - Fewer folds = faster but less reliable
        
    random_seed : int, default=42
        Random seed for reproducibility.
        
        WHAT: Controls random number generator.
        WHY: Same seed = same fold splits = reproducible results.
        HOW: Use default or your preferred number.
        
        IMPACT:
        - Same seed = identical results every run
        - Different seed = slightly different metrics
    
    -------------------------------------------------------------------------
    RETURNS:
    -------------------------------------------------------------------------
    dict
        Dictionary with keys:
        - 'baseline_accuracy': list of 5 accuracy scores
        - 'baseline_f1': list of 5 F1 scores
        - 'baseline_ap': list of 5 average precision scores
        - 'enhanced_accuracy': list of 5 accuracy scores (augmented features)
        - 'enhanced_f1': list of 5 F1 scores
        - 'enhanced_ap': list of 5 average precision scores
    
    -------------------------------------------------------------------------
    EXAMPLE:
    -------------------------------------------------------------------------
    >>> X, y = load_wine(return_X_y=True)
    >>> y_binary = create_binary_labels(y)
    >>> results = run_cross_validation(X, y_binary)
    >>> print(np.mean(results['baseline_accuracy']))
    0.85  # Example value
    """
    # -------------------------------------------------------------------------
    # Initialize metric storage
    # -------------------------------------------------------------------------
    # Each list will store one value per fold
    results = {
        'baseline_accuracy': [],
        'baseline_f1': [],
        'baseline_ap': [],
        'enhanced_accuracy': [],
        'enhanced_f1': [],
        'enhanced_ap': []
    }
    
    # -------------------------------------------------------------------------
    # Create stratified k-fold splitter
    # -------------------------------------------------------------------------
    # StratifiedKFold ensures each fold has proportional class distribution
    # shuffle=True randomizes sample order before splitting
    # random_state makes it reproducible
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # -------------------------------------------------------------------------
    # Cross-validation loop
    # -------------------------------------------------------------------------
    # skf.split(X, y) yields (train_indices, test_indices) for each fold
    # enumerate() gives us the fold number (0, 1, 2, 3, 4)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Print progress
        print(f"Processing Fold {fold_idx + 1}/{n_folds}...")
        
        # ---------------------------------------------------------------------
        # Step 1: Split data into train and test for this fold
        # ---------------------------------------------------------------------
        # X[train_idx] selects rows at positions in train_idx
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # ---------------------------------------------------------------------
        # Step 2: Standardize features (FIT ON TRAIN ONLY)
        # ---------------------------------------------------------------------
        # Create a new scaler for this fold
        # fit_transform: learns mean/std from X_train AND transforms it
        # transform: uses the learned mean/std to transform X_test
        #
        # WHY FIT ON TRAIN ONLY:
        # In production, we won't know the test data's mean/std
        # We simulate this by not letting the scaler "see" test data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ---------------------------------------------------------------------
        # Step 3: Fit K-Means on training data (FIT ON TRAIN ONLY)
        # ---------------------------------------------------------------------
        # This is CRITICAL for preventing data leakage!
        #
        # The K-Means model learns cluster centroids from training data.
        # Test samples are assigned to clusters based on these centroids,
        # but they DON'T influence where centroids are placed.
        #
        # n_init=10: Run K-Means 10 times with different random starts,
        #            keep the best result (lowest inertia)
        # random_state: Makes it reproducible
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_seed)
        kmeans.fit(X_train_scaled)  # FIT ON TRAIN ONLY!
        
        # ---------------------------------------------------------------------
        # Step 4: Create augmented features
        # ---------------------------------------------------------------------
        # For training: augment with cluster membership + distances
        # For testing: use the SAME K-Means model (trained on train data)
        X_train_augmented = augment_features(X_train_scaled, kmeans)
        X_test_augmented = augment_features(X_test_scaled, kmeans)
        
        # Print feature dimensions for first fold
        if fold_idx == 0:
            print(f"  Original features: {X_train_scaled.shape[1]}")
            print(f"  Augmented features: {X_train_augmented.shape[1]}")
        
        # ---------------------------------------------------------------------
        # Step 5: Train BASELINE Perceptron (original features only)
        # ---------------------------------------------------------------------
        # Perceptron with default parameters:
        # - max_iter=1000: Maximum number of passes over training data
        # - tol=1e-3: Tolerance for stopping criterion
        # - random_state: For reproducibility
        baseline_model = Perceptron(max_iter=1000, tol=1e-3, random_state=random_seed)
        baseline_model.fit(X_train_scaled, y_train)
        
        # Get predictions
        baseline_pred = baseline_model.predict(X_test_scaled)
        
        # Get decision scores for Average Precision
        # decision_function returns distance from hyperplane
        baseline_scores = baseline_model.decision_function(X_test_scaled)
        
        # ---------------------------------------------------------------------
        # Step 6: Train ENHANCED Perceptron (augmented features)
        # ---------------------------------------------------------------------
        enhanced_model = Perceptron(max_iter=1000, tol=1e-3, random_state=random_seed)
        enhanced_model.fit(X_train_augmented, y_train)
        
        # Get predictions
        enhanced_pred = enhanced_model.predict(X_test_augmented)
        
        # Get decision scores for Average Precision
        enhanced_scores = enhanced_model.decision_function(X_test_augmented)
        
        # ---------------------------------------------------------------------
        # Step 7: Calculate metrics for BASELINE
        # ---------------------------------------------------------------------
        # Accuracy: % of correct predictions
        # F1 Score: Harmonic mean of precision and recall
        # Average Precision: Area under precision-recall curve
        results['baseline_accuracy'].append(accuracy_score(y_test, baseline_pred))
        results['baseline_f1'].append(f1_score(y_test, baseline_pred))
        results['baseline_ap'].append(average_precision_score(y_test, baseline_scores))
        
        # ---------------------------------------------------------------------
        # Step 8: Calculate metrics for ENHANCED
        # ---------------------------------------------------------------------
        results['enhanced_accuracy'].append(accuracy_score(y_test, enhanced_pred))
        results['enhanced_f1'].append(f1_score(y_test, enhanced_pred))
        results['enhanced_ap'].append(average_precision_score(y_test, enhanced_scores))
    
    print("Cross-validation complete!\n")
    return results


def create_results_table(results):
    """
    Create a formatted DataFrame with fold-wise and summary statistics.
    
    -------------------------------------------------------------------------
    WHAT THIS FUNCTION DOES:
    -------------------------------------------------------------------------
    Converts the dictionary of fold results into a nicely formatted table
    that shows:
    1. Metrics for each fold (Fold 1, 2, 3, 4, 5)
    2. Mean and standard deviation across folds
    
    -------------------------------------------------------------------------
    WHY THIS IS NEEDED:
    -------------------------------------------------------------------------
    Raw numbers in a dictionary are hard to read. This creates a publication-
    quality table that can be:
    - Displayed in the notebook
    - Saved to CSV
    - Included in reports
    
    -------------------------------------------------------------------------
    PARAMETERS:
    -------------------------------------------------------------------------
    results : dict
        Output from run_cross_validation().
        Contains lists of fold-wise metrics for each pipeline/metric combo.
    
    -------------------------------------------------------------------------
    RETURNS:
    -------------------------------------------------------------------------
    pandas.DataFrame
        Formatted table with rows for each fold + Mean ± Std row.
    """
    # -------------------------------------------------------------------------
    # Create DataFrame structure
    # -------------------------------------------------------------------------
    # Each row = one fold, columns = metrics for baseline and enhanced
    data = {
        'Fold': [f'Fold {i+1}' for i in range(N_FOLDS)] + ['Mean ± Std'],
        'Baseline Accuracy': results['baseline_accuracy'] + [None],
        'Enhanced Accuracy': results['enhanced_accuracy'] + [None],
        'Baseline F1': results['baseline_f1'] + [None],
        'Enhanced F1': results['enhanced_f1'] + [None],
        'Baseline AP': results['baseline_ap'] + [None],
        'Enhanced AP': results['enhanced_ap'] + [None]
    }
    
    df = pd.DataFrame(data)
    
    # -------------------------------------------------------------------------
    # Calculate summary statistics
    # -------------------------------------------------------------------------
    # For each metric, compute mean ± std
    summary_row = ['Mean ± Std']
    
    for col in ['Baseline Accuracy', 'Enhanced Accuracy', 
                'Baseline F1', 'Enhanced F1',
                'Baseline AP', 'Enhanced AP']:
        values = df[col].iloc[:-1].values  # Exclude the summary row
        mean_val = np.mean(values)
        std_val = np.std(values)
        df.loc[df['Fold'] == 'Mean ± Std', col] = f'{mean_val:.4f} ± {std_val:.4f}'
    
    # Format fold rows to 4 decimal places
    for col in df.columns[1:]:
        df[col] = df[col].apply(
            lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x
        )
    
    return df


def perform_statistical_tests(results):
    """
    Perform paired t-tests to check if differences are statistically significant.
    
    -------------------------------------------------------------------------
    WHAT THIS FUNCTION DOES:
    -------------------------------------------------------------------------
    Uses the paired t-test to determine if there's a statistically significant
    difference between baseline and enhanced metrics.
    
    -------------------------------------------------------------------------
    WHY THIS IS NEEDED:
    -------------------------------------------------------------------------
    If enhanced is 1% better than baseline, is that REAL or just luck?
    The t-test answers this:
    - p-value < 0.05: The difference is statistically significant (real!)
    - p-value >= 0.05: The difference could be due to random chance
    
    -------------------------------------------------------------------------
    WHY PAIRED T-TEST:
    -------------------------------------------------------------------------
    We use PAIRED t-test because:
    - Both methods are evaluated on the SAME 5 folds
    - Fold 1 results are paired (same test samples)
    - This accounts for fold-to-fold variation
    
    ALTERNATIVE:
    - Independent t-test: Used when comparing two different groups
    - Wilcoxon signed-rank: Non-parametric alternative (fewer assumptions)
    
    -------------------------------------------------------------------------
    PARAMETERS:
    -------------------------------------------------------------------------
    results : dict
        Output from run_cross_validation().
    
    -------------------------------------------------------------------------
    RETURNS:
    -------------------------------------------------------------------------
    dict
        Dictionary with p-values and interpretation for each metric.
    """
    metrics = ['accuracy', 'f1', 'ap']
    metric_names = {'accuracy': 'Accuracy', 'f1': 'F1 Score', 'ap': 'Average Precision'}
    
    stat_results = {}
    
    for metric in metrics:
        baseline_scores = results[f'baseline_{metric}']
        enhanced_scores = results[f'enhanced_{metric}']
        
        # Paired t-test
        # Returns: (t_statistic, p_value)
        # t_statistic: How many standard errors the mean difference is from 0
        # p_value: Probability of seeing this difference by chance
        t_stat, p_value = ttest_rel(enhanced_scores, baseline_scores)
        
        # Calculate mean improvement
        mean_diff = np.mean(enhanced_scores) - np.mean(baseline_scores)
        
        # Interpret significance
        if p_value < 0.05:
            significance = "Statistically Significant"
        else:
            significance = "Not Statistically Significant"
        
        stat_results[metric] = {
            'name': metric_names[metric],
            'baseline_mean': np.mean(baseline_scores),
            'enhanced_mean': np.mean(enhanced_scores),
            'mean_difference': mean_diff,
            'percent_change': (mean_diff / np.mean(baseline_scores)) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': significance
        }
    
    return stat_results


def create_comparison_plot(results, output_path):
    """
    Create a bar chart comparing baseline and enhanced metrics with error bars.
    
    -------------------------------------------------------------------------
    WHAT THIS FUNCTION DOES:
    -------------------------------------------------------------------------
    Creates a grouped bar chart showing:
    - Baseline and Enhanced metrics side by side
    - Error bars showing standard deviation
    - Clear labels and legend
    
    -------------------------------------------------------------------------
    WHY THIS IS NEEDED:
    -------------------------------------------------------------------------
    Visual comparisons are easier to understand than tables of numbers.
    Error bars show the variability across folds—if error bars overlap,
    the difference might not be significant.
    
    -------------------------------------------------------------------------
    PARAMETERS:
    -------------------------------------------------------------------------
    results : dict
        Output from run_cross_validation().
        
    output_path : str
        Where to save the PNG file.
    """
    # -------------------------------------------------------------------------
    # Calculate means and standard deviations
    # -------------------------------------------------------------------------
    metrics = ['Accuracy', 'F1 Score', 'Average Precision']
    baseline_means = [
        np.mean(results['baseline_accuracy']),
        np.mean(results['baseline_f1']),
        np.mean(results['baseline_ap'])
    ]
    baseline_stds = [
        np.std(results['baseline_accuracy']),
        np.std(results['baseline_f1']),
        np.std(results['baseline_ap'])
    ]
    enhanced_means = [
        np.mean(results['enhanced_accuracy']),
        np.mean(results['enhanced_f1']),
        np.mean(results['enhanced_ap'])
    ]
    enhanced_stds = [
        np.std(results['enhanced_accuracy']),
        np.std(results['enhanced_f1']),
        np.std(results['enhanced_ap'])
    ]
    
    # -------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------
    # Set up figure with good size for visibility
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar width and positions
    x = np.arange(len(metrics))  # [0, 1, 2]
    width = 0.35  # Width of each bar
    
    # Create bars
    # x - width/2 positions baseline bars to the left
    # x + width/2 positions enhanced bars to the right
    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label='Baseline (Original Features)', color='steelblue',
                   capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, enhanced_means, width, yerr=enhanced_stds,
                   label='Enhanced (Augmented Features)', color='darkorange',
                   capsize=5, alpha=0.8)
    
    # -------------------------------------------------------------------------
    # Customize the plot
    # -------------------------------------------------------------------------
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline vs Enhanced Perceptron: Cross-Validation Results', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    def add_labels(bars, stds):
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1, baseline_stds)
    add_labels(bars2, enhanced_stds)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # Grid behind bars
    
    # -------------------------------------------------------------------------
    # Save and show
    # -------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Comparison plot saved to: {output_path}")


def generate_executive_summary(results, stat_results):
    """
    Generate a 400-450 word executive summary with production recommendations.
    
    -------------------------------------------------------------------------
    WHAT THIS FUNCTION DOES:
    -------------------------------------------------------------------------
    Creates a professional executive summary that:
    1. Summarizes the experiment and results
    2. Analyzes statistical significance
    3. Discusses operational considerations
    4. Makes a clear production recommendation
    
    -------------------------------------------------------------------------
    WHY THIS IS NEEDED:
    -------------------------------------------------------------------------
    Senior stakeholders need a concise summary, not raw data. This summary:
    - Answers: "Should we use clustering augmentation in production?"
    - Provides evidence-based reasoning
    - Considers both statistical and practical significance
    
    -------------------------------------------------------------------------
    PARAMETERS:
    -------------------------------------------------------------------------
    results : dict
        Output from run_cross_validation().
        
    stat_results : dict
        Output from perform_statistical_tests().
    
    -------------------------------------------------------------------------
    RETURNS:
    -------------------------------------------------------------------------
    str
        The executive summary text (400-450 words).
    """
    # Count how many metrics improved
    improved_metrics = sum(
        1 for metric in stat_results.values() 
        if metric['mean_difference'] > 0
    )
    
    # Count significant improvements
    significant_improvements = sum(
        1 for metric in stat_results.values()
        if metric['mean_difference'] > 0 and metric['p_value'] < 0.05
    )
    
    # Determine recommendation
    if significant_improvements >= 2:
        recommendation = "RECOMMENDED"
        reason = "statistically significant improvements in multiple metrics"
    elif improved_metrics >= 2 and any(m['p_value'] < 0.10 for m in stat_results.values() if m['mean_difference'] > 0):
        recommendation = "CONDITIONALLY RECOMMENDED"
        reason = "consistent positive trends, though not all reached statistical significance"
    else:
        recommendation = "NOT RECOMMENDED"
        reason = "insufficient evidence of reliable improvement"
    
    # Build the summary
    summary = f"""
EXECUTIVE SUMMARY: K-MEANS FEATURE AUGMENTATION FOR WINE CLASSIFICATION
================================================================================

OBJECTIVE
---------
This study evaluated whether augmenting Perceptron classifiers with K-Means clustering 
features improves binary classification performance on the Wine dataset. We compared 
a baseline Perceptron using 13 original features against an enhanced Perceptron using 
21 features (original + 4 one-hot cluster memberships + 4 centroid distances).

METHODOLOGY
-----------
We employed stratified 5-fold cross-validation to ensure robust evaluation. Critically, 
K-Means (k=4) was fit exclusively on each training fold to prevent data leakage—a common 
pitfall that inflates performance estimates. Three metrics were captured: Accuracy, F1 
Score, and Average Precision.

RESULTS
-------
Accuracy: Baseline {stat_results['accuracy']['baseline_mean']:.4f} vs Enhanced {stat_results['accuracy']['enhanced_mean']:.4f} 
  ({stat_results['accuracy']['percent_change']:+.2f}% change, p={stat_results['accuracy']['p_value']:.4f}, {stat_results['accuracy']['significance']})

F1 Score: Baseline {stat_results['f1']['baseline_mean']:.4f} vs Enhanced {stat_results['f1']['enhanced_mean']:.4f} 
  ({stat_results['f1']['percent_change']:+.2f}% change, p={stat_results['f1']['p_value']:.4f}, {stat_results['f1']['significance']})

Average Precision: Baseline {stat_results['ap']['baseline_mean']:.4f} vs Enhanced {stat_results['ap']['enhanced_mean']:.4f} 
  ({stat_results['ap']['percent_change']:+.2f}% change, p={stat_results['ap']['p_value']:.4f}, {stat_results['ap']['significance']})

STATISTICAL SIGNIFICANCE
------------------------
Of the three metrics evaluated, {significant_improvements} showed statistically significant improvement 
(p < 0.05). The paired t-tests account for fold-to-fold variation, providing reliable 
estimates of the true performance difference.

OPERATIONAL CONSIDERATIONS
--------------------------
The enhanced pipeline introduces moderate complexity:
• Additional preprocessing step (K-Means clustering)
• Increased feature dimension (13 → 21 features, 62% increase)
• Marginally higher training time due to K-Means fit per fold
• Requirement to store cluster centroids for production inference

These overheads are minimal for modern systems but should be weighed against the 
observed performance gains.

RECOMMENDATION
--------------
Based on the evidence, feature augmentation via K-Means clustering is **{recommendation}** 
for production deployment. This conclusion is based on {reason}. 

If improvements are marginal, consider:
1. Alternative clustering algorithms (DBSCAN, Gaussian Mixture)
2. Different values of k (hyperparameter tuning)
3. More powerful classifiers that may not benefit from manual feature engineering

Final decision should balance statistical evidence with business requirements for 
accuracy, interpretability, and system complexity.
"""
    
    # Word count check
    word_count = len(summary.split())
    print(f"Executive Summary Word Count: {word_count}")
    
    return summary


# =============================================================================
# SECTION 4: MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function that orchestrates the entire experiment.
    
    -------------------------------------------------------------------------
    WHAT THIS FUNCTION DOES:
    -------------------------------------------------------------------------
    1. Loads and prepares the Wine dataset
    2. Runs cross-validation experiment
    3. Creates results table and saves to CSV
    4. Performs statistical tests
    5. Creates comparison plot
    6. Generates executive summary
    7. Saves all outputs
    
    -------------------------------------------------------------------------
    WHY MAIN FUNCTION:
    -------------------------------------------------------------------------
    Organizing code into a main() function is best practice because:
    - Makes the code reusable as a module (can import without executing)
    - Clearly shows the high-level flow
    - Easier to test individual components
    """
    print("=" * 80)
    print("K-MEANS FEATURE AUGMENTATION + PERCEPTRON CROSS-VALIDATION")
    print("=" * 80)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Load and prepare data
    # -------------------------------------------------------------------------
    print("Step 1: Loading Wine dataset...")
    wine = load_wine()
    X = wine.data  # Shape: (178, 13)
    y_original = wine.target  # Shape: (178,), values: 0, 1, 2
    
    # Create binary labels
    y_binary = create_binary_labels(y_original)
    
    print(f"  Dataset shape: {X.shape}")
    print(f"  Original classes: {np.unique(y_original)}")
    print(f"  Binary labels: Class 0 (positive) = {np.sum(y_binary)}, Class 1/2 (negative) = {np.sum(1 - y_binary)}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Run cross-validation
    # -------------------------------------------------------------------------
    print("Step 2: Running stratified 5-fold cross-validation...")
    results = run_cross_validation(X, y_binary)
    
    # -------------------------------------------------------------------------
    # Step 3: Create and save results table
    # -------------------------------------------------------------------------
    print("Step 3: Creating results table...")
    results_df = create_results_table(results)
    print(results_df.to_string(index=False))
    print()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "cross_validation_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Perform statistical tests
    # -------------------------------------------------------------------------
    print("Step 4: Performing statistical significance tests...")
    stat_results = perform_statistical_tests(results)
    
    print("\nStatistical Test Results:")
    print("-" * 60)
    for metric, stats in stat_results.items():
        print(f"{stats['name']}:")
        print(f"  Baseline Mean: {stats['baseline_mean']:.4f}")
        print(f"  Enhanced Mean: {stats['enhanced_mean']:.4f}")
        print(f"  Difference: {stats['mean_difference']:+.4f} ({stats['percent_change']:+.2f}%)")
        print(f"  p-value: {stats['p_value']:.4f}")
        print(f"  Conclusion: {stats['significance']}")
        print()
    
    # -------------------------------------------------------------------------
    # Step 5: Create comparison plot
    # -------------------------------------------------------------------------
    print("Step 5: Creating comparison plot...")
    plot_path = os.path.join(OUTPUT_DIR, "comparison_plot.png")
    create_comparison_plot(results, plot_path)
    print()
    
    # -------------------------------------------------------------------------
    # Step 6: Generate executive summary
    # -------------------------------------------------------------------------
    print("Step 6: Generating executive summary...")
    summary = generate_executive_summary(results, stat_results)
    
    # Save summary to file
    summary_path = os.path.join(OUTPUT_DIR, "executive_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Executive summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(summary)
    print("=" * 80)
    
    print("\n✅ All outputs generated successfully!")
    print(f"   - Metrics table: {csv_path}")
    print(f"   - Comparison plot: {plot_path}")
    print(f"   - Executive summary: {summary_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
# This block only runs if the script is executed directly (not imported)
# It's Python best practice for reusable modules

if __name__ == "__main__":
    main()
