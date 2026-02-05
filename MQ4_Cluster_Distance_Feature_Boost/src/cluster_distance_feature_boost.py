"""
Cluster-Distance Feature Boost for Binary Perceptron
=====================================================

This script demonstrates how adding distance-to-centroid features from K-Means
clustering can significantly improve a simple Perceptron classifier's performance.

Real-Life Analogy:
------------------
Imagine you're trying to identify which students belong to "Class A" in a school.
You have basic information (height, weight), but it's hard to tell them apart.
Now, if you add extra information like "how close each student is to the average
Class A student," your classification becomes much more accurate!

Author: Teaching Project
Date: 2026
"""

# =============================================================================
# SECTION 1: IMPORTING LIBRARIES
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 What: Import numpy for numerical operations (arrays, math)
# 2.2 Why: We need arrays to store data and do math efficiently. 
#          Lists are slow for math operations, numpy is 10-100x faster.
# 2.3 When: Always import at the start of any data science project
# 2.4 Where: Every ML/Data Science project uses numpy
# 2.5 How: import numpy as np (np is the standard abbreviation)
# 2.6 Internal: numpy uses C code for fast array operations
# 2.7 Output: No output, just makes numpy available as 'np'
# -----------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------
# 2.1 What: Import make_blobs to generate synthetic clustered data
# 2.2 Why: We need test data with known cluster structure. Real data is messy,
#          synthetic data lets us understand concepts clearly first.
# 2.3 When: For learning, testing algorithms, or when real data isn't available
# 2.4 Where: Tutorials, prototyping, algorithm comparisons
# 2.5 How: from sklearn.datasets import make_blobs
# 2.6 Internal: Generates random points around specified centers
# 2.7 Output: Returns X (features) and y (cluster labels)
# -----------------------------------------------------------------------------
from sklearn.datasets import make_blobs

# -----------------------------------------------------------------------------
# 2.1 What: Import StandardScaler to normalize features to mean=0, std=1
# 2.2 Why: Different features have different scales (e.g., age in years, 
#          salary in thousands). Scaling makes them comparable.
#          Perceptron and K-Means are sensitive to feature scales.
# 2.3 When: Before training most ML models (especially distance-based ones)
# 2.4 Where: Almost every ML pipeline includes scaling
# 2.5 How: scaler = StandardScaler(); scaler.fit_transform(X)
# 2.6 Internal: For each value: (value - mean) / std_deviation
# 2.7 Output: Transformed data where each feature has mean=0, std=1
# -----------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 2.1 What: Import KMeans for clustering data into k groups
# 2.2 Why: We want to find natural groupings in data and use distances
#          to these group centers as new features.
# 2.3 When: When you suspect data has natural clusters/groups
# 2.4 Where: Customer segmentation, image compression, feature engineering
# 2.5 How: kmeans = KMeans(n_clusters=3); kmeans.fit(X)
# 2.6 Internal: Iteratively moves centroids to minimize within-cluster distances
# 2.7 Output: Cluster labels and centroid locations
# -----------------------------------------------------------------------------
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# 2.1 What: Import Perceptron - a simple linear classifier
# 2.2 Why: It's the simplest neural network (just weights + bias). 
#          Great baseline to show improvement from feature engineering.
# 2.3 When: As a baseline, for linearly separable data, for teaching
# 2.4 Where: First step in learning neural networks, simple classification
# 2.5 How: model = Perceptron(); model.fit(X, y)
# 2.6 Internal: Learns weights w such that sign(w·x + b) predicts class
# 2.7 Output: Trained model that can predict class labels
# -----------------------------------------------------------------------------
from sklearn.linear_model import Perceptron

# -----------------------------------------------------------------------------
# 2.1 What: Import train_test_split to divide data into train and test sets
# 2.2 Why: We need separate data to train and evaluate the model.
#          Training and testing on same data gives false confidence (overfitting).
# 2.3 When: Always before training any ML model
# 2.4 Where: Every ML project with supervised learning
# 2.5 How: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# 2.6 Internal: Randomly shuffles data, splits at specified ratio
# 2.7 Output: Four arrays: training features, test features, training labels, test labels
# -----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 2.1 What: Import classification metrics (accuracy, precision, recall, ROC AUC)
# 2.2 Why: We need to measure how good our classifier is.
#          Different metrics tell us different things about performance.
# 2.3 When: After making predictions on test data
# 2.4 Where: Every classification problem needs evaluation metrics
# 2.5 How: accuracy_score(y_true, y_pred), precision_score(y_true, y_pred)
# 2.6 Internal: Compares predictions to actual labels, calculates ratios
# 2.7 Output: Numbers between 0 and 1 (higher is better)
# -----------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# -----------------------------------------------------------------------------
# 2.1 What: Import numpy column_stack to combine arrays horizontally
# 2.2 Why: We need to add distance features to original features.
#          This creates one big feature matrix.
# 2.3 When: When combining features from different sources
# 2.4 Where: Feature engineering, data preprocessing
# 2.5 How: combined = np.column_stack([X1, X2])
# 2.6 Internal: Stacks arrays side-by-side along column axis
# 2.7 Output: Combined array with columns from all input arrays
# -----------------------------------------------------------------------------
# (Already imported numpy above, column_stack is np.column_stack)

# =============================================================================
# SECTION 2: GENERATE SYNTHETIC DATA
# =============================================================================

def generate_data():
    """
    Generate synthetic blob data with 3 clusters.
    
    3.1 What: Creates 900 data points grouped into 3 clusters
    3.2 Why: We need structured data where we know the "ground truth"
             This helps us understand if our algorithm is working
    3.3 When: At the start of the project, before any training
    3.4 Where: Any clustering or classification tutorial
    3.5 How: X, cluster_ids = make_blobs(n_samples=900, centers=3, ...)
    3.6 Internal: 
        - Randomly picks 3 center points
        - Generates 300 points around each center
        - Adds Gaussian noise based on cluster_std
    3.7 Output: 
        - X: (900, 2) array of coordinates
        - y: (900,) array of binary labels (0 or 1)
    
    Returns:
        X (np.array): Feature matrix of shape (900, 2)
        y (np.array): Binary labels (1 if cluster 0, else 0)
    """
    # -------------------------------------------------------------------------
    # make_blobs Parameters:
    # -------------------------------------------------------------------------
    # n_samples=900: 
    #   3.1 What: Total number of data points to generate
    #   3.2 Why: 900 gives ~300 per cluster, enough to see patterns
    #   3.3 When: Set based on how much data you need
    #   3.4 Where: Always the first decision in data generation
    #   3.5 How: Larger = more data, better learning, slower training
    #   3.6 Internal: Divides 900 equally among 3 centers = 300 each
    #   3.7 Output: 900 rows in the X array
    #
    # centers=3:
    #   3.1 What: Number of cluster centers (groups)
    #   3.2 Why: We want 3 distinct groups to cluster
    #   3.3 When: Based on domain knowledge or experimentation
    #   3.4 Where: Determines the structure of generated data
    #   3.5 How: Higher = more complex data
    #   3.6 Internal: Randomly places 3 center points in feature space
    #   3.7 Output: Data naturally groups around 3 regions
    #
    # cluster_std=[1.0, 1.2, 1.4]:
    #   3.1 What: Standard deviation (spread) of each cluster
    #   3.2 Why: Different spreads make clusters of different sizes
    #            This creates realistic, non-uniform data
    #   3.3 When: When you want varied cluster densities
    #   3.4 Where: Realistic data often has clusters of different spreads
    #   3.5 How: Higher std = more spread out points
    #   3.6 Internal: Adds Gaussian noise with given std to each cluster
    #   3.7 Output: Cluster 0 is tight (1.0), Cluster 2 is spread (1.4)
    #
    # random_state=12:
    #   3.1 What: Seed for random number generator
    #   3.2 Why: Makes results reproducible (same data every time)
    #   3.3 When: Always use for reproducibility in teaching/testing
    #   3.4 Where: Every random operation that needs to be repeated
    #   3.5 How: Any integer works, just pick one and stick with it
    #   3.6 Internal: Seeds numpy's random number generator
    #   3.7 Output: Same random data every time you run the code
    # -------------------------------------------------------------------------
    
    X, cluster_ids = make_blobs(
        n_samples=900,
        centers=3,
        cluster_std=[1.0, 1.2, 1.4],
        random_state=12,
    )
    
    # -------------------------------------------------------------------------
    # Create binary labels: 1 if cluster_id == 0, else 0
    # 2.1 What: Converts cluster labels (0,1,2) to binary (1,0,0)
    # 2.2 Why: We want a binary classification problem
    #          "Is this point in cluster 0 or not?"
    # 2.3 When: When converting multi-class to binary problem
    # 2.4 Where: One-vs-all classification, binary problems
    # 2.5 How: (cluster_ids == 0) gives True/False, .astype(int) gives 1/0
    # 2.6 Internal: Boolean comparison, then type casting
    # 2.7 Output: ~300 points with label 1, ~600 with label 0
    # -------------------------------------------------------------------------
    y = (cluster_ids == 0).astype(int)
    
    return X, y


# =============================================================================
# SECTION 3: FEATURE ENGINEERING
# =============================================================================

def create_distance_features(X_train, X_test, n_clusters=3, random_state=12):
    """
    Create distance-to-centroid features using K-Means.
    
    3.1 What: Fits K-Means on training data and computes distances
              from each point to each of the k cluster centers
    3.2 Why: Distance features capture "cluster geometry" - how points
             relate to natural groupings in the data. This gives the
             classifier extra information about data structure.
    3.3 When: When you want to enhance a classifier with clustering info
    3.4 Where: Feature engineering, semi-supervised learning
    3.5 How: 
        1. Fit KMeans on X_train
        2. Use kmeans.transform() to get distances
        3. Apply same transformation to X_test
    3.6 Internal: 
        - K-Means finds k centroids that minimize within-cluster distances
        - transform() computes Euclidean distance from each point to each centroid
    3.7 Output: 
        - train_distances: (n_train, k) distances for training data
        - test_distances: (n_test, k) distances for test data
    
    Parameters:
        X_train (np.array): Training features, shape (n_train, n_features)
        X_test (np.array): Test features, shape (n_test, n_features)
        n_clusters (int): Number of clusters (k). Default=3.
        random_state (int): Random seed for reproducibility. Default=12.
    
    Returns:
        train_distances (np.array): Distance features for training data
        test_distances (np.array): Distance features for test data
    """
    # -------------------------------------------------------------------------
    # KMeans Parameters:
    # -------------------------------------------------------------------------
    # n_clusters=3:
    #   3.1 What: Number of clusters to find
    #   3.2 Why: We know data has 3 natural clusters (we created it that way)
    #   3.3 When: Set based on domain knowledge or elbow method
    #   3.4 Where: Always required for K-Means
    #   3.5 How: Higher k = more fine-grained clustering
    #   3.6 Internal: Creates k centroid variables to optimize
    #   3.7 Output: k distance features (one per cluster)
    #
    # random_state=12:
    #   3.1 What: Seed for centroid initialization
    #   3.2 Why: K-Means results depend on initial centroid positions
    #   3.3 When: For reproducible results
    #   3.4 Where: Any K-Means application that needs consistency
    #   3.5 How: Same seed = same initial centroids
    #   3.6 Internal: Seeds the k-means++ initialization
    #   3.7 Output: Same clustering every run
    #
    # n_init=10:
    #   3.1 What: Number of times to run K-Means with different seeds
    #   3.2 Why: K-Means can get stuck in local minima, multiple runs help
    #   3.3 When: Always use default (10) or higher for important results
    #   3.4 Where: All K-Means applications
    #   3.5 How: Higher = more robust but slower
    #   3.6 Internal: Runs algorithm n_init times, keeps best result
    #   3.7 Output: Best clustering out of 10 attempts
    # -------------------------------------------------------------------------
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    
    # -------------------------------------------------------------------------
    # Fit K-Means on training data only
    # 2.1 What: Finds k cluster centers that minimize within-cluster variance
    # 2.2 Why: We only fit on training data to avoid data leakage.
    #          Using test data would give unfair advantage (cheating!).
    # 2.3 When: During the training phase
    # 2.4 Where: Any unsupervised learning algorithm
    # 2.5 How: kmeans.fit(X_train) learns centroid positions
    # 2.6 Internal: 
    #     1. Initialize k random centroids
    #     2. Assign each point to nearest centroid
    #     3. Move centroids to mean of assigned points
    #     4. Repeat until convergence
    # 2.7 Output: kmeans object with cluster_centers_ attribute
    # -------------------------------------------------------------------------
    kmeans.fit(X_train)
    
    # -------------------------------------------------------------------------
    # Transform to get distance features
    # 2.1 What: Computes distance from each point to each cluster center
    # 2.2 Why: These distances are our new features! They tell us how
    #          far each point is from each cluster's "center of gravity".
    # 2.3 When: After fitting K-Means
    # 2.4 Where: Feature engineering, distance-based classification
    # 2.5 How: distances = kmeans.transform(X)
    # 2.6 Internal: For each point, calculates Euclidean distance to each centroid
    #              distance[i,j] = sqrt(sum((X[i] - centroid[j])^2))
    # 2.7 Output: Array of shape (n_samples, n_clusters)
    #             - Row i: distances from point i to all k centroids
    #             - Column j: distances from all points to centroid j
    # -------------------------------------------------------------------------
    train_distances = kmeans.transform(X_train)
    test_distances = kmeans.transform(X_test)
    
    return train_distances, test_distances


# =============================================================================
# SECTION 4: STANDARDIZE FEATURES
# =============================================================================

def standardize_features(X_train, X_test):
    """
    Standardize features to have mean=0 and std=1.
    
    3.1 What: Transforms features so each has mean=0 and standard deviation=1
    3.2 Why: 
        - Different features may have vastly different scales
        - K-Means uses distances, which are affected by scale
        - Perceptron convergence is better with normalized features
    3.3 When: Before training most ML models
    3.4 Where: Standard part of any ML pipeline
    3.5 How: 
        1. Fit scaler on training data (learn mean and std)
        2. Transform both train and test data
    3.6 Internal: For each feature: z = (x - mean) / std
    3.7 Output: Scaled arrays where each feature has mean≈0, std≈1
    
    Parameters:
        X_train (np.array): Training features
        X_test (np.array): Test features
    
    Returns:
        X_train_scaled (np.array): Scaled training features
        X_test_scaled (np.array): Scaled test features
    """
    
    scaler = StandardScaler()
    
    # -------------------------------------------------------------------------
    # fit_transform on training data
    # 2.1 What: Learns mean/std from training data AND transforms it
    # 2.2 Why: Training data defines the "baseline" for scaling.
    #          We learn what "normal" looks like from training data.
    # 2.3 When: First step of preprocessing
    # 2.4 Where: Any scaling/normalization task
    # 2.5 How: fit() learns parameters, transform() applies them
    #          fit_transform() does both in one step for efficiency
    # 2.6 Internal: 
    #     - Calculates mean = sum(X) / n for each feature
    #     - Calculates std = sqrt(sum((X - mean)^2) / n)
    #     - Stores these as scaler.mean_ and scaler.scale_
    # 2.7 Output: Transformed training data with mean≈0, std≈1
    # -------------------------------------------------------------------------
    X_train_scaled = scaler.fit_transform(X_train)
    
    # -------------------------------------------------------------------------
    # transform on test data (NOT fit_transform!)
    # 2.1 What: Applies the SAME scaling learned from training data
    # 2.2 Why: Test data should be scaled using training statistics.
    #          This simulates real-world: we don't know test data ahead of time.
    #          Using test data's own mean/std would be "cheating" (data leakage).
    # 2.3 When: When preprocessing new/test data
    # 2.4 Where: Any ML pipeline - crucial for correct evaluation
    # 2.5 How: scaler.transform(X_test) uses stored mean_ and scale_
    # 2.6 Internal: Uses mean and std learned from training data
    # 2.7 Output: Transformed test data using training statistics
    # -------------------------------------------------------------------------
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


# =============================================================================
# SECTION 5: TRAIN AND EVALUATE MODELS
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Train a Perceptron and evaluate with multiple metrics.
    
    3.1 What: Trains a Perceptron classifier and computes evaluation metrics
    3.2 Why: We need to measure how well our classifier performs
    3.3 When: After preparing features
    3.4 Where: Evaluation phase of any ML project
    3.5 How: 
        1. Create Perceptron
        2. Fit on training data
        3. Predict on test data
        4. Calculate metrics
    3.6 Internal: Perceptron learns weights that separate classes with a hyperplane
    3.7 Output: Dictionary with accuracy, precision, recall, ROC AUC
    
    Parameters:
        X_train (np.array): Training features
        X_test (np.array): Test features
        y_train (np.array): Training labels
        y_test (np.array): Test labels
        model_name (str): Name for display purposes
    
    Returns:
        metrics (dict): Dictionary with accuracy, precision, recall, roc_auc
    """
    
    # -------------------------------------------------------------------------
    # Create and train Perceptron
    # -------------------------------------------------------------------------
    # Perceptron Parameters:
    # random_state=12:
    #   3.1 What: Seed for weight initialization
    #   3.2 Why: Makes training reproducible
    #   3.3 When: For consistent results across runs
    #   3.4 Where: Any stochastic algorithm
    #   3.5 How: Same seed = same initial weights
    #   3.6 Internal: Seeds numpy's random for initialization
    #   3.7 Output: Same model every run
    #
    # max_iter=1000:
    #   3.1 What: Maximum training epochs (passes through data)
    #   3.2 Why: Upper limit to prevent infinite training
    #   3.3 When: Always set a reasonable maximum
    #   3.4 Where: Any iterative algorithm
    #   3.5 How: Training stops at convergence OR max_iter
    #   3.6 Internal: Counts complete passes through training data
    #   3.7 Output: Higher = potentially better fit, longer training
    #
    # tol=1e-3:
    #   3.1 What: Tolerance for convergence
    #   3.2 Why: Stops training when improvement is tiny
    #   3.3 When: Always (prevents unnecessary computation)
    #   3.4 Where: Iterative optimization algorithms
    #   3.5 How: Stops if loss improvement < tol
    #   3.6 Internal: Compares loss between epochs
    #   3.7 Output: Smaller tol = more precise, longer training
    # -------------------------------------------------------------------------
    
    model = Perceptron(random_state=12, max_iter=1000, tol=1e-3)
    
    # -------------------------------------------------------------------------
    # Fit the model
    # 2.1 What: Trains the Perceptron to find optimal weights
    # 2.2 Why: The model needs to learn from data before it can predict
    # 2.3 When: Training phase
    # 2.4 Where: Every supervised learning algorithm
    # 2.5 How: model.fit(X, y) learns weights w and bias b
    # 2.6 Internal: 
    #     For each point:
    #       1. Compute prediction: y_pred = sign(w·x + b)
    #       2. If wrong: update w = w + y * x, b = b + y
    #     Repeat until convergence or max_iter
    # 2.7 Output: Trained model with learned coef_ and intercept_
    # -------------------------------------------------------------------------
    model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # Make predictions
    # 2.1 What: Uses learned weights to classify test data
    # 2.2 Why: We need predictions to calculate metrics
    # 2.3 When: Evaluation phase
    # 2.4 Where: After training, before evaluation
    # 2.5 How: y_pred = model.predict(X_test)
    # 2.6 Internal: For each test point, computes sign(w·x + b)
    # 2.7 Output: Array of predicted class labels (0 or 1)
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)
    
    # -------------------------------------------------------------------------
    # Get probability scores for ROC AUC
    # 2.1 What: Gets decision function values (not true probabilities)
    # 2.2 Why: ROC AUC needs continuous scores, not just 0/1 predictions
    # 2.3 When: When computing ROC AUC or probability-based metrics
    # 2.4 Where: Binary classification evaluation
    # 2.5 How: scores = model.decision_function(X_test)
    # 2.6 Internal: Returns w·x + b (distance from decision boundary)
    # 2.7 Output: Array of scores (positive = class 1, negative = class 0)
    # -------------------------------------------------------------------------
    y_scores = model.decision_function(X_test)
    
    # -------------------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------------------
    
    # Accuracy: What fraction of predictions are correct?
    # 2.1 What: (TP + TN) / (TP + TN + FP + FN)
    # 2.2 Why: Simple overall measure of correctness
    # 2.3 When: First metric to check
    # 2.4 Where: Every classification problem
    # 2.5 How: accuracy_score(y_true, y_pred)
    # 2.6 Internal: Counts matches, divides by total
    # 2.7 Output: Number between 0 and 1
    accuracy = accuracy_score(y_test, y_pred)
    
    # Precision: Of predicted positives, how many are actually positive?
    # 2.1 What: TP / (TP + FP)
    # 2.2 Why: Important when false positives are costly
    #          Example: Spam filter - don't want to delete real emails
    # 2.3 When: When you care about "purity" of positive predictions
    # 2.4 Where: Spam detection, fraud detection
    # 2.5 How: precision_score(y_true, y_pred)
    # 2.6 Internal: Counts TP and FP, calculates ratio
    # 2.7 Output: Number between 0 and 1 (higher = fewer false positives)
    precision = precision_score(y_test, y_pred, zero_division=0)
    
    # Recall: Of actual positives, how many did we find?
    # 2.1 What: TP / (TP + FN)
    # 2.2 Why: Important when false negatives are costly
    #          Example: Disease detection - don't want to miss sick patients
    # 2.3 When: When you need to catch all positives
    # 2.4 Where: Medical diagnosis, security threats
    # 2.5 How: recall_score(y_true, y_pred)
    # 2.6 Internal: Counts TP and FN, calculates ratio
    # 2.7 Output: Number between 0 and 1 (higher = fewer misses)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    # ROC AUC: Area under the ROC curve
    # 2.1 What: Measures how well the model ranks positive vs negative
    # 2.2 Why: Threshold-independent metric, very robust
    # 2.3 When: For overall model quality assessment
    # 2.4 Where: Binary classification comparison
    # 2.5 How: roc_auc_score(y_true, y_scores)
    # 2.6 Internal: 
    #     - Plots True Positive Rate vs False Positive Rate
    #     - Calculates area under this curve
    #     - 0.5 = random, 1.0 = perfect
    # 2.7 Output: Number between 0.5 and 1 (higher = better ranking)
    roc_auc = roc_auc_score(y_test, y_scores)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }


# =============================================================================
# SECTION 6: MAIN EXPERIMENT
# =============================================================================

def run_experiment(n_splits=5):
    """
    Run the complete experiment comparing baseline vs enhanced Perceptron.
    
    3.1 What: Runs the full experiment over multiple random splits
    3.2 Why: Single split results can be noisy, averaging gives robust estimates
    3.3 When: For reliable model comparison
    3.4 Where: Any ML experiment that needs statistical validity
    3.5 How: 
        1. Generate data
        2. For each split: train both models, collect metrics
        3. Average metrics across splits
    3.6 Internal: Uses different random states for each split
    3.7 Output: Average metrics for baseline and enhanced models
    
    Parameters:
        n_splits (int): Number of random train/test splits. Default=5.
    
    Returns:
        baseline_avg (dict): Average metrics for baseline model
        enhanced_avg (dict): Average metrics for enhanced model
    """
    
    # Generate data once (same data for all splits)
    X, y = generate_data()
    
    print("=" * 60)
    print("CLUSTER-DISTANCE FEATURE BOOST EXPERIMENT")
    print("=" * 60)
    print(f"\nData Shape: {X.shape}")
    print(f"Class Distribution: 1s = {sum(y)}, 0s = {len(y) - sum(y)}")
    print(f"Running {n_splits} random splits...")
    print()
    
    # Storage for metrics across splits
    baseline_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': []}
    enhanced_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': []}
    
    for split_idx in range(n_splits):
        # -------------------------------------------------------------------------
        # Split data with different random state each time
        # 2.1 What: Creates train/test split with 75/25 ratio
        # 2.2 Why: Need separate data for training and evaluation
        # 2.3 When: Before training
        # 2.4 Where: Every ML experiment
        # 2.5 How: train_test_split with test_size=0.25
        # 2.6 Internal: Randomly shuffles, splits at index n*0.75
        # 2.7 Output: 675 training, 225 test samples
        # -------------------------------------------------------------------------
        random_state = 42 + split_idx  # Different seed for each split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state
        )
        
        # -------------------------------------------------------------------------
        # Standardize features
        # -------------------------------------------------------------------------
        X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
        
        # -------------------------------------------------------------------------
        # Create distance features from K-Means
        # -------------------------------------------------------------------------
        train_distances, test_distances = create_distance_features(
            X_train_scaled, X_test_scaled, n_clusters=3, random_state=12
        )
        
        # -------------------------------------------------------------------------
        # Create enhanced feature sets
        # 2.1 What: Combines original features with distance features
        # 2.2 Why: This is the "boost" - extra information for the classifier
        # 2.3 When: After creating distance features
        # 2.4 Where: Feature engineering step
        # 2.5 How: np.column_stack([X, distances])
        # 2.6 Internal: Horizontally concatenates arrays
        # 2.7 Output: 5 features (2 original + 3 distance)
        # -------------------------------------------------------------------------
        X_train_enhanced = np.column_stack([X_train_scaled, train_distances])
        X_test_enhanced = np.column_stack([X_test_scaled, test_distances])
        
        # -------------------------------------------------------------------------
        # Train and evaluate baseline model (original features only)
        # -------------------------------------------------------------------------
        baseline_result = train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test, "Baseline"
        )
        
        # -------------------------------------------------------------------------
        # Train and evaluate enhanced model (original + distance features)
        # -------------------------------------------------------------------------
        enhanced_result = train_and_evaluate(
            X_train_enhanced, X_test_enhanced, y_train, y_test, "Enhanced"
        )
        
        # Collect metrics
        for metric in baseline_metrics:
            baseline_metrics[metric].append(baseline_result[metric])
            enhanced_metrics[metric].append(enhanced_result[metric])
        
        print(f"Split {split_idx + 1}: Baseline Acc={baseline_result['accuracy']:.3f}, "
              f"Enhanced Acc={enhanced_result['accuracy']:.3f}")
    
    # -------------------------------------------------------------------------
    # Calculate average metrics
    # -------------------------------------------------------------------------
    baseline_avg = {k: np.mean(v) for k, v in baseline_metrics.items()}
    enhanced_avg = {k: np.mean(v) for k, v in enhanced_metrics.items()}
    
    return baseline_avg, enhanced_avg


def print_results(baseline_avg, enhanced_avg):
    """
    Print a formatted comparison table of results.
    
    3.1 What: Displays metrics in a nice table format
    3.2 Why: Easy comparison of baseline vs enhanced model
    3.3 When: After running the experiment
    3.4 Where: Console output
    3.5 How: Formatted print statements
    3.6 Internal: String formatting and calculations
    3.7 Output: Printed table to console
    """
    
    print("\n" + "=" * 60)
    print("RESULTS: AVERAGED OVER 5 RANDOM SPLITS")
    print("=" * 60)
    
    print(f"\n{'Metric':<12} {'Baseline':>12} {'Enhanced':>12} {'Improvement':>14}")
    print("-" * 52)
    
    for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
        base = baseline_avg[metric]
        enh = enhanced_avg[metric]
        improvement = (enh - base) * 100  # in percentage points
        
        # Highlight if improvement >= 5%
        highlight = " [OK]" if improvement >= 5 else ""
        
        print(f"{metric.upper():<12} {base:>12.4f} {enh:>12.4f} {improvement:>+12.2f}%{highlight}")
    
    print("-" * 52)
    print()


def print_explanation():
    """
    Print the 200-word explanation of results.
    
    3.1 What: Explains why distance features helped
    3.2 Why: Required deliverable with specific content
    3.3 When: After printing results
    3.4 Where: Documentation and understanding
    3.5 How: Formatted multi-line string
    3.6 Internal: Just text output
    3.7 Output: 200-word explanation referencing cluster geometry and boundary shifts
    """
    
    explanation = """
    ============================================================================
    EXPLANATION: WHY DISTANCE FEATURES HELP (200 words)
    ============================================================================
    
    The enhanced model outperforms the baseline because distance-to-centroid features
    capture CLUSTER GEOMETRY that the original 2D features cannot express.
    
    In the original feature space, the Perceptron tries to draw a single linear
    boundary (hyperplane) to separate class 1 (cluster 0) from class 0 (clusters 1, 2).
    However, cluster 0 may be positioned such that a simple line cannot cleanly
    separate it from the overlapping regions of clusters 1 and 2.
    
    By adding distance features, we transform each point into a 5D space where:
    - Points CLOSE to cluster 0's center have SMALL distance to centroid 0
    - Points FAR from cluster 0's center have LARGE distance to centroid 0
    
    This BOUNDARY SHIFT is critical: in the enhanced space, the decision boundary
    can now leverage "closeness to cluster 0" as a feature. Points with small
    distance-to-centroid-0 are highly likely to be class 1, regardless of their
    original x,y position.
    
    The cluster geometry (tight cluster 0 with std=1.0 vs spread clusters 1,2 with
    std=1.2,1.4) means that distance-to-centroid-0 becomes a strong signal for class
    membership. The Perceptron's linear boundary in this enriched space effectively
    creates a NON-LINEAR boundary in the original 2D space.
    
    ============================================================================
    """
    
    print(explanation)


# =============================================================================
# SECTION 7: RUN MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point of the program.
    
    2.1 What: Runs when script is executed directly
    2.2 Why: Allows script to be both imported (as module) and run directly
    2.3 When: When you type 'python script.py' in terminal
    2.4 Where: Every Python script that can run standalone
    2.5 How: if __name__ == "__main__": runs only when executed directly
    2.6 Internal: Python sets __name__ to "__main__" for the entry script
    2.7 Output: Executes the main experiment and prints results
    """
    
    # Run the experiment
    baseline_avg, enhanced_avg = run_experiment(n_splits=5)
    
    # Print formatted results table
    print_results(baseline_avg, enhanced_avg)
    
    # Print 200-word explanation
    print_explanation()
    
    # Verify success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)
    
    improvements = {
        metric: (enhanced_avg[metric] - baseline_avg[metric]) * 100
        for metric in baseline_avg
    }
    
    success = any(imp >= 5 for imp in improvements.values())
    
    if success:
        print("[OK] SUCCESS: At least one metric improved by >=5 percentage points!")
        for metric, imp in improvements.items():
            if imp >= 5:
                print(f"   - {metric.upper()}: +{imp:.2f}%")
    else:
        print("[X] FAILED: No metric improved by >=5 percentage points.")
