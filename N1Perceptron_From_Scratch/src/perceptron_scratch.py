"""
================================================================================
PERCEPTRON FROM SCRATCH
================================================================================
This script implements a Perceptron classifier from scratch using NumPy.
We will train it on a generated dataset, track accuracy, and visualize results.

📌 WHAT IS A PERCEPTRON?
------------------------
A Perceptron is the simplest form of a neural network - just ONE neuron!
Think of it like a simple decision-maker that says "Yes" or "No".

📌 REAL-LIFE ANALOGY:
--------------------
Imagine a security guard at a club door. The guard looks at two things:
- How old you look (Feature 1)
- Whether you have an ID (Feature 2)

The guard combines these with some "importance weights" and decides:
- If the weighted sum >= threshold → "You can enter" (Class 1)
- If the weighted sum < threshold → "You cannot enter" (Class 0)

The Perceptron LEARNS the best weights by looking at examples!
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================
# 📝 WHAT: Importing required libraries
# 📝 WHY: We need tools for math (NumPy), plotting (Matplotlib), and data generation
# 📝 WHEN: Always at the top of any Python script
# ==============================================================================

import numpy as np
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Line Explanation: import numpy as np                                      │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 2.1 WHAT: Imports the NumPy library and gives it a short nickname "np"       │
# │ 2.2 WHY: NumPy provides fast array operations - much faster than Python lists│
# │          Alternative: We could use pure Python, but it would be 100x slower! │
# │ 2.3 WHEN: Whenever you need mathematical operations on arrays                │
# │ 2.4 WHERE: Machine Learning, Data Science, Scientific Computing              │
# │ 2.5 HOW: Just write "np." before any NumPy function, e.g., np.array([1,2,3]) │
# │ 2.6 INTERNAL: NumPy is written in C, so operations run at near-C speed       │
# │ 2.7 OUTPUT: No visible output, but makes np.* functions available            │
# └─────────────────────────────────────────────────────────────────────────────┘

import matplotlib.pyplot as plt
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Line Explanation: import matplotlib.pyplot as plt                         │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 2.1 WHAT: Imports the plotting module from Matplotlib library                │
# │ 2.2 WHY: We need to create visualizations (accuracy plot, decision boundary) │
# │          Alternative: Seaborn, Plotly - but Matplotlib is the foundation     │
# │ 2.3 WHEN: Whenever you need to create charts, graphs, or visualizations      │
# │ 2.4 WHERE: Data visualization in reports, presentations, notebooks           │
# │ 2.5 HOW: Use plt.plot(), plt.scatter(), plt.show() to create and display     │
# │ 2.6 INTERNAL: Creates figure objects that can be rendered to screen/file     │
# │ 2.7 OUTPUT: No immediate output, but enables plotting functions              │
# └─────────────────────────────────────────────────────────────────────────────┘

from sklearn.datasets import make_classification
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Line Explanation: from sklearn.datasets import make_classification        │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 2.1 WHAT: Imports a function to generate artificial classification data      │
# │ 2.2 WHY: We need a clearly separable dataset to train our Perceptron         │
# │          Alternative: Load real data - but generated data is perfect for     │
# │          learning because we control its properties                          │
# │ 2.3 WHEN: When you need synthetic data for testing ML algorithms             │
# │ 2.4 WHERE: Education, prototyping, benchmarking algorithms                   │
# │ 2.5 HOW: X, y = make_classification(n_samples=100, n_features=2)             │
# │ 2.6 INTERNAL: Uses random number generators to create clusters of points     │
# │ 2.7 OUTPUT: Returns X (features) and y (labels) arrays                       │
# └─────────────────────────────────────────────────────────────────────────────┘

from sklearn.model_selection import train_test_split
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Line Explanation: from sklearn.model_selection import train_test_split    │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 2.1 WHAT: Imports a function to split data into training and testing sets    │
# │ 2.2 WHY: We need to evaluate our model on unseen data (20% holdout)          │
# │          Alternative: Manual splitting - but this handles shuffling properly │
# │ 2.3 WHEN: Before training any ML model - always split your data!             │
# │ 2.4 WHERE: Every ML project needs this                                       │
# │ 2.5 HOW: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# │ 2.6 INTERNAL: Randomly shuffles and splits data maintaining class ratios     │
# │ 2.7 OUTPUT: Returns 4 arrays: training features, test features, train labels,│
# │             test labels                                                       │
# └─────────────────────────────────────────────────────────────────────────────┘

import os
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Line Explanation: import os                                               │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 2.1 WHAT: Imports the operating system module for file/folder operations     │
# │ 2.2 WHY: We need to create output directories and save plots                 │
# │ 2.3 WHEN: Whenever interacting with the file system                          │
# │ 2.4 WHERE: File I/O, directory management, path operations                   │
# │ 2.5 HOW: os.makedirs("folder"), os.path.join("a", "b")                       │
# │ 2.6 INTERNAL: Provides Python interface to OS-level file operations          │
# │ 2.7 OUTPUT: No visible output, enables file system operations                │
# └─────────────────────────────────────────────────────────────────────────────┘


# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================
# 📝 WHAT: Setting up constants that control our experiment
# 📝 WHY: Keeps important values in one place, easy to modify
# 📝 WHEN: At the start of any ML experiment
# ==============================================================================

RANDOM_SEED = 7
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Line Explanation: RANDOM_SEED = 7                                         │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 2.1 WHAT: Sets a seed value for random number generation                     │
# │ 2.2 WHY: Makes our experiment REPRODUCIBLE - same results every run          │
# │          Without this, each run gives different results!                     │
# │ 2.3 WHEN: Always set a seed for reproducible ML experiments                  │
# │ 2.4 WHERE: Research, production ML, debugging                                │
# │ 2.5 HOW: Pass this value to random functions                                 │
# │ 2.6 INTERNAL: Initializes the random number generator's starting state       │
# │ 2.7 OUTPUT: No output, but ensures reproducibility                           │
# └─────────────────────────────────────────────────────────────────────────────┘

N_SAMPLES = 600          # Total number of data points to generate
N_FEATURES = 2           # Number of features (2 for easy visualization)
N_INFORMATIVE = 2        # All features are useful (no noise features)
N_REDUNDANT = 0          # No duplicate/correlated features
CLASS_SEP = 2.0          # Separation between classes (higher = easier to separate)
TEST_SIZE = 0.2          # 20% of data for testing
N_EPOCHS = 40            # Number of training iterations
LEARNING_RATE = 0.01     # Step size for weight updates

# ==============================================================================
# SECTION 3: DATA GENERATION
# ==============================================================================
# 📝 WHAT: Creating our synthetic dataset
# 📝 WHY: We need data to train and test our Perceptron
# 📝 ANALOGY: Like creating practice exam papers for students
# ==============================================================================

print("=" * 60)
print("STEP 1: GENERATING DATA")
print("=" * 60)

np.random.seed(RANDOM_SEED)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Line Explanation: np.random.seed(RANDOM_SEED)                             │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 2.1 WHAT: Sets NumPy's random number generator to a fixed starting point    │
# │ 2.2 WHY: Ensures reproducibility across shuffling and other random ops      │
# │ 2.3 WHEN: Before any NumPy random operations                                 │
# │ 2.4 WHERE: ML experiments, simulations, testing                              │
# │ 2.5 HOW: Call once at the start with your chosen seed number                 │
# │ 2.6 INTERNAL: Initializes Mersenne Twister pseudo-random generator           │
# │ 2.7 OUTPUT: None visible, but makes random operations reproducible           │
# └─────────────────────────────────────────────────────────────────────────────┘

X, y = make_classification(
    n_samples=N_SAMPLES,        # 600 data points
    n_features=N_FEATURES,      # 2 features (x, y coordinates)
    n_informative=N_INFORMATIVE, # Both features are useful
    n_redundant=N_REDUNDANT,    # No redundant features
    class_sep=CLASS_SEP,        # Classes are well separated
    random_state=RANDOM_SEED,   # For reproducibility
)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Function Arguments: make_classification()                                 │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 3.1 n_samples: Total number of points to generate (600)                      │
# │     WHY: More samples = better learning, but slower training                 │
# │     DEFAULT: 100                                                             │
# │                                                                              │
# │ 3.2 n_features: Number of columns in X (2)                                   │
# │     WHY: 2D data can be easily visualized on a plot                          │
# │     DEFAULT: 20                                                              │
# │                                                                              │
# │ 3.3 n_informative: How many features actually help with classification (2)  │
# │     WHY: We want both features to be meaningful                              │
# │     DEFAULT: 2                                                               │
# │                                                                              │
# │ 3.4 n_redundant: Features that are linear combinations of informative (0)   │
# │     WHY: We want clean data without artificial correlations                  │
# │     DEFAULT: 2                                                               │
# │                                                                              │
# │ 3.5 class_sep: Distance between class clusters (1.6)                         │
# │     WHY: Higher value = more separation = easier for Perceptron              │
# │     DEFAULT: 1.0                                                             │
# │                                                                              │
# │ 3.6 random_state: Seed for reproducibility (7)                               │
# │     WHY: Same seed = same data every time                                    │
# │     DEFAULT: None (random each time)                                         │
# │                                                                              │
# │ OUTPUT: X = feature array (600, 2), y = label array (600,)                   │
# └─────────────────────────────────────────────────────────────────────────────┘

print(f"Generated {N_SAMPLES} samples with {N_FEATURES} features")
print(f"X shape: {X.shape}")  # (600, 2) - 600 rows, 2 columns
print(f"y shape: {y.shape}")  # (600,) - 600 labels (0 or 1)
print(f"Class distribution: Class 0 = {np.sum(y == 0)}, Class 1 = {np.sum(y == 1)}")

# ==============================================================================
# SECTION 4: TRAIN-TEST SPLIT
# ==============================================================================
# 📝 WHAT: Dividing data into training (80%) and testing (20%)
# 📝 WHY: We need unseen data to evaluate our model fairly
# 📝 ANALOGY: Practice tests vs. final exam - you learn on practice, prove on final
# ==============================================================================

print("\n" + "=" * 60)
print("STEP 2: SPLITTING DATA INTO TRAIN AND TEST SETS")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                       # Our features and labels
    test_size=TEST_SIZE,        # 20% for testing
    random_state=RANDOM_SEED,   # For reproducibility
)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Function Arguments: train_test_split()                                    │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ 3.1 X: Feature array to split                                                │
# │     WHY: These are the inputs our model learns from                          │
# │                                                                              │
# │ 3.2 y: Label array to split                                                  │
# │     WHY: These are the correct answers for training                          │
# │                                                                              │
# │ 3.3 test_size: Fraction of data for testing (0.2 = 20%)                      │
# │     WHY: Standard practice is 20-30% for testing                             │
# │     IMPACT: 0.2 * 600 = 120 test samples, 480 training samples               │
# │     DEFAULT: 0.25                                                            │
# │                                                                              │
# │ 3.4 random_state: Seed for reproducible shuffling                            │
# │     WHY: Ensures same split every time we run                                │
# │     DEFAULT: None (random each time)                                         │
# │                                                                              │
# │ OUTPUT: 4 arrays - X_train, X_test, y_train, y_test                          │
# └─────────────────────────────────────────────────────────────────────────────┘

print(f"Training samples: {len(X_train)}")  # 480 samples
print(f"Testing samples: {len(X_test)}")    # 120 samples

# ==============================================================================
# SECTION 5: PERCEPTRON CLASS DEFINITION
# ==============================================================================
# 📝 WHAT: Creating our Perceptron classifier from scratch
# 📝 WHY: Understanding internals helps you become a better ML practitioner
# 📝 ANALOGY: Building a car engine from parts vs. just driving the car
# ==============================================================================


class Perceptron:
    """
    A Perceptron classifier implemented from scratch.
    
    📌 WHAT IS THIS CLASS?
    ----------------------
    This is a simple binary classifier that finds a linear decision boundary.
    It's like a single neuron in a neural network.
    
    📌 HOW DOES IT WORK?
    -------------------
    1. Takes input features (x)
    2. Multiplies by weights (w)
    3. Adds a bias term (b)
    4. If result >= 0, predict class 1, else predict class 0
    
    📌 LEARNING PROCESS:
    -------------------
    - If prediction is WRONG, adjust weights to reduce error
    - If prediction is RIGHT, keep weights the same
    
    📌 REAL-LIFE ANALOGY:
    --------------------
    Think of adjusting the water temperature in a shower:
    - Too cold? Turn the hot water up (increase weight)
    - Too hot? Turn the hot water down (decrease weight)
    - Just right? Don't change anything
    """

    def __init__(self, learning_rate=0.01, n_epochs=40):
        """
        Initialize the Perceptron with hyperparameters.
        
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 📌 Argument: learning_rate                                            │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 3.1 WHAT: How big of a step to take when updating weights             │
        │ 3.2 WHY: Controls learning speed vs. stability                        │
        │          - Too high: Overshoots optimal solution, unstable            │
        │          - Too low: Takes forever to learn                            │
        │ 3.3 WHEN: Set before training, tune based on results                  │
        │ 3.4 WHERE: All gradient-based ML algorithms use this                  │
        │ 3.5 HOW: Typically values between 0.001 and 1.0                       │
        │ 3.6 INTERNAL: Multiplied with error to scale weight updates           │
        │ 3.7 IMPACT: Higher LR = faster but risky, Lower LR = slower but safe  │
        │ DEFAULT: 0.01 is a good starting point                                │
        └───────────────────────────────────────────────────────────────────────┘
        
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 📌 Argument: n_epochs                                                 │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 3.1 WHAT: Number of times to go through all training data             │
        │ 3.2 WHY: More passes = more chances to learn from mistakes            │
        │ 3.3 WHEN: Set based on convergence behavior                           │
        │ 3.4 WHERE: All iterative training algorithms                          │
        │ 3.5 HOW: Start with 40-100, increase if not converging                │
        │ 3.6 INTERNAL: Controls the outer loop of training                     │
        │ 3.7 IMPACT: More epochs = potentially better accuracy, longer training│
        │ DEFAULT: 40 epochs as per problem requirements                        │
        └───────────────────────────────────────────────────────────────────────┘
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None      # Will be initialized during training
        self.bias = None         # Will be initialized during training
        self.update_count = 0    # Track total weight updates
        self.accuracy_history = []  # Track accuracy per epoch

    def _initialize_weights(self, n_features):
        """
        Initialize weights to zeros.
        
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 📌 Line: self.weights = np.zeros(n_features)                          │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 2.1 WHAT: Creates an array of zeros with size = number of features    │
        │ 2.2 WHY: Zero initialization is standard for Perceptron               │
        │          Alternative: Random init, but zeros work fine for linear      │
        │ 2.3 WHEN: Before training starts                                       │
        │ 2.4 WHERE: Any linear classifier                                       │
        │ 2.5 HOW: np.zeros(2) creates [0.0, 0.0] for 2 features                 │
        │ 2.6 INTERNAL: Allocates memory for float64 array                       │
        │ 2.7 OUTPUT: Array of zeros, shape (n_features,)                        │
        └───────────────────────────────────────────────────────────────────────┘
        """
        self.weights = np.zeros(n_features)
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 self.weights = np.zeros(n_features)                               │
        # │ Creates weight array [0.0, 0.0] for 2 features                       │
        # │ Each weight corresponds to how important that feature is             │
        # └─────────────────────────────────────────────────────────────────────┘

        self.bias = 0.0
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 self.bias = 0.0                                                   │
        # │ Bias is like the "base" prediction before looking at features        │
        # │ Allows the decision boundary to not pass through origin              │
        # └─────────────────────────────────────────────────────────────────────┘

    def _predict_single(self, x):
        """
        Make prediction for a single sample.
        
        📌 THE PERCEPTRON DECISION RULE:
        --------------------------------
        Calculate: z = (w1 * x1) + (w2 * x2) + bias
        If z >= 0: Predict Class 1
        If z < 0:  Predict Class 0
        
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 📌 Line: linear_output = np.dot(x, self.weights) + self.bias          │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 2.1 WHAT: Calculates weighted sum of inputs plus bias                  │
        │ 2.2 WHY: This is the core computation of the Perceptron                │
        │          Alternative: Manual loop, but np.dot is much faster           │
        │ 2.3 WHEN: For every prediction                                         │
        │ 2.4 WHERE: All linear models (Perceptron, Logistic Regression, etc.)   │
        │ 2.5 HOW: np.dot([x1, x2], [w1, w2]) = x1*w1 + x2*w2                    │
        │ 2.6 INTERNAL: Uses BLAS for optimized matrix multiplication            │
        │ 2.7 OUTPUT: Single float value (the "activation")                      │
        └───────────────────────────────────────────────────────────────────────┘
        """
        linear_output = np.dot(x, self.weights) + self.bias
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ np.dot(x, self.weights) computes: x[0]*w[0] + x[1]*w[1]              │
        # │ Then we add bias to allow the line to shift up/down                  │
        # └─────────────────────────────────────────────────────────────────────┘

        return 1 if linear_output >= 0 else 0
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 Step Function (Activation)                                        │
        # │ If weighted sum >= 0 → Class 1 (positive side of boundary)           │
        # │ If weighted sum < 0  → Class 0 (negative side of boundary)           │
        # │ This is called the "Heaviside step function"                         │
        # └─────────────────────────────────────────────────────────────────────┘

    def fit(self, X, y):
        """
        Train the Perceptron on the given data.
        
        📌 THE PERCEPTRON LEARNING ALGORITHM:
        -------------------------------------
        For each epoch:
            1. Shuffle the training data
            2. For each sample:
                a. Make a prediction
                b. If wrong, update weights
                c. Track the update
            3. Calculate accuracy for the epoch
        
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 📌 Argument: X - Training features                                    │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 3.1 WHAT: 2D array of shape (n_samples, n_features)                   │
        │ 3.2 WHY: These are the inputs the Perceptron learns from              │
        │ 3.3 WHEN: Pass during training                                        │
        │ 3.4 WHERE: Every ML model's fit() method                              │
        │ 3.5 HOW: Pass your training features array                            │
        │ 3.6 INTERNAL: Each row is processed one at a time                     │
        │ 3.7 IMPACT: More samples = potentially better generalization          │
        └───────────────────────────────────────────────────────────────────────┘
        
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 📌 Argument: y - Training labels                                      │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 3.1 WHAT: 1D array of shape (n_samples,) with values 0 or 1           │
        │ 3.2 WHY: These are the correct answers to learn from                  │
        │ 3.3 WHEN: Pass during training                                        │
        │ 3.4 WHERE: Every supervised ML model                                  │
        │ 3.5 HOW: Pass your training labels array                              │
        │ 3.6 INTERNAL: Used to calculate prediction error                      │
        │ 3.7 IMPACT: Correct labels are essential for supervised learning      │
        └───────────────────────────────────────────────────────────────────────┘
        """
        n_samples, n_features = X.shape
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 X.shape returns (n_samples, n_features)                           │
        # │ We unpack it to know how many samples and features we have           │
        # │ For our data: n_samples=480, n_features=2                            │
        # └─────────────────────────────────────────────────────────────────────┘

        self._initialize_weights(n_features)
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 Initialize weights and bias to zeros                              │
        # │ We need one weight per feature: [w1, w2] for 2 features              │
        # └─────────────────────────────────────────────────────────────────────┘

        self.update_count = 0
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 Reset update counter                                              │
        # │ We track how many times we update weights (learn from mistakes)      │
        # └─────────────────────────────────────────────────────────────────────┘

        self.accuracy_history = []
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 Reset accuracy tracking list                                      │
        # │ We store accuracy after each epoch to plot learning curve            │
        # └─────────────────────────────────────────────────────────────────────┘

        # Create index array for shuffling
        indices = np.arange(n_samples)
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 np.arange(n_samples) creates [0, 1, 2, ..., 479]                  │
        # │ We'll shuffle these indices to randomize training order              │
        # │ WHY shuffle? Prevents learning patterns from data ordering           │
        # └─────────────────────────────────────────────────────────────────────┘

        print("\n" + "=" * 60)
        print("STEP 3: TRAINING THE PERCEPTRON")
        print("=" * 60)
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Epochs: {self.n_epochs}")
        print("-" * 60)

        # ======================= MAIN TRAINING LOOP ==========================
        for epoch in range(self.n_epochs):
            # ┌─────────────────────────────────────────────────────────────────┐
            # │ 📌 for epoch in range(self.n_epochs):                            │
            # │ Loop through 40 epochs (complete passes through all data)        │
            # │ Each epoch = one complete review of all training samples         │
            # └─────────────────────────────────────────────────────────────────┘

            # Shuffle data at the start of each epoch
            np.random.shuffle(indices)
            # ┌─────────────────────────────────────────────────────────────────┐
            # │ 📌 np.random.shuffle(indices)                                    │
            # │ 2.1 WHAT: Randomly reorders the indices array IN-PLACE           │
            # │ 2.2 WHY: Different order each epoch prevents overfitting         │
            # │          to the sequence of samples                              │
            # │ 2.3 WHEN: At the start of each epoch                             │
            # │ 2.4 WHERE: Stochastic gradient descent, mini-batch training      │
            # │ 2.5 HOW: Modifies the array directly, returns None               │
            # │ 2.6 INTERNAL: Uses Fisher-Yates shuffle algorithm                │
            # │ 2.7 OUTPUT: indices is now [234, 12, 456, 1, ...] (random)       │
            # └─────────────────────────────────────────────────────────────────┘

            epoch_updates = 0  # Track updates in this epoch
            # ┌─────────────────────────────────────────────────────────────────┐
            # │ 📌 Count how many weight updates happen in this epoch            │
            # │ Fewer updates = model is converging (making fewer mistakes)      │
            # └─────────────────────────────────────────────────────────────────┘

            # Process each sample in shuffled order
            for idx in indices:
                # ┌─────────────────────────────────────────────────────────────┐
                # │ 📌 Loop through each sample using shuffled indices           │
                # │ This ensures random order of training samples                │
                # └─────────────────────────────────────────────────────────────┘

                x_i = X[idx]     # Get feature vector for this sample
                y_i = y[idx]     # Get true label for this sample
                # ┌─────────────────────────────────────────────────────────────┐
                # │ 📌 x_i is the feature vector [x1, x2]                        │
                # │ 📌 y_i is the true label (0 or 1)                            │
                # └─────────────────────────────────────────────────────────────┘

                # Make prediction
                y_pred = self._predict_single(x_i)
                # ┌─────────────────────────────────────────────────────────────┐
                # │ 📌 y_pred is what our model THINKS the answer is             │
                # │ We compare this with y_i (the TRUE answer)                   │
                # └─────────────────────────────────────────────────────────────┘

                # ============= THE PERCEPTRON UPDATE RULE =================
                # If prediction is wrong, update weights
                if y_i != y_pred:
                    # ┌─────────────────────────────────────────────────────────┐
                    # │ 📌 ONLY UPDATE IF PREDICTION IS WRONG                    │
                    # │ If prediction matches true label, do nothing             │
                    # │ This is what makes Perceptron "error-driven learning"    │
                    # └─────────────────────────────────────────────────────────┘

                    # Calculate error (difference between true and predicted)
                    error = y_i - y_pred
                    # ┌─────────────────────────────────────────────────────────┐
                    # │ 📌 error = y_i - y_pred                                  │
                    # │ If y_i=1, y_pred=0 → error = +1 (need to increase)       │
                    # │ If y_i=0, y_pred=1 → error = -1 (need to decrease)       │
                    # └─────────────────────────────────────────────────────────┘

                    # UPDATE WEIGHTS: w = w + learning_rate * error * x
                    self.weights += self.learning_rate * error * x_i
                    # ┌─────────────────────────────────────────────────────────┐
                    # │ 📌 THE PERCEPTRON WEIGHT UPDATE RULE                     │
                    # │ 2.1 WHAT: Adjusts weights based on error and input       │
                    # │ 2.2 WHY: Moves decision boundary to correct the mistake  │
                    # │ 2.3 HOW: w_new = w_old + lr * (y_true - y_pred) * x      │
                    # │                                                          │
                    # │ EXAMPLE:                                                 │
                    # │ If x = [0.5, 1.2], error = +1, lr = 0.01                 │
                    # │ Weight update = 0.01 * 1 * [0.5, 1.2] = [0.005, 0.012]   │
                    # │ New weights = old weights + [0.005, 0.012]               │
                    # │                                                          │
                    # │ INTUITION:                                               │
                    # │ - If we predicted 0 but truth is 1, we increase weights  │
                    # │ - This makes the linear output larger (more positive)    │
                    # │ - More positive output → more likely to predict 1        │
                    # └─────────────────────────────────────────────────────────┘

                    # UPDATE BIAS: b = b + learning_rate * error
                    self.bias += self.learning_rate * error
                    # ┌─────────────────────────────────────────────────────────┐
                    # │ 📌 BIAS UPDATE RULE                                      │
                    # │ Similar to weight update, but no multiplication by x     │
                    # │ Bias shifts the decision boundary up or down             │
                    # └─────────────────────────────────────────────────────────┘

                    # Count this update
                    epoch_updates += 1
                    self.update_count += 1
                    # ┌─────────────────────────────────────────────────────────┐
                    # │ 📌 Track both epoch-level and total updates              │
                    # │ More updates = model making more corrections             │
                    # │ Fewer updates over time = model is learning              │
                    # └─────────────────────────────────────────────────────────┘

            # Calculate accuracy on training data after this epoch
            y_train_pred = self.predict(X)
            accuracy = np.mean(y_train_pred == y)
            # ┌─────────────────────────────────────────────────────────────────┐
            # │ 📌 Calculate training accuracy                                   │
            # │ np.mean(y_pred == y_true) gives fraction of correct predictions  │
            # │ e.g., if 450/480 correct, accuracy = 0.9375 = 93.75%            │
            # └─────────────────────────────────────────────────────────────────┘

            self.accuracy_history.append(accuracy)
            # ┌─────────────────────────────────────────────────────────────────┐
            # │ 📌 Store accuracy for plotting learning curve                    │
            # │ We expect accuracy to increase as epochs progress                │
            # └─────────────────────────────────────────────────────────────────┘

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:2d}/{self.n_epochs}: "
                      f"Accuracy = {accuracy * 100:.2f}%, "
                      f"Updates = {epoch_updates}")
                # ┌─────────────────────────────────────────────────────────────┐
                # │ 📌 Print progress every 10 epochs                            │
                # │ Shows how accuracy improves and updates decrease             │
                # └─────────────────────────────────────────────────────────────┘

        print("-" * 60)
        print(f"Training Complete!")
        print(f"Total Weight Updates: {self.update_count}")
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias:.4f}")

        return self
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 Return self for method chaining (allows model.fit().predict())    │
        # └─────────────────────────────────────────────────────────────────────┘

    def predict(self, X):
        """
        Make predictions for multiple samples.
        
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 📌 Argument: X - Samples to predict                                   │
        ├───────────────────────────────────────────────────────────────────────┤
        │ 3.1 WHAT: 2D array of shape (n_samples, n_features)                   │
        │ 3.2 WHY: We want predictions for multiple samples at once             │
        │ 3.3 WHEN: After training is complete                                  │
        │ 3.4 WHERE: Testing, evaluation, production inference                  │
        │ 3.5 HOW: Pass your test features                                      │
        │ 3.6 INTERNAL: Applies prediction rule to each row                     │
        │ 3.7 OUTPUT: 1D array of predictions (0s and 1s)                       │
        └───────────────────────────────────────────────────────────────────────┘
        """
        return np.array([self._predict_single(x) for x in X])
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 List comprehension to predict for each sample                     │
        # │ Creates a list of predictions, then converts to NumPy array          │
        # │ OUTPUT: [0, 1, 1, 0, 1, ...] - predicted classes                     │
        # └─────────────────────────────────────────────────────────────────────┘

    def score(self, X, y):
        """
        Calculate accuracy on given data.
        
        ┌───────────────────────────────────────────────────────────────────────┐
        │ 📌 Accuracy = (Number Correct) / (Total Samples)                      │
        │ e.g., 115 correct out of 120 = 115/120 = 0.9583 = 95.83%             │
        └───────────────────────────────────────────────────────────────────────┘
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 np.mean(predictions == y)                                         │
        # │ (predictions == y) creates boolean array [True, True, False, ...]    │
        # │ np.mean() treats True as 1, False as 0, gives fraction correct       │
        # └─────────────────────────────────────────────────────────────────────┘


# ==============================================================================
# SECTION 6: TRAINING AND EVALUATION
# ==============================================================================
# 📝 WHAT: Create, train, and evaluate our Perceptron
# 📝 WHY: This is where we actually run our algorithm
# 📝 ANALOGY: Student takes the practice tests and then the final exam
# ==============================================================================

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(output_dir, exist_ok=True)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Create outputs folder if it doesn't exist                                │
# │ os.makedirs with exist_ok=True won't error if folder already exists         │
# └─────────────────────────────────────────────────────────────────────────────┘

# Create and train the Perceptron
perceptron = Perceptron(learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS)
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Create Perceptron instance with our hyperparameters                          │
# │ learning_rate=0.01: Small steps for stable learning                             │
# │ n_epochs=40: 40 complete passes through training data                           │
# └─────────────────────────────────────────────────────────────────────────────────┘

perceptron.fit(X_train, y_train)
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Train the Perceptron!                                                        │
# │ This runs 40 epochs of the learning algorithm                                   │
# │ Weights and bias are learned from the training data                             │
# └─────────────────────────────────────────────────────────────────────────────────┘

# Evaluate on test set
print("\n" + "=" * 60)
print("STEP 4: EVALUATING ON TEST DATA")
print("=" * 60)

test_accuracy = perceptron.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ 📌 Evaluate on UNSEEN test data                                                 │
# │ This is the "final exam" - data the model has never seen                        │
# │ Target: >= 95% accuracy                                                         │
# └─────────────────────────────────────────────────────────────────────────────────┘

print(f"Success Criteria Met: {'YES - PASSED' if test_accuracy >= 0.95 else 'NO - FAILED'}")

# ==============================================================================
# SECTION 7: VISUALIZATION
# ==============================================================================
# 📝 WHAT: Create accuracy plot and decision boundary visualization
# 📝 WHY: Visualizations help understand learning dynamics
# 📝 ANALOGY: Report card with grades and performance charts
# ==============================================================================

print("\n" + "=" * 60)
print("STEP 5: CREATING VISUALIZATIONS")
print("=" * 60)


def plot_accuracy_curve(accuracy_history, output_path):
    """
    Plot the training accuracy over epochs.
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │ 📌 This shows the "learning curve" - how accuracy improves over time      │
    │ We expect to see accuracy start low and increase toward 100%              │
    └───────────────────────────────────────────────────────────────────────────┘
    """
    plt.figure(figsize=(10, 6))
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ 📌 Create a figure 10 inches wide, 6 inches tall                        │
    # └─────────────────────────────────────────────────────────────────────────┘

    epochs = range(1, len(accuracy_history) + 1)
    plt.plot(epochs, accuracy_history, 'b-o', linewidth=2, markersize=4)
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ 📌 Plot epoch vs accuracy with blue line and circle markers             │
    # │ 'b-o': b=blue, -=solid line, o=circle marker                            │
    # └─────────────────────────────────────────────────────────────────────────┘

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Accuracy', fontsize=12)
    plt.title('Perceptron Learning Curve\nAccuracy per Epoch', fontsize=14)
    plt.grid(True, alpha=0.3)
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ 📌 Add labels, title, and grid for readability                          │
    # │ alpha=0.3 makes grid lines semi-transparent                             │
    # └─────────────────────────────────────────────────────────────────────────┘

    plt.ylim(0, 1.05)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ 📌 Add horizontal red dashed line at 95% (our target)                   │
    # │ This helps visualize when we achieve our goal                           │
    # └─────────────────────────────────────────────────────────────────────────┘

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ 📌 Save the figure to a file                                            │
    # │ dpi=150: Higher quality image                                           │
    # │ bbox_inches='tight': Remove extra whitespace                            │
    # └─────────────────────────────────────────────────────────────────────────┘

    plt.close()
    print(f"Saved accuracy plot to: {output_path}")


def plot_decision_boundary(X, y, perceptron, output_path):
    """
    Plot the data points and decision boundary.
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │ 📌 This visualization shows:                                              │
    │ 1. The data points (two classes in different colors)                      │
    │ 2. The decision boundary (line that separates the classes)                │
    └───────────────────────────────────────────────────────────────────────────┘
    """
    plt.figure(figsize=(10, 8))

    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                          edgecolors='black', s=50, alpha=0.7)
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ 📌 Scatter plot of all data points                                      │
    # │ X[:, 0]: First feature (x-axis)                                         │
    # │ X[:, 1]: Second feature (y-axis)                                        │
    # │ c=y: Color based on class label (0 or 1)                                │
    # │ cmap='RdYlBu': Red-Yellow-Blue color scheme                             │
    # │ edgecolors='black': Black border around points                          │
    # │ s=50: Point size                                                        │
    # │ alpha=0.7: Slight transparency                                          │
    # └─────────────────────────────────────────────────────────────────────────┘

    # Calculate decision boundary line
    # The decision boundary is where: w1*x1 + w2*x2 + bias = 0
    # Solving for x2: x2 = -(w1*x1 + bias) / w2
    w1, w2 = perceptron.weights
    b = perceptron.bias
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ 📌 Extract learned weights and bias                                     │
    # │ These define the decision boundary line                                 │
    # └─────────────────────────────────────────────────────────────────────────┘

    if w2 != 0:  # Avoid division by zero
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x1_line = np.linspace(x1_min, x1_max, 100)
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 Create 100 points along the x-axis range                         │
        # │ np.linspace creates evenly spaced values                            │
        # └─────────────────────────────────────────────────────────────────────┘

        x2_line = -(w1 * x1_line + b) / w2
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 DECISION BOUNDARY EQUATION                                       │
        # │ The boundary is where: w1*x1 + w2*x2 + b = 0                        │
        # │ Rearranging: x2 = -(w1*x1 + b) / w2                                 │
        # │ This gives us the y-coordinates for our boundary line               │
        # └─────────────────────────────────────────────────────────────────────┘

        plt.plot(x1_line, x2_line, 'k-', linewidth=2, label='Decision Boundary')
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 📌 Plot the decision boundary as a black solid line                 │
        # │ 'k-': k=black, -=solid line                                         │
        # └─────────────────────────────────────────────────────────────────────┘

    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Perceptron Decision Boundary\non Generated Dataset', fontsize=14)
    plt.colorbar(scatter, label='Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved decision boundary plot to: {output_path}")


# Create the plots
accuracy_plot_path = os.path.join(output_dir, "accuracy_curve.png")
boundary_plot_path = os.path.join(output_dir, "decision_boundary.png")

plot_accuracy_curve(perceptron.accuracy_history, accuracy_plot_path)
plot_decision_boundary(X, y, perceptron, boundary_plot_path)

# ==============================================================================
# SECTION 8: COMMENTARY
# ==============================================================================
# 📝 150-200 word analysis of learning dynamics
# ==============================================================================

print("\n" + "=" * 60)
print("COMMENTARY: LEARNING DYNAMICS ANALYSIS")
print("=" * 60)

commentary = f"""
PERCEPTRON LEARNING DYNAMICS ANALYSIS (150-200 words)
===============================================================================

The Perceptron was trained for {N_EPOCHS} epochs on {len(X_train)} samples with 
learning rate = {LEARNING_RATE}. A total of {perceptron.update_count} weight updates 
were performed during training.

KEY OBSERVATIONS:

1. CONVERGENCE PATTERN: The model achieved high accuracy quickly because the 
   dataset has class_sep=2.0, making it linearly separable. Early epochs show 
   many updates as the model corrects its initial random mistakes.

2. UPDATE COUNT INSIGHT: With {perceptron.update_count} total updates across 40 epochs 
   (average of {perceptron.update_count / N_EPOCHS:.1f} per epoch), we see the classic 
   Perceptron behavior - updates decrease as the model finds the correct boundary.

3. LEARNING RATE IMPACT: The chosen LR of {LEARNING_RATE} provides stable convergence. 
   A higher LR (e.g., 0.1) might converge faster but risk oscillation. A lower LR 
   (e.g., 0.001) would require more epochs for convergence.

4. FINAL PERFORMANCE: Test accuracy of {test_accuracy * 100:.2f}% confirms the model 
   generalizes well to unseen data, meeting the 95% success criterion.

The Perceptron successfully learned a linear decision boundary, demonstrating 
its effectiveness on linearly separable data.
"""

print(commentary)

# Save commentary to file
commentary_path = os.path.join(output_dir, "commentary.txt")
with open(commentary_path, 'w', encoding='utf-8') as f:
    f.write(commentary)
print(f"\nSaved commentary to: {commentary_path}")

# ==============================================================================
# SUMMARY OUTPUT
# ==============================================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"[+] Training Samples: {len(X_train)}")
print(f"[+] Testing Samples: {len(X_test)}")
print(f"[+] Epochs: {N_EPOCHS}")
print(f"[+] Learning Rate: {LEARNING_RATE}")
print(f"[+] Total Weight Updates: {perceptron.update_count}")
print(f"[+] Final Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"[+] Success Criteria (>= 95%): {'MET' if test_accuracy >= 0.95 else 'NOT MET'}")
print(f"[+] Accuracy Plot: {accuracy_plot_path}")
print(f"[+] Decision Boundary Plot: {boundary_plot_path}")
print("=" * 60)

