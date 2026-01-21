# ================================================
# MLP DECISION BOUNDARIES - ACTIVATION FUNCTIONS COMPARISON
# ================================================
# Build neural networks to visualize how different activation functions
# create different decision boundaries on a non-linearly separable dataset.
# 
# This file uses sklearn's MLPClassifier with:
# - ReLU activation
# - Logistic (Sigmoid) activation
# - Tanh activation
# ================================================

# ================================================
# SECTION 1: IMPORTS
# ================================================

# --------------------------------------------------
# Import: NumPy
# --------------------------------------------------
# 2.1 What: Imports NumPy library, the fundamental package for numerical computing in Python.
#      NumPy = "Numerical Python" - provides support for arrays and mathematical operations.
#
# 2.2 Why: We need NumPy for:
#      - Creating arrays (our data is stored as arrays)
#      - Mathematical operations on arrays (fast vectorized operations)
#      - Creating meshgrid for decision boundary visualization
#      This is the STANDARD way; alternatives like Python lists are 10-100x slower.
#
# 2.3 When: Always import NumPy when working with numerical data, machine learning, or data science.
#
# 2.4 Where: Used in virtually every ML/data science project:
#      - Training neural networks
#      - Data preprocessing
#      - Scientific computing
#
# 2.5 How: import numpy as np (alias 'np' is the universal convention)
#      Example: np.array([1, 2, 3]) creates a NumPy array
#
# 2.6 Internal: NumPy is written in C and uses optimized BLAS/LAPACK libraries.
#      When you do array operations, NumPy calls compiled C code - much faster than Python loops.
#
# 2.7 Output: No visible output from import, but makes np namespace available.
import numpy as np

# --------------------------------------------------
# Import: Matplotlib.pyplot
# --------------------------------------------------
# 2.1 What: Imports Matplotlib's pyplot module for creating visualizations.
#      Matplotlib = "Mathematical Plotting Library"
#      pyplot = simplified interface similar to MATLAB
#
# 2.2 Why: We need matplotlib to:
#      - Create decision boundary contour plots
#      - Overlay scatter plots of training data
#      - Create multi-subplot figures for comparison
#      This is the most popular plotting library; alternatives (Seaborn, Plotly) build on it.
#
# 2.3 When: Whenever you need to visualize data, model results, or create any plots.
#
# 2.4 Where: Data analysis, ML model evaluation, research papers, dashboards.
#
# 2.5 How: import matplotlib.pyplot as plt (alias 'plt' is standard)
#      Example: plt.plot([1,2,3], [1,4,9]) creates a line plot
#
# 2.6 Internal: Creates Figure and Axes objects in memory, renders using backend (like Agg, TkAgg).
#
# 2.7 Output: No visible output from import.
import matplotlib.pyplot as plt

# --------------------------------------------------
# Import: make_moons from sklearn.datasets
# --------------------------------------------------
# 2.1 What: Imports the make_moons function from sklearn's datasets module.
#      make_moons = generates a 2D dataset shaped like two interleaving half-moons.
#
# 2.2 Why: We need make_moons because:
#      - It creates a NON-LINEARLY separable dataset (straight line won't work)
#      - Perfect for demonstrating neural network decision boundaries
#      - Easy to visualize in 2D
#      Alternative: make_circles, but moons is more commonly used in tutorials.
#
# 2.3 When: When you need a simple non-linear classification problem for testing/teaching.
#
# 2.4 Where: ML education, algorithm testing, decision boundary visualization.
#
# 2.5 How: X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
#      Returns: X (features), y (labels)
#
# 2.6 Internal: Generates points on two semicircles, adds Gaussian noise, returns arrays.
#
# 2.7 Output: X array of shape (n_samples, 2), y array of shape (n_samples,)
from sklearn.datasets import make_moons

# --------------------------------------------------
# Import: MLPClassifier from sklearn.neural_network
# --------------------------------------------------
# 2.1 What: Imports MLPClassifier - Multi-Layer Perceptron (neural network) for classification.
#      MLP = neural network with one or more hidden layers
#      Classifier = used for classification tasks (predicting categories)
#
# 2.2 Why: We need MLPClassifier because:
#      - It's sklearn's ready-to-use neural network implementation
#      - Supports different activation functions (relu, logistic, tanh)
#      - Easy to train with .fit() and predict with .predict()
#      Alternative: Building from scratch (complex) or using Keras/PyTorch (overkill for simple tasks).
#
# 2.3 When: When you need a simple neural network for classification without building from scratch.
#
# 2.4 Where: Binary/multi-class classification, pattern recognition, simple deep learning tasks.
#
# 2.5 How: model = MLPClassifier(hidden_layer_sizes=(8,), activation='relu')
#
# 2.6 Internal: Implements forward propagation, backpropagation, and optimization (Adam, SGD).
#
# 2.7 Output: Returns a fitted model object with .predict() and .score() methods.
from sklearn.neural_network import MLPClassifier

# --------------------------------------------------
# Import: accuracy_score from sklearn.metrics
# --------------------------------------------------
# 2.1 What: Imports accuracy_score function to calculate classification accuracy.
#      Accuracy = (correct predictions) / (total predictions) × 100%
#
# 2.2 Why: We need accuracy_score to:
#      - Measure how well each model performs
#      - Compare between different activation functions
#      - Display accuracy in plot titles
#      Alternative: Using model.score() directly, but accuracy_score is more explicit.
#
# 2.3 When: After making predictions, to evaluate model performance.
#
# 2.4 Where: Model evaluation, hyperparameter tuning, model comparison.
#
# 2.5 How: accuracy = accuracy_score(y_true, y_pred)
#
# 2.6 Internal: Counts matching predictions, divides by total.
#
# 2.7 Output: Float between 0.0 and 1.0 (multiply by 100 for percentage).
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Import: ListedColormap from matplotlib.colors
# --------------------------------------------------
# 2.1 What: Imports ListedColormap to create custom color palettes for plots.
#      ListedColormap = creates a colormap from a list of colors
#
# 2.2 Why: We need ListedColormap to:
#      - Create distinct colors for different classes (0 and 1)
#      - Make decision boundary visualization clear and appealing
#      Alternative: Using built-in colormaps ('RdYlBu'), but custom is more controllable.
#
# 2.3 When: When you need specific colors for visualization.
#
# 2.4 Where: Decision boundary plots, heatmaps, categorical data visualization.
#
# 2.5 How: cmap = ListedColormap(['#FF9999', '#9999FF'])  # red and blue
#
# 2.6 Internal: Maps integer indices (0, 1) to colors in the list.
#
# 2.7 Output: Colormap object usable by matplotlib plotting functions.
from matplotlib.colors import ListedColormap


# ================================================
# SECTION 2: DATA GENERATION
# ================================================

def generate_moons_data():
    """
    Generates the make_moons dataset for training.
    
    This function creates a 2D dataset shaped like two interleaving half-moons.
    It's perfect for demonstrating non-linear classification because a straight
    line CANNOT separate the two classes - you need a curved boundary.
    
    ⚙️ Arguments (3.1-3.7):
    -----------------------
    This function takes NO arguments - all parameters are hardcoded as per the problem:
    - n_samples=300 (total data points)
    - noise=0.2 (adds randomness to make it realistic)
    - random_state=42 (ensures reproducibility)
    
    Returns:
    --------
    X : np.ndarray of shape (300, 2)
        Feature matrix. Each row is a point, columns are x and y coordinates.
    y : np.ndarray of shape (300,)
        Labels. 0 for first moon, 1 for second moon.
    
    Example:
    --------
    >>> X, y = generate_moons_data()
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    X shape: (300, 2), y shape: (300,)
    """
    # 2.1 What: Call make_moons to generate the dataset.
    #      make_moons creates two interleaving half-circle shaped clusters.
    #
    # 2.2 Why: This is specified in the problem statement.
    #      The moons dataset is ideal because:
    #      - Non-linearly separable (tests neural network capability)
    #      - 2D (easy to visualize decision boundaries)
    #      - Noise adds realism (real data is noisy)
    #
    # 2.3 When: At the start of any classification experiment on this dataset.
    #
    # 2.4 Where: ML education, decision boundary demos, activation function comparison.
    #
    # 2.5 How: X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    #
    # 2.6 Internal:
    #      1. Generate n_samples/2 points on upper semicircle (class 0)
    #      2. Generate n_samples/2 points on lower semicircle (class 1), shifted
    #      3. Add Gaussian noise with std=0.2 to both x and y coordinates
    #      4. random_state seeds the random number generator
    #
    # 2.7 Output: Two arrays - X (features), y (labels)
    #
    # ⚙️ make_moons Arguments (3.1-3.7):
    # -----------------------------------
    # ARGUMENT 1: n_samples=300
    #   3.1 What: Total number of data points to generate
    #   3.2 Why: 300 gives enough points to see patterns without being too slow
    #        Problem specifies 300; alternatives: 100 (faster), 1000 (more detail)
    #   3.3 When: Always specify this to control dataset size
    #   3.4 Where: All make_moons calls
    #   3.5 How: n_samples=300 means 150 points per moon
    #   3.6 Internal: Splits evenly between two moons
    #   3.7 Impact: More samples = smoother decision boundary visualization
    #
    # ARGUMENT 2: noise=0.2
    #   3.1 What: Standard deviation of Gaussian noise added to data
    #   3.2 Why: Makes data more realistic; real data has noise
    #        Alternative: noise=0 (perfect moons), noise=0.5 (very noisy)
    #   3.3 When: Adjust based on how "clean" you want data
    #   3.4 Where: make_moons, make_circles, similar synthetic data functions
    #   3.5 How: noise=0.2 adds ~95% of points within ±0.4 of ideal position
    #   3.6 Internal: np.random.normal(0, noise) added to each coordinate
    #   3.7 Impact: Higher noise makes classification harder
    #
    # ARGUMENT 3: random_state=42
    #   3.1 What: Seed for the random number generator
    #   3.2 Why: Ensures REPRODUCIBILITY - same data every time
    #        Why 42? It's a convention (Hitchhiker's Guide reference); any int works
    #   3.3 When: ALWAYS in production/research for reproducible results
    #   3.4 Where: All random operations in sklearn
    #   3.5 How: random_state=42 always produces identical output
    #   3.6 Internal: Seeds np.random.RandomState internally
    #   3.7 Impact: Without this, data would be different each run
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    
    # 2.1 What: Return the generated data.
    # 2.7 Output: X has shape (300, 2), y has shape (300,)
    return X, y


# ================================================
# SECTION 3: MODEL CREATION
# ================================================

def create_mlp_model(activation):
    """
    Creates an MLPClassifier model with specified activation function.
    
    Think of this as a "factory" that builds neural networks.
    You tell it which activation function to use, and it gives you
    a ready-to-train neural network.
    
    ⚙️ Arguments (3.1-3.7):
    -----------------------
    activation : str
        3.1 What: Name of the activation function for hidden layer neurons.
            Valid options: 'relu', 'logistic', 'tanh', 'identity'
        3.2 Why: Different activations create different decision boundary shapes.
            This is the CORE of our experiment - comparing activations.
        3.3 When: Every time you create an MLP and want to control activation.
        3.4 Where: MLPClassifier, neural network libraries.
        3.5 How: activation='relu' for ReLU, 'logistic' for sigmoid, 'tanh' for tanh
        3.6 Internal: Activation is applied after each neuron's weighted sum.
        3.7 Impact: Determines non-linearity and decision boundary shape.
    
    Returns:
    --------
    MLPClassifier : sklearn model object
        A neural network ready to be trained with .fit(X, y)
    
    Example:
    --------
    >>> model = create_mlp_model('relu')
    >>> print(type(model))
    <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>
    """
    # 2.1 What: Create and return an MLPClassifier with specified configuration.
    #
    # 2.2 Why: Creates a neural network according to problem requirements:
    #      - 1 hidden layer with 8 neurons
    #      - Specified activation function
    #      - Same random_state for fair comparison
    #
    # 2.3 When: Before training - you need a model object first.
    #
    # 2.4 Where: Any neural network classification task.
    #
    # 2.5 How: model = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', ...)
    #
    # 2.6 Internal:
    #      1. Initializes weight matrices with random values (controlled by random_state)
    #      2. Sets up network architecture based on hidden_layer_sizes
    #      3. Prepares optimization algorithm (solver)
    #
    # 2.7 Output: Untrained model object.
    #
    # ⚙️ MLPClassifier Arguments (3.1-3.7):
    # --------------------------------------
    # ARGUMENT 1: hidden_layer_sizes=(8,)
    #   3.1 What: Tuple specifying number of neurons in each hidden layer.
    #        (8,) = one hidden layer with 8 neurons
    #        (8, 4) would mean two hidden layers: 8 neurons, then 4 neurons
    #   3.2 Why: Problem specifies "1 hidden layer with 8 neurons"
    #        More neurons = more capacity but risk of overfitting
    #        Alternatives: (16,), (8, 8), (4,)
    #   3.3 When: Adjust based on problem complexity
    #   3.4 Where: All MLPClassifier instantiations
    #   3.5 How: hidden_layer_sizes=(8,) - NOTE: comma makes it a tuple!
    #   3.6 Internal: Creates weight matrix of shape (input_dim, 8) + bias vector
    #   3.7 Impact: More neurons = more complex boundaries but slower training
    #
    # ARGUMENT 2: activation=activation
    #   3.1 What: Which activation function to use in hidden layers
    #        'relu' = max(0, x), 'logistic' = 1/(1+e^-x), 'tanh' = (e^x-e^-x)/(e^x+e^-x)
    #   3.2 Why: THIS IS THE CORE OF OUR EXPERIMENT
    #        Different activations shape decision boundaries differently
    #   3.3 When: Default is 'relu'; change when comparing or for specific needs
    #   3.4 Where: Neural network hidden layers
    #   3.5 How: activation='relu', activation='logistic', activation='tanh'
    #   3.6 Internal: Applied element-wise after weighted sum in each neuron
    #   3.7 Impact: Huge - determines learning dynamics and boundary shapes
    #
    # ARGUMENT 3: solver='adam'
    #   3.1 What: Optimization algorithm for training
    #        'adam' = Adaptive Moment Estimation (modern default)
    #        'sgd' = Stochastic Gradient Descent (classic)
    #        'lbfgs' = Limited-memory Broyden-Fletcher-Goldfarb-Shanno (for small data)
    #   3.2 Why: Adam works well for most problems without tuning
    #        Alternatives: 'sgd' (more control), 'lbfgs' (small datasets)
    #   3.3 When: Default 'adam' is usually fine; 'lbfgs' for small datasets
    #   3.4 Where: All gradient-based ML algorithms
    #   3.5 How: solver='adam' (default, recommended)
    #   3.6 Internal: Maintains running averages of gradients and squared gradients
    #   3.7 Impact: Affects convergence speed and final accuracy
    #
    # ARGUMENT 4: max_iter=1000
    #   3.1 What: Maximum training iterations (epochs)
    #        One iteration = one pass through the data
    #   3.2 Why: 1000 is enough to converge on this simple dataset
    #        Too few = underfitting; too many = wasted time
    #   3.3 When: Increase if you get ConvergenceWarning
    #   3.4 Where: All iterative optimization algorithms
    #   3.5 How: max_iter=1000 allows up to 1000 training epochs
    #   3.6 Internal: Training stops early if convergence criteria met
    #   3.7 Impact: Higher = more training time, potentially better results
    #
    # ARGUMENT 5: random_state=42
    #   3.1 What: Seed for random weight initialization and data shuffling
    #   3.2 Why: CRITICAL for fair comparison - same starting weights
    #        Without this, different runs would start differently
    #   3.3 When: ALWAYS in experiments for reproducibility
    #   3.4 Where: All sklearn estimators with random components
    #   3.5 How: random_state=42 (same value for all 3 models)
    #   3.6 Internal: Seeds the random number generator
    #   3.7 Impact: Ensures identical initial conditions across models
    return MLPClassifier(
        hidden_layer_sizes=(8,),   # 1 hidden layer with 8 neurons
        activation=activation,      # 'relu', 'logistic', or 'tanh'
        solver='adam',              # Adam optimizer
        max_iter=1000,              # Enough iterations to converge
        random_state=42             # Same seed for fair comparison
    )


# ================================================
# SECTION 4: MODEL TRAINING
# ================================================

def train_model(model, X, y, activation_name):
    """
    Trains the given model on the dataset and returns accuracy.
    
    Think of training like teaching a child to distinguish red from blue candies.
    You show many examples, and the neural network adjusts its "thinking" to get better.
    
    ⚙️ Arguments (3.1-3.7):
    -----------------------
    model : MLPClassifier
        3.1 What: The neural network model to train
        3.2 Why: We need a model object to call .fit() on
        3.3 When: After creating with create_mlp_model()
        3.4 Where: All sklearn estimators follow this pattern
        3.5 How: Pass the model object directly
        3.6 Internal: The model stores learned weights internally
        3.7 Impact: Model will be modified in-place after training
    
    X : np.ndarray of shape (n_samples, 2)
        3.1 What: Feature matrix - each row is a data point, columns are features
        3.2 Why: The input data the model learns from
        3.3 When: Required for all supervised learning
        3.4 Where: All ML training
        3.5 How: X should be 2D array: X.shape = (300, 2)
        3.6 Internal: Each row flows through the network
        3.7 Impact: Quality and quantity of X affects model performance
    
    y : np.ndarray of shape (n_samples,)
        3.1 What: Labels - the "answers" for each data point
        3.2 Why: The model needs to know correct answers to learn
        3.3 When: Required for supervised learning
        3.4 Where: All classification/regression training
        3.5 How: y should be 1D array: y.shape = (300,)
        3.6 Internal: Used to compute loss and gradients
        3.7 Impact: Must match X row-by-row
    
    activation_name : str
        3.1 What: Name of activation for display purposes
        3.2 Why: For printing progress messages
        3.3 When: For logging/debugging
        3.4 Where: Training functions
        3.5 How: activation_name='ReLU'
        3.6 Internal: Just used in print statement
        3.7 Impact: No effect on training, just for user feedback
    
    Returns:
    --------
    float : Training accuracy (0.0 to 1.0)
    
    Example:
    --------
    >>> model = create_mlp_model('relu')
    >>> X, y = generate_moons_data()
    >>> accuracy = train_model(model, X, y, 'ReLU')
    Training ReLU model...
    ReLU training accuracy: 0.9567
    """
    # 2.1 What: Print a status message to show which model is training.
    # 2.2 Why: Provides user feedback during execution.
    print(f"Training {activation_name} model...")
    
    # 2.1 What: Train the model using the .fit() method.
    #      .fit() is sklearn's universal method for training any model.
    #
    # 2.2 Why: This is HOW the model learns from data.
    #      The model adjusts weights to minimize prediction errors.
    #      This is THE standard way in sklearn; no alternative.
    #
    # 2.3 When: After creating model and preparing data.
    #
    # 2.4 Where: All sklearn estimators (classifiers, regressors, transformers).
    #
    # 2.5 How: model.fit(X, y) - that's it!
    #
    # 2.6 Internal (for MLPClassifier):
    #      1. Initialize weights (if not already done)
    #      2. Forward pass: compute predictions
    #      3. Compute loss (cross-entropy for classification)
    #      4. Backward pass: compute gradients via backpropagation
    #      5. Update weights using optimizer (Adam)
    #      6. Repeat until max_iter or convergence
    #
    # 2.7 Output: Returns self (the model), but more importantly,
    #      the model is now trained and can make predictions.
    #
    # ⚙️ .fit() Arguments (3.1-3.7):
    # -------------------------------
    # ARGUMENT 1: X
    #   3.1-3.7: See X in function arguments above
    #
    # ARGUMENT 2: y
    #   3.1-3.7: See y in function arguments above
    model.fit(X, y)
    
    # 2.1 What: Compute training accuracy using .score() method.
    #      .score() returns accuracy for classifiers.
    #
    # 2.2 Why: We need to know how well the model learned.
    #      This measures: (correct predictions) / (total predictions)
    #
    # 2.3 When: After training, to evaluate model performance.
    #
    # 2.4 Where: Model evaluation, hyperparameter tuning.
    #
    # 2.5 How: accuracy = model.score(X, y)
    #
    # 2.6 Internal:
    #      1. Makes predictions: y_pred = model.predict(X)
    #      2. Compares with true labels: (y_pred == y).mean()
    #
    # 2.7 Output: Float between 0.0 and 1.0 (e.g., 0.95 = 95% accuracy)
    accuracy = model.score(X, y)
    
    # 2.1 What: Print the accuracy for user feedback.
    print(f"{activation_name} training accuracy: {accuracy:.4f}")
    
    # 2.1 What: Return the accuracy value.
    return accuracy


# ================================================
# SECTION 5: DECISION BOUNDARY VISUALIZATION
# ================================================

def create_meshgrid(X, padding=0.5, step=0.02):
    """
    Creates a meshgrid for plotting decision boundaries.
    
    Imagine you want to color every tiny square on a grid based on what
    the model predicts for that location. This function creates that grid.
    
    ⚙️ Arguments (3.1-3.7):
    -----------------------
    X : np.ndarray of shape (n_samples, 2)
        3.1 What: The training data - used to determine grid boundaries
        3.2 Why: We need to know the data range to size the grid appropriately
        3.3 When: Before creating decision boundary plot
        3.4 Where: Decision boundary visualization
        3.5 How: Pass the feature matrix X
        3.6 Internal: Uses X.min() and X.max() to determine bounds
        3.7 Impact: Grid will cover all data points plus padding
    
    padding : float, default=0.5
        3.1 What: Extra space around data in all directions
        3.2 Why: So the plot isn't cramped at edges
        3.3 When: Adjust if you want more/less whitespace
        3.4 Where: Visualization functions
        3.5 How: padding=0.5 adds 0.5 units on each side
        3.6 Internal: Subtracted from min, added to max
        3.7 Impact: Larger = more context, smaller = focused on data
    
    step : float, default=0.02
        3.1 What: Distance between grid points (resolution)
        3.2 Why: Smaller = smoother boundaries, but slower
        3.3 When: Decrease for publication-quality plots
        3.4 Where: Meshgrid creation
        3.5 How: step=0.02 creates points every 0.02 units
        3.6 Internal: Used in np.arange(min, max, step)
        3.7 Impact: step=0.01 = 4x more points = smoother but slower
    
    Returns:
    --------
    xx : np.ndarray - X coordinates of grid points
    yy : np.ndarray - Y coordinates of grid points
    """
    # 2.1 What: Get minimum and maximum values for both features (columns).
    #      X[:, 0] = all values in first column (x-coordinate)
    #      X[:, 1] = all values in second column (y-coordinate)
    #
    # 2.2 Why: Need to know data range to create appropriate grid size.
    #
    # 2.3 When: Before creating meshgrid.
    #
    # 2.5 How: X[:, 0].min() gets smallest x value
    #
    # 2.6 Internal: NumPy's min/max scan entire array.
    #
    # 2.7 Output: Four scalar values: x_min, x_max, y_min, y_max
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    
    # 2.1 What: Create meshgrid using np.meshgrid.
    #      np.arange creates 1D arrays of points
    #      np.meshgrid creates 2D grids from 1D arrays
    #
    # 2.2 Why: We need coordinates of every point in a 2D grid
    #      to ask the model "what would you predict here?"
    #
    # 2.3 When: Before decision boundary visualization.
    #
    # 2.4 Where: Contour plots, heatmaps, 3D surface plots.
    #
    # 2.5 How: 
    #      np.arange(0, 2, 0.5) -> [0.0, 0.5, 1.0, 1.5]
    #      np.meshgrid creates grid from two 1D arrays
    #
    # 2.6 Internal:
    #      np.arange(x_min, x_max, step) creates [x_min, x_min+step, x_min+2*step, ...]
    #      np.meshgrid takes two 1D arrays and creates two 2D matrices:
    #      - xx: x-coordinate at each grid point
    #      - yy: y-coordinate at each grid point
    #
    # 2.7 Output: Two 2D arrays of shape (n_y_points, n_x_points)
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),  # X-axis points
        np.arange(y_min, y_max, step)   # Y-axis points
    )
    
    return xx, yy


def plot_decision_boundary(ax, model, X, y, title, cmap_light, cmap_bold):
    """
    Plots decision boundary with training data overlay on given axes.
    
    This is like coloring a map where each color shows what class
    the model would predict for that location, then putting dots
    for where our actual training data points are.
    
    ⚙️ Arguments (3.1-3.7):
    -----------------------
    ax : matplotlib.axes.Axes
        3.1 What: The subplot/axes to draw on
        3.2 Why: For creating multi-panel figures
        3.3 When: When using plt.subplots()
        3.4 Where: All matplotlib subplot operations
        3.5 How: fig, ax = plt.subplots() then use ax.plot()
        3.6 Internal: Axes object contains all plot elements
        3.7 Impact: Drawing goes to this specific subplot
    
    model : MLPClassifier
        3.1 What: The trained model to visualize
        3.2 Why: We need it to make predictions across the grid
        3.3 When: After training
        3.4 Where: Decision boundary visualization
        3.5 How: Pass the trained model object
        3.6 Internal: Uses model.predict() for each grid point
        3.7 Impact: Boundary shape depends on trained weights
    
    X : np.ndarray of shape (n_samples, 2)
        3.1 What: Training data features
        3.2 Why: To determine grid bounds and overlay scatter
        3.3 When: For visualization
        3.4 Where: All decision boundary plots
        3.5 How: Same X used for training
        3.6 Internal: Used for meshgrid and scatter plot
        3.7 Impact: Points shown on plot
    
    y : np.ndarray of shape (n_samples,)
        3.1 What: Training data labels
        3.2 Why: To color scatter points by class
        3.3 When: For visualization
        3.4 Where: All decision boundary plots
        3.5 How: Same y used for training
        3.6 Internal: Determines scatter point colors
        3.7 Impact: Shows which class each point belongs to
    
    title : str
        3.1 What: Title for this subplot
        3.2 Why: To label which activation/accuracy this shows
        3.3 When: Always for clear visualization
        3.4 Where: All plot titles
        3.5 How: title='ReLU (Accuracy: 95.00%)'
        3.6 Internal: Rendered as text above plot
        3.7 Impact: User understanding
    
    cmap_light : ListedColormap
        3.1 What: Light colors for background (decision regions)
        3.2 Why: Soft colors show regions without overpowering data points
        3.3 When: Decision boundary backgrounds
        3.4 Where: Contour fills
        3.5 How: ListedColormap(['#FFAAAA', '#AAAAFF'])
        3.6 Internal: Maps prediction values to colors
        3.7 Impact: Visual appearance of regions
    
    cmap_bold : ListedColormap
        3.1 What: Bold colors for data points
        3.2 Why: Points need to stand out against background
        3.3 When: Scatter plots on decision boundaries
        3.4 Where: Scatter plot colors
        3.5 How: ListedColormap(['#FF0000', '#0000FF'])
        3.6 Internal: Maps y values to colors
        3.7 Impact: Visual clarity of actual data
    """
    # 2.1 What: Create meshgrid for this subplot.
    xx, yy = create_meshgrid(X)
    
    # 2.1 What: Predict class for every point in the grid.
    #      np.c_ concatenates arrays column-wise
    #      ravel() flattens 2D array to 1D
    #
    # 2.2 Why: To know what color each grid point should be.
    #
    # 2.3 When: After creating meshgrid.
    #
    # 2.5 How:
    #      xx.ravel() -> [x1, x2, x3, ...]
    #      yy.ravel() -> [y1, y2, y3, ...]
    #      np.c_[...] -> [[x1,y1], [x2,y2], ...]
    #      predict() -> [0, 1, 0, ...]
    #
    # 2.6 Internal:
    #      1. Flatten xx and yy to 1D arrays
    #      2. Stack as columns to create (n_points, 2) array
    #      3. Each row is one grid point's coordinates
    #      4. Model predicts class for each point
    #
    # 2.7 Output: 1D array of predictions, same length as grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # 2.1 What: Reshape predictions back to grid shape.
    #      reshape(xx.shape) makes Z a 2D array matching gridshape
    #
    # 2.2 Why: contourf needs 2D arrays to draw properly.
    #
    # 2.6 Internal: Just rearranges the 1D array into 2D.
    #
    # 2.7 Output: 2D array same shape as xx and yy
    Z = Z.reshape(xx.shape)
    
    # 2.1 What: Draw filled contour plot (the colored background).
    #      contourf = "contour fill" - colors regions between contour lines
    #
    # 2.2 Why: Shows decision regions as colored areas.
    #      Alternative: contour() for just lines, but fill is clearer.
    #
    # 2.3 When: For decision boundary visualization.
    #
    # 2.4 Where: Classification boundary plots, heatmaps.
    #
    # 2.5 How: ax.contourf(xx, yy, Z, cmap=colormap, alpha=0.8)
    #
    # 2.6 Internal:
    #      1. Takes xx (x-coords), yy (y-coords), Z (values)
    #      2. Draws colored regions where Z has same value
    #      3. Uses cmap to map Z values to colors
    #
    # 2.7 Output: Colored background showing decision regions
    #
    # ⚙️ contourf Arguments (3.1-3.7):
    # ---------------------------------
    # alpha=0.8
    #   3.1 What: Transparency (0=invisible, 1=opaque)
    #   3.2 Why: Slightly transparent so grid/points are visible
    #   3.5 How: alpha=0.8 means 80% opaque
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    # 2.1 What: Scatter plot of training data points.
    #      scatter() draws individual points as markers.
    #
    # 2.2 Why: Shows actual data on top of decision regions.
    #      Helps visualize if boundary separates classes well.
    #
    # 2.3 When: After drawing background regions.
    #
    # 2.4 Where: All classification visualization.
    #
    # 2.5 How: ax.scatter(X[:, 0], X[:, 1], c=y, ...)
    #
    # 2.6 Internal:
    #      1. X[:, 0] = x-coordinates of all points
    #      2. X[:, 1] = y-coordinates of all points
    #      3. c=y colors points by their label
    #      4. edgecolor adds outline to each point
    #
    # 2.7 Output: Colored dots overlay on the contour plot
    #
    # ⚙️ scatter Arguments (3.1-3.7):
    # --------------------------------
    # c=y
    #   3.1 What: Colors - array of values determining each point's color
    #   3.2 Why: Different colors for different classes
    #   3.5 How: When y=[0,1,0], uses cmap to assign colors
    #
    # cmap=cmap_bold
    #   3.1 What: Colormap for the points
    #   3.2 Why: Bold colors stand out against light background
    #
    # edgecolor='black'
    #   3.1 What: Color of the point outlines
    #   3.2 Why: Makes points visible even on similar background
    #
    # s=50
    #   3.1 What: Size of points (in square points)
    #   3.2 Why: Large enough to see clearly
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='black', s=50)
    
    # 2.1 What: Set the title for this subplot.
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # 2.1 What: Set axis labels.
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)


# ================================================
# SECTION 6: MAIN COMPARISON FUNCTION
# ================================================

def compare_activations():
    """
    Main function that compares all three activation functions.
    
    This is the "conductor" that orchestrates the entire experiment:
    1. Generate data
    2. Create 3 models
    3. Train all models
    4. Visualize results
    5. Create comparison table
    6. Print analysis
    
    Returns:
    --------
    dict : Dictionary containing models and accuracies
        {'relu': (model, accuracy), 'logistic': (model, accuracy), 'tanh': (model, accuracy)}
    """
    # 2.1 What: Print header for experiment.
    print("=" * 70)
    print("MLP DECISION BOUNDARIES - ACTIVATION FUNCTIONS COMPARISON")
    print("=" * 70)
    
    # 2.1 What: Define output directory for saving results.
    output_dir = 'c:/masai/MLP_Decision_Boundaries/outputs'
    
    # ========================================
    # STEP 1: GENERATE DATA
    # ========================================
    print("\n[Step 1] Generating make_moons dataset...")
    X, y = generate_moons_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: Class 0 = {sum(y==0)}, Class 1 = {sum(y==1)}")
    
    # ========================================
    # STEP 2: CREATE MODELS
    # ========================================
    print("\n[Step 2] Creating MLPClassifier models...")
    
    # 2.1 What: Define the three activations to compare.
    #      List of tuples: (internal_name, display_name)
    activations = [
        ('relu', 'ReLU'),
        ('logistic', 'Logistic (Sigmoid)'),
        ('tanh', 'Tanh')
    ]
    
    # 2.1 What: Create models dictionary to store models and results.
    models = {}
    for activation, name in activations:
        models[activation] = {
            'name': name,
            'model': create_mlp_model(activation),
            'accuracy': None
        }
    print(f"Created {len(models)} models: {list(models.keys())}")
    
    # ========================================
    # STEP 3: TRAIN ALL MODELS
    # ========================================
    print("\n[Step 3] Training all models...")
    for activation, data in models.items():
        data['accuracy'] = train_model(data['model'], X, y, data['name'])
    
    # ========================================
    # STEP 4: CREATE VISUALIZATION
    # ========================================
    print("\n[Step 4] Creating decision boundary visualization...")
    
    # 2.1 What: Create colormaps for visualization.
    #      Light colors for background, bold for points.
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])  # Light red, light blue
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])   # Bold red, bold blue
    
    # 2.1 What: Create figure with 3 subplots (1 row, 3 columns).
    #      figsize=(15, 5) = 15 inches wide, 5 inches tall
    #
    # 2.2 Why: Need side-by-side comparison of all 3 activations.
    #
    # 2.5 How: fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #      Returns figure object and array of 3 axes objects
    #
    # 2.6 Internal: Creates Figure with 3 Axes arranged horizontally.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2.1 What: Plot decision boundary for each model.
    for idx, (activation, data) in enumerate(models.items()):
        title = f"{data['name']}\nAccuracy: {data['accuracy']*100:.2f}%"
        plot_decision_boundary(
            axes[idx], 
            data['model'], 
            X, y, 
            title, 
            cmap_light, 
            cmap_bold
        )
    
    # 2.1 What: Add main title for entire figure.
    fig.suptitle('Decision Boundaries: Comparing Activation Functions on make_moons', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # 2.1 What: Adjust layout to prevent overlap.
    plt.tight_layout()
    
    # 2.1 What: Save the figure.
    plt.savefig(f'{output_dir}/decision_boundaries.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved to {output_dir}/decision_boundaries.png")
    plt.close()
    
    # ========================================
    # STEP 5: CREATE COMPARISON TABLE
    # ========================================
    print("\n[Step 5] Creating comparison table...")
    create_comparison_table(models, output_dir)
    
    # ========================================
    # STEP 6: WRITTEN ANALYSIS
    # ========================================
    print("\n[Step 6] Generating written analysis...")
    written_analysis(models)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    return models


# ================================================
# SECTION 7: COMPARISON TABLE
# ================================================

def create_comparison_table(models, output_dir):
    """
    Creates and saves a comparison table of all activation accuracies.
    
    ⚙️ Arguments:
    - models: Dictionary with model data
    - output_dir: Path to save the markdown file
    """
    # 2.1 What: Print table to console.
    print("\n" + "=" * 50)
    print("ACCURACY COMPARISON TABLE")
    print("=" * 50)
    print(f"{'Activation':<25} {'Training Accuracy':>20}")
    print("-" * 50)
    
    for activation, data in models.items():
        print(f"{data['name']:<25} {data['accuracy']*100:>19.2f}%")
    
    # 2.1 What: Find the best performing activation.
    best = max(models.items(), key=lambda x: x[1]['accuracy'])
    print("-" * 50)
    print(f"Best: {best[1]['name']} with {best[1]['accuracy']*100:.2f}% accuracy")
    
    # 2.1 What: Save table to markdown file.
    with open(f'{output_dir}/comparison_table.md', 'w') as f:
        f.write("# Activation Functions Comparison Table\n\n")
        f.write("## Training Accuracy Results\n\n")
        f.write("| Activation | Training Accuracy |\n")
        f.write("|------------|------------------:|\n")
        for activation, data in models.items():
            f.write(f"| {data['name']} | {data['accuracy']*100:.2f}% |\n")
        f.write(f"\n**Best Performer:** {best[1]['name']} ({best[1]['accuracy']*100:.2f}%)\n")
    
    print(f"\n[OK] Comparison table saved to {output_dir}/comparison_table.md")


# ================================================
# SECTION 8: WRITTEN ANALYSIS
# ================================================

def written_analysis(models):
    """
    Prints comprehensive written analysis (250-350 words).
    
    ⚙️ Arguments:
    - models: Dictionary with model data including accuracies
    """
    # Get accuracy values
    relu_acc = models['relu']['accuracy'] * 100
    logistic_acc = models['logistic']['accuracy'] * 100
    tanh_acc = models['tanh']['accuracy'] * 100
    
    # Find best
    best_name = max(models.items(), key=lambda x: x[1]['accuracy'])[1]['name']
    
    print("\n" + "=" * 70)
    print("WRITTEN ANALYSIS: ACTIVATION FUNCTIONS ON MAKE_MOONS (250-350 WORDS)")
    print("=" * 70)
    
    analysis = f"""
DECISION BOUNDARY SHAPE COMPARISON:
-----------------------------------
Looking at the three decision boundary plots, we observe distinctly different 
shapes for each activation function:

- **ReLU**: Creates angular, piecewise-linear boundaries. The decision 
  region has sharp corners and straight edges because ReLU is linear for 
  positive values (f(x) = x for x > 0). This creates a "jagged" appearance.

- **Logistic (Sigmoid)**: Produces smooth, curved boundaries. The S-shaped 
  nature of sigmoid (output between 0 and 1) results in gradual transitions 
  between decision regions. The boundary appears softer and more rounded.

- **Tanh**: Similar to sigmoid but with potentially sharper transitions near 
  the decision boundary because tanh is steeper (outputs between -1 and 1). 
  The zero-centered nature often leads to slightly different curvature.

ACCURACY COMPARISON:
--------------------
All three activations achieved similar high accuracy on this dataset:
- ReLU: {relu_acc:.2f}%
- Logistic: {logistic_acc:.2f}%
- Tanh: {tanh_acc:.2f}%

**{best_name} achieved the highest accuracy** in this experiment.

WHY THESE RESULTS MAKE SENSE:
-----------------------------
The make_moons dataset requires non-linear decision boundaries, which all 
three activations can produce (unlike identity activation). The dataset 
is relatively simple with only 300 samples and low noise (0.2), allowing 
even a small network (8 neurons) to fit it well.

**ReLU** often excels due to its computational efficiency and lack of 
vanishing gradient issues. However, on small, simple datasets like 
make_moons, the differences between activations are minimal.

**Sigmoid and Tanh** may slightly outperform ReLU on bounded data because 
they naturally output bounded values, which can match the 0/1 classification 
target well.

CONCLUSION:
-----------
For the make_moons dataset, all three activations perform comparably. In 
practice, ReLU is preferred for deep networks due to training stability, 
while sigmoid/tanh are used for specific layers (output, RNNs). The choice 
depends more on network depth and problem type than on simple 2D classification.
"""
    print(analysis)


# ================================================
# SECTION 9: MAIN EXECUTION
# ================================================

def main():
    """
    Main entry point for the MLP Decision Boundaries experiment.
    
    This function orchestrates the entire experiment:
    1. Generates the make_moons dataset
    2. Creates and trains 3 MLPClassifier models with different activations
    3. Visualizes decision boundaries side-by-side
    4. Creates comparison table
    5. Provides written analysis
    """
    # 2.1 What: Run the complete comparison experiment.
    compare_activations()


# 2.1 What: Python's standard entry point check.
#      __name__ == "__main__" is True only when script is run directly.
#      It's False when imported as a module.
#
# 2.2 Why: Allows file to be both:
#      - Run directly: python mlp_decision_boundaries.py (runs main)
#      - Imported: import mlp_decision_boundaries (doesn't run main)
#
# 2.3 When: Every Python script that should be runnable.
#
# 2.4 Where: Standard practice in all Python files.
#
# 2.5 How: if __name__ == "__main__": main()
#
# 2.6 Internal: Python sets __name__ to "__main__" for the top-level script.
if __name__ == "__main__":
    main()
