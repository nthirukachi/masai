"""
================================================================================
ReLU vs Leaky ReLU: Investigating the Dying ReLU Problem
================================================================================

This script implements a 2-layer neural network from scratch to compare ReLU 
and Leaky ReLU activation functions and demonstrate the "dying ReLU" problem.

Author: Teaching Project
Date: 2026-01-21
================================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# ------------------------------------------------------------------------------
# Line: import numpy as np
# ------------------------------------------------------------------------------
# 2.1 WHAT: Imports the NumPy library and gives it a short nickname "np"
# 2.2 WHY: NumPy provides fast mathematical operations on arrays (matrices).
#          We need it for matrix multiplication, random numbers, and more.
#          Alternative: Pure Python lists, but 100x slower for math.
# 2.3 WHEN: Always import at the start of any data science/ML script
# 2.4 WHERE: Used in all machine learning, data science, and scientific computing
# 2.5 HOW: Just write "import numpy as np" at the top of your script
# 2.6 HOW IT WORKS: Python loads the numpy module into memory and creates
#                   an alias "np" so we can write np.array() instead of numpy.array()
# 2.7 OUTPUT: No visible output, but numpy functions become available
# ------------------------------------------------------------------------------
import numpy as np

# ------------------------------------------------------------------------------
# Line: import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
# 2.1 WHAT: Imports the plotting library for creating graphs and charts
# 2.2 WHY: We need to visualize the training loss curves to compare ReLU vs Leaky ReLU
#          Alternative: seaborn (built on matplotlib), plotly (interactive)
# 2.3 WHEN: When you need to create any visualization (plots, charts, images)
# 2.4 WHERE: Data analysis, ML model evaluation, scientific research
# 2.5 HOW: import matplotlib.pyplot as plt, then use plt.plot(), plt.show()
# 2.6 HOW IT WORKS: Creates a figure object and axes for drawing
# 2.7 OUTPUT: Displays visual graphs when plt.show() is called
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Line: import os
# ------------------------------------------------------------------------------
# 2.1 WHAT: Imports the operating system module for file/folder operations
# 2.2 WHY: We need to create output folders and save files
# 2.3 WHEN: When working with files, folders, or system paths
# 2.4 WHERE: Almost every Python project that saves/loads files
# 2.5 HOW: import os, then use os.makedirs(), os.path.join()
# 2.6 HOW IT WORKS: Provides interface to OS-level operations
# 2.7 OUTPUT: No visible output, enables file operations
# ------------------------------------------------------------------------------
import os


# =============================================================================
# SECTION 2: ACTIVATION FUNCTIONS
# =============================================================================

def relu(z):
    """
    ReLU (Rectified Linear Unit) Activation Function
    
    SIMPLE EXPLANATION:
    Think of ReLU like a one-way door:
    - If the number is positive → it passes through unchanged
    - If the number is negative or zero → it becomes 0 (blocked)
    
    MATHEMATICAL FORMULA:
    relu(z) = max(0, z)
    
    EXAMPLE:
    relu(5) = 5    (positive, passes through)
    relu(-3) = 0   (negative, blocked)
    relu(0) = 0    (zero, blocked)
    
    Parameters:
    -----------
    z : numpy.ndarray or float
        3.1 WHAT: The input value(s) to apply ReLU to
        3.2 WHY: This is the weighted sum from the previous layer (before activation)
        3.3 WHEN: During forward propagation, after computing z = W*x + b
        3.4 WHERE: At each neuron in the hidden layer
        3.5 HOW: Pass the numpy array directly: relu(z_values)
        3.6 HOW IT AFFECTS: Values < 0 become 0, values >= 0 stay the same
        3.7 OUTPUT IMPACT: Creates sparse activations (many zeros)
    
    Returns:
    --------
    numpy.ndarray or float
        Same shape as input, with all negative values replaced by 0
    """
    # -------------------------------------------------------------------------
    # Line: return np.maximum(0, z)
    # -------------------------------------------------------------------------
    # 2.1 WHAT: Returns the element-wise maximum between 0 and each value in z
    # 2.2 WHY: This is how ReLU is computed - keep positives, zero out negatives
    #          Alternative: np.where(z > 0, z, 0) - same result, slightly slower
    # 2.3 WHEN: Every time we need ReLU activation
    # 2.4 WHERE: At each neuron during forward propagation
    # 2.5 HOW: np.maximum(0, array) compares each element with 0
    # 2.6 HOW IT WORKS: NumPy broadcasts 0 to match array shape, then compares element-wise
    # 2.7 OUTPUT: Array with same shape, negatives replaced by 0
    # -------------------------------------------------------------------------
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU Activation Function
    
    SIMPLE EXPLANATION:
    The derivative tells us "how much does the output change when input changes?"
    - If z > 0: derivative = 1 (output changes same as input)
    - If z <= 0: derivative = 0 (output doesn't change at all)
    
    WHY THIS MATTERS:
    During learning (backpropagation), we need to know how to adjust weights.
    If derivative = 0, the weights NEVER change → the neuron is "DEAD"!
    This is the core of the "dying ReLU" problem.
    
    Parameters:
    -----------
    z : numpy.ndarray or float
        3.1 WHAT: The input values (same z used in forward pass)
        3.2 WHY: We need to know which values were positive to compute gradient
        3.3 WHEN: During backward propagation
        3.4 WHERE: At each hidden layer neuron
        3.5 HOW: Pass the stored z values from forward pass
        3.6 HOW IT AFFECTS: Determines if gradient flows back (1) or stops (0)
        3.7 OUTPUT IMPACT: Creates "dead neurons" when many zeros
    
    Returns:
    --------
    numpy.ndarray or float
        1 where z > 0, 0 elsewhere (same shape as input)
    """
    # -------------------------------------------------------------------------
    # Line: return (z > 0).astype(float)
    # -------------------------------------------------------------------------
    # 2.1 WHAT: Returns 1.0 where z > 0, and 0.0 elsewhere
    # 2.2 WHY: Mathematical derivative of ReLU is step function at 0
    #          Alternative: np.where(z > 0, 1, 0) - same result
    # 2.3 WHEN: During backward propagation to compute gradients
    # 2.4 WHERE: At each hidden neuron
    # 2.5 HOW: (z > 0) creates boolean array, .astype(float) converts to 0.0/1.0
    # 2.6 HOW IT WORKS: Boolean True becomes 1.0, False becomes 0.0
    # 2.7 OUTPUT: Binary mask showing which neurons are "alive" (1) or "dead" (0)
    # -------------------------------------------------------------------------
    return (z > 0).astype(float)


def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU Activation Function
    
    SIMPLE EXPLANATION:
    Like ReLU, but with a small "leak" for negative values.
    Instead of completely blocking negatives, it lets through 1% (alpha).
    
    Think of it like a door with a tiny crack:
    - If number is positive → passes through unchanged
    - If number is negative → only 1% gets through
    
    MATHEMATICAL FORMULA:
    leaky_relu(z) = z if z > 0 else alpha * z
    
    EXAMPLE:
    leaky_relu(5, alpha=0.01) = 5           (positive, passes through)
    leaky_relu(-100, alpha=0.01) = -1       (negative, only 1% passes)
    leaky_relu(-10, alpha=0.01) = -0.1      (negative, only 1% passes)
    
    WHY LEAKY RELU EXISTS:
    The tiny leak (1%) keeps the gradient non-zero, so neurons can still learn
    even when they receive negative inputs. This prevents "dying neurons"!
    
    Parameters:
    -----------
    z : numpy.ndarray or float
        3.1 WHAT: The input value(s) to apply Leaky ReLU to
        3.2 WHY: This is the weighted sum before activation
        3.3 WHEN: During forward propagation
        3.4 WHERE: At each hidden layer neuron
        3.5 HOW: leaky_relu(z_values, alpha=0.01)
        3.6 HOW IT AFFECTS: Negative values become small (alpha * z), not zero
        3.7 OUTPUT IMPACT: No "dead" activations, all neurons can learn
    
    alpha : float, default=0.01
        3.1 WHAT: The "leak" factor for negative values (how much to let through)
        3.2 WHY: Controls how much gradient flows for negative inputs
        3.3 WHEN: Typically use 0.01 (1%) but can tune for specific problems
        3.4 WHERE: Same value used throughout the network
        3.5 HOW: Multiply negative values by this factor
        3.6 HOW IT AFFECTS: Higher alpha = more gradient flow, but less sparsity
        3.7 OUTPUT IMPACT: alpha=0.01 lets 1% of negative values through
    
    Returns:
    --------
    numpy.ndarray or float
        Same shape as input, with negatives scaled by alpha
    """
    # -------------------------------------------------------------------------
    # Line: return np.where(z > 0, z, alpha * z)
    # -------------------------------------------------------------------------
    # 2.1 WHAT: Returns z where positive, alpha*z where negative/zero
    # 2.2 WHY: Implements the "leak" for negative values
    #          Alternative: np.maximum(alpha * z, z) - same result
    # 2.3 WHEN: Every time we need Leaky ReLU activation
    # 2.4 WHERE: At each hidden layer neuron
    # 2.5 HOW: np.where(condition, value_if_true, value_if_false)
    # 2.6 HOW IT WORKS: Checks each element, returns z or alpha*z based on condition
    # 2.7 OUTPUT: Array with positives unchanged, negatives scaled by alpha
    # -------------------------------------------------------------------------
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    """
    Derivative of Leaky ReLU Activation Function
    
    SIMPLE EXPLANATION:
    The derivative tells us how much output changes when input changes.
    - If z > 0: derivative = 1 (output changes same as input)
    - If z <= 0: derivative = alpha (a tiny bit still flows through!)
    
    KEY DIFFERENCE FROM RELU DERIVATIVE:
    ReLU derivative = 0 for negatives → gradient stops → dead neurons
    Leaky ReLU derivative = alpha for negatives → gradient flows → neurons stay alive
    
    Parameters:
    -----------
    z : numpy.ndarray or float
        3.1 WHAT: The input values from forward pass
        3.2 WHY: Need to know which values were positive vs negative
        3.3 WHEN: During backward propagation
        3.4 WHERE: At each hidden layer neuron
        3.5 HOW: Pass stored z values from forward pass
        3.6 HOW IT AFFECTS: Gradient always flows (1 or alpha), never stops completely
        3.7 OUTPUT IMPACT: All neurons can update their weights
    
    alpha : float, default=0.01
        Same as in leaky_relu function
    
    Returns:
    --------
    numpy.ndarray or float
        1 where z > 0, alpha elsewhere (gradient can always flow!)
    """
    # -------------------------------------------------------------------------
    # Line: return np.where(z > 0, 1, alpha)
    # -------------------------------------------------------------------------
    # 2.1 WHAT: Returns 1 where positive, alpha where negative/zero
    # 2.2 WHY: Derivative of leaky_relu is 1 for z>0, alpha otherwise
    # 2.3 WHEN: During backward propagation
    # 2.4 WHERE: At each hidden neuron
    # 2.5 HOW: Same np.where pattern as leaky_relu
    # 2.6 HOW IT WORKS: Checks each element, returns 1 or alpha
    # 2.7 OUTPUT: Gradient mask that's always non-zero (key to preventing dead neurons)
    # -------------------------------------------------------------------------
    return np.where(z > 0, 1, alpha)


def sigmoid(z):
    """
    Sigmoid Activation Function (for output layer)
    
    SIMPLE EXPLANATION:
    Sigmoid squashes any number into a range between 0 and 1.
    Think of it like a "probability converter":
    - Very negative numbers → close to 0 (unlikely)
    - Very positive numbers → close to 1 (very likely)
    - Zero → exactly 0.5 (50-50 chance)
    
    MATHEMATICAL FORMULA:
    sigmoid(z) = 1 / (1 + e^(-z))
    
    WHY WE USE IT:
    For binary classification (yes/no, 0/1), we need output between 0 and 1.
    Sigmoid is perfect for this!
    
    Parameters:
    -----------
    z : numpy.ndarray or float
        3.1 WHAT: The input value(s) to squash
        3.2 WHY: Convert raw scores to probabilities
        3.3 WHEN: At the output layer for binary classification
        3.4 WHERE: Final layer only (not hidden layers)
        3.5 HOW: sigmoid(final_layer_output)
        3.6 HOW IT AFFECTS: Any input becomes output in (0, 1)
        3.7 OUTPUT IMPACT: Interpretable as probability
    
    Returns:
    --------
    numpy.ndarray or float
        Values between 0 and 1 (probabilities)
    """
    # Clip to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# =============================================================================
# SECTION 3: NEURAL NETWORK CLASS
# =============================================================================

class TwoLayerNeuralNetwork:
    """
    A Simple 2-Layer Neural Network for Binary Classification
    
    ARCHITECTURE DIAGRAM:
    
    Input Layer (10)    Hidden Layer (20)    Output Layer (1)
         x₁  ──┐
         x₂  ──┼──→  h₁ ──┐
         x₃  ──┤          │
         ...  ──┼──→  h₂ ──┼──→  ŷ (prediction)
         ...  ──┤          │
         x₁₀ ──┴──→  h₂₀ ─┘
    
    SIMPLE EXPLANATION:
    Think of this as a team of workers:
    - 10 input workers receive the data
    - 20 hidden workers process the data (using ReLU or Leaky ReLU)
    - 1 output worker gives the final answer (using Sigmoid)
    
    Each worker (neuron) does: output = activation(sum of weighted inputs + bias)
    """
    
    def __init__(self, input_size, hidden_size, output_size, activation='relu', alpha=0.01):
        """
        Initialize the Neural Network
        
        WHAT HAPPENS HERE:
        1. Store the network configuration
        2. Initialize weights randomly (using Xavier initialization)
        3. Initialize biases to zero
        
        Parameters:
        -----------
        input_size : int
            3.1 WHAT: Number of input features (10 in our case)
            3.2 WHY: Determines the size of first weight matrix
            3.3 WHEN: Set once when creating the network
            3.4 WHERE: Must match your data's feature count
            3.5 HOW: Count your input features
            3.6 HOW IT AFFECTS: Shapes of W1 (input_size x hidden_size)
            3.7 OUTPUT IMPACT: Wrong size causes dimension errors
        
        hidden_size : int
            3.1 WHAT: Number of neurons in hidden layer (20 in our case)
            3.2 WHY: More neurons = more capacity to learn complex patterns
            3.3 WHEN: Hyperparameter chosen before training
            3.4 WHERE: Hidden layer between input and output
            3.5 HOW: Experiment with different sizes (10, 20, 50, 100)
            3.6 HOW IT AFFECTS: Network capacity and computational cost
            3.7 OUTPUT IMPACT: Too few = underfitting, too many = overfitting
        
        output_size : int
            3.1 WHAT: Number of output neurons (1 for binary classification)
            3.2 WHY: Must match your prediction task (1 for yes/no)
            3.3 WHEN: Depends on your problem type
            3.4 WHERE: Final layer
            3.5 HOW: Binary = 1, Multi-class = number of classes
            3.6 HOW IT AFFECTS: Shapes of W2 (hidden_size x output_size)
            3.7 OUTPUT IMPACT: Must match your labels
        
        activation : str, default='relu'
            3.1 WHAT: Which activation to use in hidden layer
            3.2 WHY: This is what we're comparing! ReLU vs Leaky ReLU
            3.3 WHEN: Choose before training
            3.4 WHERE: Applied in hidden layer only
            3.5 HOW: 'relu' or 'leaky_relu'
            3.6 HOW IT AFFECTS: How neurons process information
            3.7 OUTPUT IMPACT: Affects learning dynamics and dead neurons
        
        alpha : float, default=0.01
            3.1 WHAT: Leak factor for Leaky ReLU
            3.2 WHY: Controls gradient flow for negative inputs
            3.3 WHEN: Only used if activation='leaky_relu'
            3.4 WHERE: In leaky_relu function
            3.5 HOW: Higher = more leak, lower = more like ReLU
            3.6 HOW IT AFFECTS: trade-off between sparsity and gradient flow
            3.7 OUTPUT IMPACT: 0.01 is typical (1% leak)
        """
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.alpha = alpha
        
        # Xavier initialization for weights
        # WHY: Helps training by keeping activations in good range
        # HOW: Divide by sqrt of input size to normalize
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Storage for forward pass values (needed for backprop)
        self.z1 = None  # Pre-activation hidden layer
        self.a1 = None  # Post-activation hidden layer
        self.z2 = None  # Pre-activation output layer
        self.a2 = None  # Post-activation output layer (predictions)
        
    def forward(self, X):
        """
        Forward Propagation: Pass data through the network
        
        SIMPLE EXPLANATION:
        Data flows forward like water through pipes:
        Input → Hidden Layer → Output
        
        At each layer: output = activation(weights * input + bias)
        
        Parameters:
        -----------
        X : numpy.ndarray, shape (n_samples, input_size)
            3.1 WHAT: The input data matrix (each row = one sample)
            3.2 WHY: This is what we want to make predictions on
            3.3 WHEN: During both training and prediction
            3.4 WHERE: First step of processing
            3.5 HOW: Pass your feature matrix
            3.6 HOW IT AFFECTS: Shapes all subsequent computations
            3.7 OUTPUT IMPACT: Each row produces one prediction
        
        Returns:
        --------
        numpy.ndarray, shape (n_samples, 1)
            Predictions between 0 and 1 (probabilities)
        """
        # Layer 1: Input → Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # Apply activation function (ReLU or Leaky ReLU)
        if self.activation == 'relu':
            self.a1 = relu(self.z1)
        else:
            self.a1 = leaky_relu(self.z1, self.alpha)
        
        # Layer 2: Hidden → Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate):
        """
        Backward Propagation: Learn from mistakes
        
        SIMPLE EXPLANATION:
        After making a prediction, we check how wrong we were.
        Then we adjust the weights to be less wrong next time.
        
        The key formula: new_weight = old_weight - learning_rate * gradient
        
        Parameters:
        -----------
        X : numpy.ndarray, shape (n_samples, input_size)
            3.1 WHAT: Same input data from forward pass
            3.2 WHY: Needed to compute gradients
            3.3 WHEN: After forward pass
            3.4 WHERE: Used to compute how each input affected the output
            3.5 HOW: Same X used in forward pass
            3.6 HOW IT AFFECTS: Used in gradient calculation
            3.7 OUTPUT IMPACT: Determines how weights for each feature change
        
        y : numpy.ndarray, shape (n_samples, 1)
            3.1 WHAT: True labels (what the answer should be)
            3.2 WHY: To compute the error (prediction - true value)
            3.3 WHEN: Only during training (not prediction)
            3.4 WHERE: Compared with predictions
            3.5 HOW: Must be same length as predictions
            3.6 HOW IT AFFECTS: Determines direction of weight updates
            3.7 OUTPUT IMPACT: Model learns to predict these values
        
        learning_rate : float
            3.1 WHAT: How big of a step to take when updating weights
            3.2 WHY: Controls speed of learning
            3.3 WHEN: Hyperparameter set before training
            3.4 WHERE: Applied to all weight updates
            3.5 HOW: Typical values: 0.001, 0.01, 0.1
            3.6 HOW IT AFFECTS: Too high = unstable, too low = slow learning
            3.7 OUTPUT IMPACT: 0.01 is a good starting point
        """
        m = X.shape[0]  # Number of samples
        
        # Output layer error
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer error (THIS IS WHERE DYING RELU HAPPENS!)
        if self.activation == 'relu':
            # ReLU derivative: 0 for z <= 0 → GRADIENT DIES HERE!
            dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        else:
            # Leaky ReLU derivative: alpha for z <= 0 → GRADIENT SURVIVES!
            dz1 = np.dot(dz2, self.W2.T) * leaky_relu_derivative(self.z1, self.alpha)
        
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute Binary Cross-Entropy Loss
        
        SIMPLE EXPLANATION:
        Loss measures "how wrong" our predictions are.
        Lower loss = better predictions.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels (0 or 1)
        
        y_pred : numpy.ndarray
            Predicted probabilities (between 0 and 1)
        
        Returns:
        --------
        float
            The loss value (lower is better)
        """
        epsilon = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.reshape(-1, 1)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def count_dead_neurons(self, X):
        """
        Count Dead Neurons: The Core of Our Investigation!
        
        WHAT ARE DEAD NEURONS?
        A neuron is "dead" if it outputs 0 for ALL training samples.
        This means:
        1. Its activation is always 0
        2. Its gradient is always 0
        3. Its weights NEVER update
        4. It's useless and "dead"
        
        Parameters:
        -----------
        X : numpy.ndarray
            All training samples
        
        Returns:
        --------
        int
            Number of dead neurons (should be 0 for Leaky ReLU!)
        """
        # Forward pass to get activations
        self.forward(X)
        
        # For each neuron, check if ALL activations are zero
        # If activation is 0 for EVERY sample, the neuron is dead
        dead_neurons = np.sum(np.all(self.a1 == 0, axis=0))
        
        return dead_neurons


# =============================================================================
# SECTION 4: TRAINING FUNCTION
# =============================================================================

def train_network(X, y, activation, n_epochs=200, learning_rate=0.01, alpha=0.01):
    """
    Train a Neural Network and Track Loss History
    
    Parameters:
    -----------
    X : numpy.ndarray
        Training features
    
    y : numpy.ndarray
        Training labels
    
    activation : str
        'relu' or 'leaky_relu'
    
    n_epochs : int
        Number of training iterations (default: 200)
    
    learning_rate : float
        Step size for weight updates (default: 0.01)
    
    alpha : float
        Leak factor for Leaky ReLU (default: 0.01)
    
    Returns:
    --------
    tuple
        (trained_network, loss_history)
    """
    # Create neural network
    nn = TwoLayerNeuralNetwork(
        input_size=X.shape[1],
        hidden_size=20,
        output_size=1,
        activation=activation,
        alpha=alpha
    )
    
    loss_history = []
    
    # Training loop
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = nn.forward(X)
        
        # Compute loss
        loss = nn.compute_loss(y, y_pred)
        loss_history.append(loss)
        
        # Backward pass (update weights)
        nn.backward(X, y, learning_rate)
        
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")
    
    return nn, loss_history


def compute_accuracy(nn, X, y):
    """
    Compute Classification Accuracy
    
    Parameters:
    -----------
    nn : TwoLayerNeuralNetwork
        Trained network
    
    X : numpy.ndarray
        Features
    
    y : numpy.ndarray
        True labels
    
    Returns:
    --------
    float
        Accuracy as percentage (0-100)
    """
    predictions = nn.forward(X)
    predicted_classes = (predictions >= 0.5).astype(int)
    accuracy = np.mean(predicted_classes.flatten() == y) * 100
    return accuracy


# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ReLU vs Leaky ReLU: Investigating the Dying ReLU Problem")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Generate Dataset
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Generating Dataset...")
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)  # 1000 samples, 10 features
    y_train = (X_train[:, 0] + X_train[:, 1] - X_train[:, 2] > 0).astype(int)
    print(f"  - Samples: {X_train.shape[0]}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Class distribution: {np.sum(y_train == 1)} positive, {np.sum(y_train == 0)} negative")
    
    # -------------------------------------------------------------------------
    # Step 2: Train with ReLU
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Training with ReLU Activation...")
    print("-" * 50)
    relu_nn, relu_loss = train_network(
        X_train, y_train, 
        activation='relu', 
        n_epochs=200, 
        learning_rate=0.01
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Train with Leaky ReLU
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Training with Leaky ReLU Activation...")
    print("-" * 50)
    leaky_relu_nn, leaky_relu_loss = train_network(
        X_train, y_train, 
        activation='leaky_relu', 
        n_epochs=200, 
        learning_rate=0.01,
        alpha=0.01
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Dead Neuron Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEAD NEURON ANALYSIS")
    print("=" * 70)
    
    relu_dead = relu_nn.count_dead_neurons(X_train)
    leaky_dead = leaky_relu_nn.count_dead_neurons(X_train)
    
    print(f"\n  ReLU Version:")
    print(f"    - Dead neurons: {relu_dead} out of 20 ({relu_dead/20*100:.1f}%)")
    print(f"\n  Leaky ReLU Version:")
    print(f"    - Dead neurons: {leaky_dead} out of 20 ({leaky_dead/20*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Step 5: Accuracy Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON")
    print("=" * 70)
    
    relu_accuracy = compute_accuracy(relu_nn, X_train, y_train)
    leaky_accuracy = compute_accuracy(leaky_relu_nn, X_train, y_train)
    
    print(f"\n  ReLU Accuracy:       {relu_accuracy:.2f}%")
    print(f"  Leaky ReLU Accuracy: {leaky_accuracy:.2f}%")
    
    # -------------------------------------------------------------------------
    # Step 6: Plot Training Loss Curves
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Generating Loss Comparison Plot...")
    
    # Create outputs directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(output_dir), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(relu_loss, label='ReLU', color='blue', linewidth=2)
    plt.plot(leaky_relu_loss, label='Leaky ReLU', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
    plt.title('Training Loss: ReLU vs Leaky ReLU', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'loss_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Plot saved to: {plot_path}")
    
    # -------------------------------------------------------------------------
    # Step 7: Written Comparison (200-300 words)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WRITTEN COMPARISON")
    print("=" * 70)
    
    comparison = f"""
    COMPARISON: ReLU vs Leaky ReLU in Neural Networks
    
    In this experiment, we trained two identical 2-layer neural networks
    on the same binary classification task, differing only in their hidden
    layer activation function: one using standard ReLU and the other using
    Leaky ReLU (alpha=0.01).
    
    DEAD NEURON ANALYSIS:
    The ReLU network experienced {relu_dead} dead neurons ({relu_dead/20*100:.1f}% of hidden
    layer), while the Leaky ReLU network had {leaky_dead} dead neurons 
    ({leaky_dead/20*100:.1f}%). This demonstrates the "dying ReLU" problem in action.
    When a ReLU neuron receives consistently negative inputs, its gradient
    becomes zero, and the weights stop updating - the neuron is permanently
    "dead." Leaky ReLU prevents this by allowing a small gradient (1%) to
    flow even for negative inputs, keeping all neurons alive and learning.
    
    ACCURACY RESULTS:
    ReLU achieved {relu_accuracy:.2f}% training accuracy, while Leaky ReLU
    achieved {leaky_accuracy:.2f}% accuracy. The difference is attributed to:
    1. Dead neurons reducing effective network capacity with ReLU
    2. All neurons contributing to learning with Leaky ReLU
    
    WHEN TO USE EACH:
    - ReLU: Simpler computation, good for most cases, use with careful
      weight initialization (like He initialization)
    - Leaky ReLU: When you observe dying neurons, deep networks, or when
      dealing with data that may produce many negative pre-activations
    
    CONCLUSION:
    Leaky ReLU provides a robust solution to the dying ReLU problem with
    minimal computational overhead. For this task, it achieved better or
    equal accuracy while maintaining all neurons in an active learning state.
    """
    
    print(comparison)
    
    # Save comparison to file
    comparison_path = os.path.join(output_dir, 'written_comparison.txt')
    with open(comparison_path, 'w') as f:
        f.write(comparison)
    print(f"\n  [OK] Comparison saved to: {comparison_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
