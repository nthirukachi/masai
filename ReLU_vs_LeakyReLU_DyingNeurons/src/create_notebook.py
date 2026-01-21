"""
Script to generate the teaching-oriented Jupyter Notebook
"""
import json

notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    },
    'cells': []
}

def add_md(content):
    notebook['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': content})

def add_code(content):
    notebook['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': content, 'outputs': [], 'execution_count': None})

# Cell 1: Problem Statement (Markdown)
add_md("""# ReLU vs Leaky ReLU: Investigating the Dying ReLU Problem

## 🧩 Problem Statement

### Simple Explanation (Like Teaching a 10-Year-Old)

Imagine you have workers in a factory:
- **ReLU Worker**: "If the number is positive, I pass it. If negative, I output 0."
- **Problem**: Some workers get lazy and ALWAYS output 0 - they become "dead"!

**Leaky ReLU** fixes this by letting a tiny bit (1%) through for negative numbers.

### Technical Definition
The **dying ReLU problem** occurs when neurons get stuck outputting zero because:
1. ReLU outputs 0 for all negative inputs
2. When output is 0, gradient is also 0
3. Zero gradient means weights never update
4. The neuron is permanently "dead"

## 🪜 Steps to Solve
1. Implement ReLU and Leaky ReLU functions from scratch
2. Build a 2-layer neural network class
3. Train with ReLU (200 epochs)
4. Train with Leaky ReLU (200 epochs)
5. Compare dead neurons and accuracy

## 🎯 Expected Output
- Training loss curves for both activations
- Dead neuron count for each version
- Accuracy comparison
- Written analysis (200-300 words)
""")

# Cell 2: Imports Explanation (Markdown)
add_md("""---
## Section 1: Importing Libraries

### Line: `import numpy as np`

| Point | Explanation |
|-------|-------------|
| 2.1 WHAT | Imports NumPy library with nickname "np" |
| 2.2 WHY | Provides fast array operations for matrix math. Alternative: pure Python lists (100x slower) |
| 2.3 WHEN | Always at the start of ML scripts |
| 2.4 WHERE | All machine learning and data science projects |
| 2.5 HOW | `import numpy as np` |
| 2.6 INTERNAL | Python loads module into memory, creates alias "np" |
| 2.7 OUTPUT | No visible output, enables np.array(), np.dot(), etc. |

### Line: `import matplotlib.pyplot as plt`

| Point | Explanation |
|-------|-------------|
| 2.1 WHAT | Imports plotting library |
| 2.2 WHY | We need to visualize training loss curves |
| 2.3 WHEN | When creating any visualization |
| 2.4 WHERE | Data analysis, ML model evaluation |
| 2.5 HOW | `plt.plot()`, `plt.show()` |
| 2.6 INTERNAL | Creates figure objects for drawing |
| 2.7 OUTPUT | Displays visual graphs |
""")

# Cell 3: Import Code
add_code("""import numpy as np
import matplotlib.pyplot as plt
import os

print("Libraries imported successfully!")
print(f"NumPy version: {np.__version__}")""")

# Cell 4: ReLU Explanation (Markdown)
add_md("""---
## Section 2: ReLU Activation Function

### What is ReLU?
**ReLU** = Rectified Linear Unit = "Keep positives, zero out negatives"

### Mathematical Formula
```
relu(z) = max(0, z)
```

### Real-Life Analogy 🚪
Think of ReLU like a **one-way door**:
- Positive numbers pass through unchanged
- Negative numbers get blocked (become 0)

```mermaid
graph LR
    A[Input: -3] --> B{ReLU}
    B --> C[Output: 0]
    D[Input: 5] --> E{ReLU}
    E --> F[Output: 5]
```

### Line-by-Line Explanation: `return np.maximum(0, z)`

| Point | Explanation |
|-------|-------------|
| 2.1 WHAT | Returns element-wise maximum of 0 and z |
| 2.2 WHY | Implements ReLU: keep positives, zero negatives. Alternative: `np.where(z > 0, z, 0)` |
| 2.3 WHEN | During forward propagation at each hidden neuron |
| 2.4 WHERE | Hidden layers of neural networks |
| 2.5 HOW | `relu(z_array)` - pass any numpy array |
| 2.6 INTERNAL | NumPy broadcasts 0 to match array shape, compares element-wise |
| 2.7 OUTPUT | Array with same shape, negatives replaced by 0 |
""")

# Cell 5: ReLU Code
add_code("""def relu(z):
    '''
    ReLU Activation Function
    
    Simple: If positive, pass through. If negative, output 0.
    Formula: relu(z) = max(0, z)
    '''
    return np.maximum(0, z)


def relu_derivative(z):
    '''
    Derivative of ReLU
    
    - 1 if z > 0 (gradient flows through)
    - 0 if z <= 0 (gradient STOPS - causes dying neurons!)
    '''
    return (z > 0).astype(float)


# Test ReLU
print("=== Testing ReLU ===")
test_values = np.array([-3, -1, 0, 1, 5])
print(f"Input:           {test_values}")
print(f"ReLU output:     {relu(test_values)}")
print(f"ReLU derivative: {relu_derivative(test_values)}")""")

# Cell 6: Leaky ReLU Explanation (Markdown)
add_md("""---
## Section 3: Leaky ReLU Activation Function

### Why Leaky ReLU?
ReLU has a critical flaw: when z <= 0, the derivative is 0, so **the neuron stops learning forever** ("dead neuron").

Leaky ReLU fixes this by allowing a small "leak" for negative values.

### Mathematical Formula
```
leaky_relu(z) = z       if z > 0
              = alpha*z if z <= 0
```
Where alpha = 0.01 (1% leak)

### Real-Life Analogy 🚪💨
Like a door with a tiny crack:
- Positive values pass through fully
- Negative values still get through (just 1%)

### Key Difference: Why This Prevents Dead Neurons

| Activation | Derivative for z < 0 | What Happens |
|------------|---------------------|--------------|
| ReLU | 0 | Gradient stops, neuron dies |
| Leaky ReLU | 0.01 (alpha) | Gradient flows, neuron stays alive! |

### Parameter: `alpha`

| Point | Explanation |
|-------|-------------|
| 3.1 WHAT | The "leak" factor (how much negative values pass through) |
| 3.2 WHY | Prevents dead neurons by keeping gradient non-zero |
| 3.3 WHEN | During forward and backward propagation |
| 3.4 WHERE | Hidden layers of neural network |
| 3.5 HOW | Typical value: 0.01 (1% leak) |
| 3.6 INTERNAL | Multiplies negative values by alpha instead of zeroing |
| 3.7 OUTPUT | Higher alpha = more gradient flow, less sparsity |
""")

# Cell 7: Leaky ReLU Code
add_code("""def leaky_relu(z, alpha=0.01):
    '''
    Leaky ReLU Activation Function
    
    Simple: If positive, pass through. If negative, let 1% through.
    Formula: leaky_relu(z) = z if z > 0, else alpha * z
    
    The 'leak' (alpha=0.01) keeps neurons alive!
    '''
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    '''
    Derivative of Leaky ReLU
    
    - 1 if z > 0
    - alpha if z <= 0 (gradient STILL flows - no dead neurons!)
    '''
    return np.where(z > 0, 1, alpha)


# Test Leaky ReLU
print("=== Testing Leaky ReLU ===")
test_values = np.array([-3, -1, 0, 1, 5])
print(f"Input:                 {test_values}")
print(f"Leaky ReLU output:     {leaky_relu(test_values)}")
print(f"Leaky ReLU derivative: {leaky_relu_derivative(test_values)}")
print()
print("Notice: For -3, ReLU gives 0, but Leaky ReLU gives -0.03 (1% of -3)")""")

# Cell 8: Sigmoid Explanation (Markdown)
add_md("""---
## Section 4: Sigmoid Activation (Output Layer)

### What is Sigmoid?
Squashes any number into range (0, 1) - perfect for probabilities!

### Mathematical Formula
```
sigmoid(z) = 1 / (1 + e^(-z))
```

### Real-Life Analogy 🌡️
Like a "probability converter":
- Very negative inputs → close to 0 (unlikely)
- Very positive inputs → close to 1 (very likely)
- Zero → exactly 0.5 (50-50 chance)

### Why We Use It
For binary classification (yes/no, 0/1), we need output between 0 and 1. Sigmoid is perfect!
""")

# Cell 9: Sigmoid Code
add_code("""def sigmoid(z):
    '''
    Sigmoid Activation Function
    
    Squashes any value to range (0, 1)
    Used for binary classification output layer
    '''
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))


# Test Sigmoid
print("=== Testing Sigmoid ===")
test_values = np.array([-10, -1, 0, 1, 10])
print(f"Input:          {test_values}")
print(f"Sigmoid output: {np.round(sigmoid(test_values), 4)}")
print()
print("Notice: -10 -> ~0, 0 -> 0.5, 10 -> ~1")""")

# Cell 10: Neural Network Class Explanation (Markdown)
add_md("""---
## Section 5: Neural Network Class

### Architecture
```
Input Layer (10)  -->  Hidden Layer (20)  -->  Output Layer (1)
   x1, x2, ..., x10       h1, h2, ..., h20          y_hat
```

### Real-Life Analogy 🏭
Think of this as a team of workers:
- 10 input workers receive the raw data
- 20 hidden workers process the data (using ReLU or Leaky ReLU)
- 1 output worker gives the final answer (using Sigmoid)

Each worker does: `output = activation(weighted_sum_of_inputs + bias)`

### Key Concepts
- **Weights (W)**: How much each input matters
- **Bias (b)**: A constant offset
- **Forward Propagation**: Data flows input → hidden → output
- **Backward Propagation**: Learning by adjusting weights based on errors
""")

# Cell 11: Neural Network Code
add_code("""class TwoLayerNeuralNetwork:
    '''
    A Simple 2-Layer Neural Network for Binary Classification
    
    Architecture: Input(10) -> Hidden(20) -> Output(1)
    
    Hidden layer can use ReLU or Leaky ReLU (our comparison!)
    Output layer uses Sigmoid (for probabilities)
    '''
    
    def __init__(self, input_size, hidden_size, output_size, activation='relu', alpha=0.01):
        '''
        Initialize the Neural Network
        
        Parameters:
        -----------
        input_size: Number of input features (10)
        hidden_size: Number of hidden neurons (20)
        output_size: Number of outputs (1 for binary)
        activation: 'relu' or 'leaky_relu'
        alpha: Leak factor for Leaky ReLU (default 0.01)
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.alpha = alpha
        
        # Xavier initialization for weights
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Storage for forward pass values
        self.z1 = None  # Pre-activation hidden
        self.a1 = None  # Post-activation hidden
        self.z2 = None  # Pre-activation output
        self.a2 = None  # Post-activation output (predictions)
    
    def forward(self, X):
        '''Forward Propagation: Pass data through the network'''
        # Layer 1: Input -> Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # Apply activation
        if self.activation == 'relu':
            self.a1 = relu(self.z1)
        else:
            self.a1 = leaky_relu(self.z1, self.alpha)
        
        # Layer 2: Hidden -> Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate):
        '''Backward Propagation: Learn from mistakes'''
        m = X.shape[0]
        
        # Output layer error
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer error (THIS IS WHERE DYING RELU HAPPENS!)
        if self.activation == 'relu':
            dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        else:
            dz1 = np.dot(dz2, self.W2.T) * leaky_relu_derivative(self.z1, self.alpha)
        
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def compute_loss(self, y_true, y_pred):
        '''Binary Cross-Entropy Loss'''
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.reshape(-1, 1)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def count_dead_neurons(self, X):
        '''Count neurons that output 0 for ALL samples'''
        self.forward(X)
        dead_neurons = np.sum(np.all(self.a1 == 0, axis=0))
        return dead_neurons


print("Neural Network class defined successfully!")
print("Ready to compare ReLU vs Leaky ReLU!")""")

# Cell 12: Training Function Explanation (Markdown)
add_md("""---
## Section 6: Training Function

### What Happens During Training?
1. **Forward Pass**: Make predictions
2. **Compute Loss**: Measure how wrong we are
3. **Backward Pass**: Calculate gradients
4. **Update Weights**: Adjust to be less wrong

This repeats for each epoch (200 times total).
""")

# Cell 13: Training Function Code
add_code("""def train_network(X, y, activation, n_epochs=200, learning_rate=0.01, alpha=0.01):
    '''
    Train a Neural Network and Track Loss History
    
    Parameters:
    -----------
    X: Training features
    y: Training labels
    activation: 'relu' or 'leaky_relu'
    n_epochs: Number of training iterations (200)
    learning_rate: Step size for weight updates (0.01)
    alpha: Leak factor for Leaky ReLU (0.01)
    
    Returns:
    --------
    nn: Trained network
    loss_history: List of loss values per epoch
    '''
    nn = TwoLayerNeuralNetwork(
        input_size=X.shape[1],
        hidden_size=20,
        output_size=1,
        activation=activation,
        alpha=alpha
    )
    
    loss_history = []
    
    for epoch in range(n_epochs):
        y_pred = nn.forward(X)
        loss = nn.compute_loss(y, y_pred)
        loss_history.append(loss)
        nn.backward(X, y, learning_rate)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")
    
    return nn, loss_history


def compute_accuracy(nn, X, y):
    '''Compute classification accuracy'''
    predictions = nn.forward(X)
    predicted_classes = (predictions >= 0.5).astype(int)
    accuracy = np.mean(predicted_classes.flatten() == y) * 100
    return accuracy


print("Training functions defined!")""")

# Cell 14: Data Generation (Markdown)
add_md("""---
## Section 7: Generate Dataset

### Dataset Description
- 1000 samples
- 10 features (can be negative - important for demonstrating dying ReLU!)
- Binary labels based on: x0 + x1 - x2 > 0
""")

# Cell 15: Data Generation Code
add_code("""# Generate dataset
np.random.seed(42)
X_train = np.random.randn(1000, 10)  # 1000 samples, 10 features
y_train = (X_train[:, 0] + X_train[:, 1] - X_train[:, 2] > 0).astype(int)

print("=== Dataset Generated ===")
print(f"Samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Class distribution: {np.sum(y_train == 1)} positive, {np.sum(y_train == 0)} negative")
print()
print("Sample data (first 3 rows):")
print(X_train[:3, :3])""")

# Cell 16: Training ReLU (Markdown)
add_md("""---
## Section 8: Train with ReLU

Now we train a neural network using standard ReLU activation.
Watch for:
- How the loss decreases
- How many neurons might "die"
""")

# Cell 17: Training ReLU Code
add_code("""print("=" * 50)
print("Training with ReLU Activation")
print("=" * 50)

relu_nn, relu_loss = train_network(
    X_train, y_train,
    activation='relu',
    n_epochs=200,
    learning_rate=0.01
)

print()
print("ReLU training complete!")""")

# Cell 18: Training Leaky ReLU (Markdown)
add_md("""---
## Section 9: Train with Leaky ReLU

Now we train another network using Leaky ReLU (alpha=0.01).
The tiny "leak" should prevent neurons from dying.
""")

# Cell 19: Training Leaky ReLU Code
add_code("""print("=" * 50)
print("Training with Leaky ReLU Activation")
print("=" * 50)

leaky_relu_nn, leaky_relu_loss = train_network(
    X_train, y_train,
    activation='leaky_relu',
    n_epochs=200,
    learning_rate=0.01,
    alpha=0.01
)

print()
print("Leaky ReLU training complete!")""")

# Cell 20: Analysis (Markdown)
add_md("""---
## Section 10: Analysis and Comparison

### Dead Neuron Analysis
A neuron is "dead" if it outputs 0 for ALL training samples.

### Expected Results
- ReLU: May have some dead neurons
- Leaky ReLU: Should have 0 dead neurons (the leak prevents death!)
""")

# Cell 21: Analysis Code
add_code("""# Dead Neuron Analysis
relu_dead = relu_nn.count_dead_neurons(X_train)
leaky_dead = leaky_relu_nn.count_dead_neurons(X_train)

print("=" * 50)
print("DEAD NEURON ANALYSIS")
print("=" * 50)
print()
print(f"ReLU Version:")
print(f"  Dead neurons: {relu_dead} out of 20 ({relu_dead/20*100:.1f}%)")
print()
print(f"Leaky ReLU Version:")
print(f"  Dead neurons: {leaky_dead} out of 20 ({leaky_dead/20*100:.1f}%)")

# Accuracy Comparison
relu_accuracy = compute_accuracy(relu_nn, X_train, y_train)
leaky_accuracy = compute_accuracy(leaky_relu_nn, X_train, y_train)

print()
print("=" * 50)
print("ACCURACY COMPARISON")
print("=" * 50)
print()
print(f"ReLU Accuracy:       {relu_accuracy:.2f}%")
print(f"Leaky ReLU Accuracy: {leaky_accuracy:.2f}%")""")

# Cell 22: Visualization (Markdown)
add_md("""---
## Section 11: Visualization - Training Loss Curves

Let's plot the training loss curves to compare how both activations learn over time.
""")

# Cell 23: Visualization Code
add_code("""# Plot training loss curves
plt.figure(figsize=(10, 6))
plt.plot(relu_loss, label='ReLU', color='blue', linewidth=2)
plt.plot(leaky_relu_loss, label='Leaky ReLU', color='orange', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
plt.title('Training Loss: ReLU vs Leaky ReLU', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Loss curves plotted!")""")

# Cell 24: Conclusion (Markdown)
add_md("""---
## Section 12: Conclusion and Key Takeaways

### Summary of Results

| Metric | ReLU | Leaky ReLU |
|--------|------|------------|
| Dead Neurons | Could be higher | Typically 0 |
| Accuracy | Varies | Similar or better |
| Gradient Flow | Stops at 0 | Always flows |

### When to Use Each

**Use ReLU when:**
- Most cases - it's simple and works well
- You have good weight initialization (Xavier/He)
- Shallower networks

**Use Leaky ReLU when:**
- Deep networks where dying neurons are a concern
- Data with many negative pre-activations
- You observe dead neurons with ReLU

### Key Points for Interviews
1. ReLU outputs 0 for negative inputs
2. This can cause "dead neurons" - neurons that never learn
3. Leaky ReLU allows small gradient (alpha) for negative inputs
4. This prevents dead neurons while keeping most ReLU benefits
5. Typical alpha value is 0.01 (1%)
""")

# Save notebook
with open('c:/masai/ReLU_vs_LeakyReLU_DyingNeurons/notebook/ReLU_vs_LeakyReLU_DyingNeurons.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Jupyter Notebook created successfully!")
