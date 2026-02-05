import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os

# Create necessary directories explicitly if they don't exist (safety check)
# 2.1 What the line does
# Creates the output directory if it is missing.
# 2.2 Why it is used
# To prevent 'FileNotFoundError' when saving files later.
# 2.3 When to use it
# whenever your script generates files.
# 2.4 Where to use it
# At the start of the script or before saving.
# 2.5 How to use it
# os.makedirs('path/to/dir', exist_ok=True)
# 2.6 How it works internally
# Checks filesystem; creates directory node if absent.
# 2.7 Output with sample examples
# A new folder 'outputs' appears in the project.
os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------
# 2.1 What the line does
# Sets a fixed seed for random number generation.
# 2.2 Why it is used
# To ensure that every time we run the code, we get the exact same results.
# 2.3 When to use it
# In any ML project involving random initialization or data splitting.
# 2.4 Where to use it
# At the very beginning of the script.
# 2.5 How to use it
# torch.manual_seed(42); np.random.seed(42)
# 2.6 How it works internally
# Initializes the pseudo-random number generator with a specific starting state.
# 2.7 Output with sample examples
# Random numbers generated subsequently will be identical across runs.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------
# 2. Data Generation & Preparation
# ---------------------------------------------------------

def prepare_data():
    """
    Generates and pre-processes regression data.
    
    Returns:
        dataloaders (dict): Loaders for train, val, test.
        input_dim (int): Number of input features.
    """
    # Generate Data
    # 2.1 What the line does
    # Creates a synthetic regression dataset.
    # 2.2 Why it is used
    # To have a controlled dataset for testing algorithms without needing real-world files.
    # 2.3 When to use it
    # For tutorials, benchmarks, or testing logic.
    # 2.4 Where to use it
    # In the data loading phase.
    # 2.5 How to use it
    # X, y = make_regression(n_samples=100, n_features=10)
    # 2.6 How it works internally
    # Generates random feature matrix X and linear target y with noise.
    # 2.7 Output with sample examples
    # X shape: (2000, 40), y shape: (2000,)
    X, y = make_regression(n_samples=2000, n_features=40, noise=15, random_state=SEED)
    
    # Split Data (70% Train, 15% Val, 15% Test)
    # 2.1 What the line does
    # Splits data into training and temporary (val+test) sets.
    # 2.2 Why it is used
    # To separate data for learning vs. evaluating.
    # 2.3 When to use it
    # Always in supervised learning.
    # 2.4 Where to use it
    # After loading data.
    # 2.5 How to use it
    # train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.3)
    # 2.6 How it works internally
    # Randomly shuffles indices and slices arrays.
    # 2.7 Output with sample examples
    # Two sets of arrays.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
    
    # Scaling
    # 2.1 What the line does
    # Standardizes features (mean=0, variance=1).
    # 2.2 Why it is used
    # Optimizers like SGD converge faster if features are on the same scale.
    # 2.3 When to use it
    # Almost always for neural networks.
    # 2.4 Where to use it
    # Before feeding data to the model.
    # 2.5 How to use it
    # scaler.fit_transform(train_data)
    # 2.6 How it works internally
    # Calculates mean and std of training set, applies z-score formula.
    # 2.7 Output with sample examples
    # Transformed arrays with values mostly between -3 and 3.
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_val = scaler_x.transform(X_val)
    X_test = scaler_x.transform(X_test)

    # Scale Targets as well (Crucial for fixed LR)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Convert to PyTorch Tensors
    # 2.1 What the line does
    # Converts numpy arrays to PyTorch FloatTensors.
    # 2.2 Why it is used
    # PyTorch models require Tensor inputs, not numpy arrays.
    # 2.3 When to use it
    # When bridging Scikit-Learn/Numpy and PyTorch.
    # 2.4 Where to use it
    # Before creating DataLoaders.
    # 2.5 How to use it
    # tensor_x = torch.FloatTensor(numpy_x)
    # 2.6 How it works internally
    # Allocates memory for tensors and copies data.
    # 2.7 Output with sample examples
    # Useable tensors for autograd.
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    # DataLoaders
    # 2.1 What the line does
    # Wraps datasets to provide batching and shuffling.
    # 2.2 Why it is used
    # Efficiently feeds data to the model in chunks (batches).
    # 2.3 When to use it
    # In the training loop.
    # 2.4 Where to use it
    # After creating datasets.
    # 2.5 How to use it
    # loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # 2.6 How it works internally
    # Creates an iterator that yields batches of data.
    # 2.7 Output with sample examples
    # An iterator object.
    loaders = {
        'train': DataLoader(train_dataset, batch_size=64, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=64, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=64, shuffle=False)
    }
    
    return loaders, X_train.shape[1], scaler_y

# ---------------------------------------------------------
# 3. Model Definition (Three-Layer MLP)
# ---------------------------------------------------------

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron.
    """
    def __init__(self, input_dim):
        # 3.1 What it does
        # Initializes the parent class (nn.Module).
        # 3.2 Why it is used
        # Required for correct PyTorch module tracking.
        # 3.3 When to use it
        # In __init__ of any custom PyTorch model.
        # 3.4 Where to use it
        # First line of __init__.
        # 3.5 How to use it
        # super(MLP, self).__init__()
        # 3.6 How it affects execution internally
        # Sets up internal registration of parameters.
        # 3.7 Output impact with examples
        # Enabling .parameters() and .to(device) methods.
        super(MLP, self).__init__()
        
        # Layers
        # 2.1 What the line does
        # Defines the sequences of layers: Linear -> ReLU -> Linear -> ReLU -> Linear.
        # 2.2 Why it is used
        # To create a deep neural network capable of learning non-linear patterns.
        # 2.3 When to use it
        # When building the architecture.
        # 2.4 Where to use it
        # Inside __init__.
        # 2.5 How to use it
        # nn.Sequential(nn.Linear(in, out), nn.ReLU())
        # 2.6 How it works internally
        # Stacks layers so output of one becomes input of next.
        # 2.7 Output with sample examples
        # A callable model object.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Regression output (1 neuron)
        )
        
    def forward(self, x):
        """
        Forward pass logic.
        
        Args:
            x (Tensor): Input batch.
            
        Returns:
            Tensor: Prediction.
        """
        # 3.1 What it does
        # Passes input x through the defined layers.
        # 3.2 Why it is used
        # To calculate predictions.
        # 3.3 When to use it
        # Automatically called when you do model(x).
        # 3.4 Where to use it
        # Inside forward method.
        # 3.5 How to use it
        # return self.layers(x)
        # 3.6 How it affects execution internally
        # Computes matrix multiplications and activations.
        # 3.7 Output impact with examples
        # Raw prediction values.
        return self.layers(x)

# ---------------------------------------------------------
# 4. Training Function
# ---------------------------------------------------------

def train_model(model_class, input_dim, optimizer_name, learning_rate, momentum=0.0):
    """
    Trains a fresh model instance with specified optimizer.
    
    Args:
        model_class: Class of the model to instantiate.
        input_dim: Number of input features.
        optimizer_name (str): 'SGD' or 'Adam'.
        learning_rate (float): Step size for optimizer.
        momentum (float): Momentum factor for SGD.
        
    Returns:
        history (dict): Losses and RMSE per epoch.
    """
    # Initialize fresh model
    # 2.1 What the line does
    # Creates a new instance of the MLP.
    # 2.2 Why it is used
    # To Ensure each run starts with fresh weights (though we reset seed ideally or re-init).
    # Since we set seed globally at start, strict comparison might need re-seeding or weight copying.
    # Here rely on fresh instantiation having consistent randomness if called in same order?
    # Actually, to guarantee EXACT same initialization for both runs is tricky unless we save initial state.
    # Instead, we will fix seed right before creation for FAIR comparison.
    torch.manual_seed(SEED) 
    model = model_class(input_dim)
    
    # Loss Function
    # 2.1 What the line does
    # Defines Mean Squared Error loss.
    # 2.2 Why it is used
    # Standard loss for regression.
    # 2.3 When to use it
    # Regression tasks.
    # 2.4 Where to use it
    # Before training loop.
    # 2.5 How to use it
    # criterion = nn.MSELoss()
    # 2.6 How it works internally
    # Computes average squared difference.
    # 2.7 Output with sample examples
    # A single scalar gradient-tracking tensor.
    criterion = nn.MSELoss()
    
    # Optimizer Selection
    if optimizer_name == 'SGD':
        # 3.1 What it does
        # Creates SGD optimizer.
        # 3.2 Why it is used
        # Updates weights based on gradient direction.
        # 3.3 When to use it
        # Baseline optimizer, often generalizes well.
        # 3.4 Where to use it
        # After model creation.
        # 3.5 How to use it
        # optim.SGD(params, lr=lr)
        # 3.6 How it affects execution internally
        # Stores params and update rules.
        # 3.7 Output impact with examples
        # Weights change after step().
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        # 3.1 What it does
        # Creates Adam optimizer.
        # 3.2 Why it is used
        # Adaptive learning rates, usually faster convergence.
        # 3.3 When to use it
        # Default go-to for many tasks.
        # 3.4 Where to use it
        # After model creation.
        # 3.5 How to use it
        # optim.Adam(params, lr=lr)
        # 3.6 How it affects execution internally
        # Tracks momentum and variance of gradients.
        # 3.7 Output impact with examples
        # Weights change dynamically.
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    best_val_rmse = float('inf')
    
    loaders, _, _ = prepare_data() # Re-calling to get loaders is fine, data is consistent via seed
    
    # Training Loop
    EPOCHS = 40
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in loaders['train']:
            optimizer.zero_grad()      # Reset gradients
            outputs = model(X_batch)   # Forward pass
            loss = criterion(outputs, y_batch) # Compute loss
            loss.backward()            # Backward pass
            optimizer.step()           # Update weights
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(loaders['train'])
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in loaders['val']:
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(loaders['val'])
        val_rmse = np.sqrt(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_rmse'].append(val_rmse)
        
        # Save Best Model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), f'outputs/best_model_{optimizer_name.lower()}.pth')
            
    print(f"Finished {optimizer_name}: Best Val RMSE: {best_val_rmse:.4f}")
    return history, best_val_rmse

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    optimizers_to_run = [
        {'name': 'SGD', 'lr': 5e-3, 'momentum': 0.9},
        {'name': 'Adam', 'lr': 1e-3, 'momentum': 0.0}
    ]
    
    results = {}
    results = {}
    loaders, input_dim, scaler_y = prepare_data() # Get input dim and scaler
    
    for opt_config in optimizers_to_run:
        print(f"Training with {opt_config['name']}...")
        hist, best_rmse = train_model(
            MLP, 
            input_dim, 
            opt_config['name'], 
            opt_config['lr'], 
            opt_config['momentum']
        )
        results[opt_config['name']] = {'history': hist, 'best_rmse': best_rmse}

    # ---------------------------------------------------------
    # 6. Visualization & Reporting
    # ---------------------------------------------------------
    
    # Plotting
    # 2.1 What the line does
    # Creates a figure for plotting routines.
    # 2.2 Why it is used
    # To visualize the learning curves.
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(results['SGD']['history']['train_loss'], label='SGD Train Loss')
    plt.plot(results['Adam']['history']['train_loss'], label='Adam Train Loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # RMSE Plot
    plt.subplot(1, 2, 2)
    plt.plot(results['SGD']['history']['val_rmse'], label='SGD Val RMSE')
    plt.plot(results['Adam']['history']['val_rmse'], label='Adam Val RMSE')
    plt.title('Validation RMSE Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/training_curves.png')
    plt.close()
    
    # Summary Table
    # 2.1 What the line does
    # create a summary dict
    summary_data = {
        'Optimizer': ['SGD', 'Adam'],
        'Best Val RMSE': [results['SGD']['best_rmse'], results['Adam']['best_rmse']]
    }
    df = pd.DataFrame(summary_data)
    print("\nFinal Results:")
    print(df)
    
    # Save text summary
    with open('outputs/summary_report.md', 'w') as f:
        f.write("# Optimizer Comparison Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Recommendation\n")
        if results['Adam']['best_rmse'] < results['SGD']['best_rmse']:
            f.write("Adam is recommended. It converged faster and achieved a lower final RMSE.")
        else:
            f.write("SGD is recommended. It achieved better generalization despite potentially slower initial convergence.")
