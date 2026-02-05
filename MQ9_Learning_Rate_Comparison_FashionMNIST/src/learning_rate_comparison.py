import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# ---------------------------------------------------------
# 1. SETUP AND REPRODUCIBILITY
# ---------------------------------------------------------

# ### 🔹 Line Explanation
# #### 2.1 What the line does
# Sets a fixed seed for random number generation.
# #### 2.2 Why it is used
# To ensure that our results are reproducible. Every time we run this code, we get the exact same results.
# #### 2.3 When to use it
# Whenever using random processes (like splitting data or initializing weights) in scientific experiments.
# #### 2.4 Where to use it
# At the very beginning of any Machine Learning script.
# #### 2.5 How to use it
# `set_seed(42)`
# #### 2.6 How it works internally
# It initializes the pseudo-random number generator logic with a specific starting number.
# #### 2.7 Output with sample examples
# No visible output, but subsequent random numbers will be deterministic.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

set_seed(42)

# ---------------------------------------------------------
# 2. CONFIGURATION & DEVICE
# ---------------------------------------------------------

# ### 🔹 Line Explanation
# #### 2.1 What the line does
# Checks if a GPU (Graphics Processing Unit) is available and sets the device accordingly.
# #### 2.2 Why it is used
# GPUs are much faster than CPUs for matrix operations in Deep Learning.
# #### 2.3 When to use it
# Always when writing PyTorch code to ensure it runs on the best available hardware.
# #### 2.4 Where to use it
# Before creating models or moving data tensors.
# #### 2.5 How to use it
# `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
# #### 2.6 How it works internally
# It queries the underlying hardware driver (CUDA for NVIDIA) to see if a compatible device exists.
# #### 2.7 Output with sample examples
# Prints 'Device: cuda' or 'Device: cpu'.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 15

# ---------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------

# ### 🔹 Line Explanation
# #### 2.1 What the line does
# Defines a transformation pipeline to convert images to Tensors and normalize them.
# The original dataset is [0, 255], we want [0, 1]. ToTensor() does this scaling automatically.
# #### 2.2 Why it is used
# Neural networks work best with small, floating-point numbers.
# #### 2.3 When to use it
# When loading image data.
# #### 2.4 Where to use it
# In the `transform` argument of dataset loaders.
# #### 2.5 How to use it
# `transforms.Compose([transforms.ToTensor()])`
# #### 2.6 How it works internally
# It applies each transformation in the list sequentially.
# #### 2.7 Output with sample examples
# Input: Image (0-255). Output: Tensor (0.0-1.0).
transform = transforms.Compose([
    transforms.ToTensor()
])

# ### ⚙️ Function / Method Arguments Explanation
# `torchvision.datasets.FashionMNIST`
# #### 3.1 What it does
# Downloads and loads the Fashion-MNIST dataset.
# #### 3.2 Why it is used
# Standard benchmark dataset for testing simple computer vision models.
# #### 3.3 When to use it
# When you need a standardized dataset of clothing items.
# #### 3.4 Where to use it
# In the data preparation phase.
# #### 3.5 How to use it
# `dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)`
# #### 3.6 How it works internally
# Checks if data exists; if not, downloads zip file, extracts it, and reads binary files.
# #### 3.7 Output impact with examples
# Returns a Dataset object containing 60,000 training images.

print("Loading Fashion-MNIST data...")
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Determine split sizes (we will use test set as validation for simplicity in this project)
# Logic: We strictly follow the user prompt which asks for training and validation loss.
# The user prompted "train split of 60,000 images". We will use the official test set (10k) as validation
# to track performance on unseen data.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ---------------------------------------------------------
# 4. MODEL DEFINITION
# ---------------------------------------------------------

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with 2 hidden layers as requested.
    Input: 28x28 = 784 pixels
    Hidden 1: 256 neurons (common choice)
    Hidden 2: 128 neurons
    Output: 10 classes (Fashion items)
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Flatten input: [Batch, 28, 28] -> [Batch, 784]
        self.flatten = nn.Flatten()
        
        # Layer 1
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        
        # Layer 2
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        
        # Output Layer
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------------------------------------------
# 5. TRAINING UTILITIES
# ---------------------------------------------------------

# ### ⚙️ Function / Method Arguments Explanation
# `train_one_epoch`
# #### 3.1 What it does
# Performs one complete pass (epoch) over the training data to update model weights.
# #### 3.2 Why it is used
# To teach the model by minimizing error.
# #### 3.3 When to use it
# Inside the main training loop, repeated for each epoch.
# #### 3.4 Where to use it
# Called by the `run_experiment` function.
# #### 3.5 How to use it
# `loss, acc = train_one_epoch(model, loader, criterion, optimizer)`
# #### 3.6 How it affects execution internally
# Calculates gradients and steps the optimizer.
# #### 3.7 Output impact with examples
# Returns average loss and accuracy for that epoch.
def train_one_epoch(model, loader, criterion, optimizer):
    model.train() # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = running_loss / total
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc

def validate(model, loader, criterion):
    model.eval() # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # No gradients needed for validation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / total
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc

def run_experiment(learning_rate, experiment_name):
    print(f"\n--- Starting {experiment_name} with LR={learning_rate} ---")
    
    # Re-initialize model to ensure fresh start for each run
    set_seed(42) # Reset seed so initialization is identical for both runs
    model = SimpleMLP().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        # Save best checkpoint
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), f"outputs/{experiment_name}_best.pth")
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%")
        
    print(f"Finished {experiment_name}. Best Val Acc: {best_val_acc:.2f}%")
    return history

# ---------------------------------------------------------
# 6. EXECUTION AND COMPARING
# ---------------------------------------------------------

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Run A: High Learning Rate
history_a = run_experiment(learning_rate=1e-3, experiment_name="Run_A_HighLR")

# Run B: Low Learning Rate
history_b = run_experiment(learning_rate=5e-4, experiment_name="Run_B_LowLR")

# ---------------------------------------------------------
# 7. PLOTTING RESULTS
# ---------------------------------------------------------

# ### 🔹 Line Explanation
# #### 2.1 What the line does
# Creates a figure with 2 subplots (Learning Curves).
# #### 2.2 Why it is used
# To visually compare how the two models learned over time.
# #### 2.3 When to use it
# After collecting training metrics history.
# #### 2.4 Where to use it
# In the analysis / conclusion phase.
# #### 2.5 How to use it
# `plt.subplots(1, 2, figsize=(14, 5))`
# #### 2.6 How it works internally
# Creates a Matplotlib object window with a grid of plots.
# #### 2.7 Output with sample examples
# A blank canvas divided into two sections.

plt.figure(figsize=(14, 6))

# Plot Validation Loss
plt.subplot(1, 2, 1)
plt.plot(history_a['val_loss'], label='Run A (LR=1e-3)', color='red', marker='o')
plt.plot(history_b['val_loss'], label='Run B (LR=5e-4)', color='blue', marker='s')
plt.title('Validation Loss Comparison\n(Lower is Better)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Plot Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(history_a['val_acc'], label='Run A (LR=1e-3)', color='red', marker='o')
plt.plot(history_b['val_acc'], label='Run B (LR=5e-4)', color='blue', marker='s')
plt.title('Validation Accuracy Comparison\n(Higher is Better)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save plot
plt.tight_layout()
plt.savefig('outputs/learning_rate_comparison_plot.png')
print("\ncomparison plot saved to 'outputs/learning_rate_comparison_plot.png'")

# ---------------------------------------------------------
# 8. AUTOMATIC ANALYSIS PARAGRAPH
# ---------------------------------------------------------

print("\n--- CONCLUSION ---")
val_loss_a_final = history_a['val_loss'][-1]
val_loss_b_final = history_b['val_loss'][-1]

print(f"Final Val Loss A (High LR): {val_loss_a_final:.4f}")
print(f"Final Val Loss B (Low LR): {val_loss_b_final:.4f}")

if val_loss_b_final < val_loss_a_final and np.std(history_b['val_loss'][-5:]) < np.std(history_a['val_loss'][-5:]):
    conclusion = """
    I would ship the model from Run B (Learning Rate = 5e-4).
    Run B demonstrates a more stable convergence with smoother loss curves and achieves a comparable or better final validation loss.
    Run A (High LR) shows signs of volatility (bumpy curve), indicating that the optimizer is overshooting the minima slightly.
    The lower learning rate provides a better balance between speed and stability for this architecture.
    """
else:
    conclusion = """
    Based on the results, both learning rates performed similarly, but we look at stability.
    If Run B is smoother, it is generally preferred for production stability.
    """

print(conclusion)
