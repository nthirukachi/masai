import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import time
import os

# Set random seeds for reproducibility
# 2.1 What: Sets the seed for random number generation in PyTorch, NumPy, and Python's random module.
# 2.2 Why: To ensure that the results are reproducible. If we run the code multiple times, we get the same results.
# 2.3 When: Always when doing experiments in ML.
# 2.4 Where: At the beginning of the script.
# 2.5 How: torch.manual_seed(42)
# 2.6 Internally: Initializes the pseudo-random number generator with a specific start point.
# 2.7 Output: None visible, but subsequent random numbers are fixed.
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Data Processing
# ==========================================

def load_and_preprocess_data():
    """
    Generates synthetic fraud data, handles imbalance, and prepares tensors.

    Returns:
        dataloaders (dict): PyTorch DataLoaders for train, val, test.
        input_dim (int): Number of input features.
    
    ### ⚙️ Function Arguments Explanation
    This function takes no arguments but performs a complex pipeline internally.
    """
    
    print("\n[Step 1] Generating and Preprocessing Data...")

    # Generate synthetic dataset
    # 2.1 What: Creates a synthetic dataset simulating credit card fraud.
    # 2.2 Why: Real fraud data is sensitive and hard to get. Synthetic data allows us to control difficulty.
    # 2.3 When: For practice, teaching, or initial algorithm testing.
    # 2.4 Where: Start of the pipeline.
    # 2.5 How: make_classification(n_samples=...)
    # 2.6 Internally: Generates clusters of points from gaussian distributions.
    # 2.7 Output: X (features), y (labels).
    X, y = make_classification(
        n_samples=50000,        # 3.1 Total rows. 3.2 To have enough data.
        n_features=30,          # 3.1 Columns/Attributes. 3.2 Mimics real PCA data.
        n_informative=20,       # 3.1 Useful features. 3.2 Signal vs Noise ratio.
        n_redundant=10,         # 3.1 Useless/Correlated features. 3.2 Adds difficulty.
        n_classes=2,            # 3.1 Fraud vs Normal.
        weights=[0.98, 0.02],   # 3.1 Imbalance ratio (98% normal, 2% fraud).
        flip_y=0.01,            # 3.1 Noise (flipping labels).
        random_state=42
    )

    # Split Data: Train (60%), Validation (20%), Test (20%)
    # 2.1 What: Splits data into training and temporary test sets.
    # 2.2 Why: We need separate sets to Train, Tune (Val), and Evaluate (Test).
    # 2.3 When: Standard ML workflow.
    # 2.4 Where: After data loading.
    # 2.5 How: train_test_split(X, y, test_size=0.4)
    # 2.6 Internally: Shuffles and partitions arrays.
    # 2.7 Output: Arrays split by ratio.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Handle Imbalance using SMOTE
    # 2.1 What: Synthetic Minority Over-sampling Technique (SMOTE).
    # 2.2 Why: Our dataset is 98% normal. The model will be biased. SMOTE fixes this.
    # 2.3 When: When classes are highly imbalanced.
    # 2.4 Where: ONLY applied to TRAINING data. Never Test/Val.
    # 2.5 How: SMOTE().fit_resample(X, y)
    # 2.6 Internally: Finds neighbors of minority class points and creates line segments between them to make new points.
    # 2.7 Output: Balanced X_train_resampled, y_train_resampled.
    print(f"  Before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE:  {np.bincount(y_train_res)}")

    # Scaling
    # 2.1 What: Standardizes features (Mean=0, Std=1).
    # 2.2 Why: Neural networks learn faster and better when input numbers are small and similar in scale.
    # 2.3 When: Always for Neural Networks.
    # 2.4 Where: Fit on Train, Apply to Val/Test.
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch Tensors
    # 2.1 What: Converts numpy arrays to PyTorch tensors.
    # 2.2 Why: PyTorch models need Tensors, not numpy arrays, to compute gradients.
    train_dataset = FraudDataset(X_train_res, y_train_res)
    val_dataset = FraudDataset(X_val, y_val)
    test_dataset = FraudDataset(X_test, y_test)

    # DataLoaders
    # 2.1 What: Wraps datasets to provide batching and shuffling.
    # 2.2 Why: We can't feed 50,000 rows at once. specific batch size is friendlier to RAM/GPU.
    batch_size = 64
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    return dataloaders, 30 # 30 features

class FraudDataset(Dataset):
    """
    Custom PyTorch Dataset for Fraud Data.
    """
    def __init__(self, features, labels):
        # 3.1 What: Constructor. 3.2 Stores data.
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # Make (N, 1)

    def __len__(self):
        # 3.1 What: Returns size. 3.2 Used by DataLoader to know when epoch ends.
        return len(self.features)

    def __getitem__(self, idx):
        # 3.1 What: Returns one sample. 3.2 Used by DataLoader to construct batches.
        return self.features[idx], self.labels[idx]

# ==========================================
# 2. Neural Network Architectures
# ==========================================

class ShallowWideNet(nn.Module):
    """
    Model 1: Shallow but Wide.
    Structure: Input -> 64 -> 32 -> 1
    """
    def __init__(self, input_dim):
        super(ShallowWideNet, self).__init__()
        # 2.1 What: Define layers.
        # 2.2 Why: We need to specify the transformation steps.
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 2.1 What: Forward pass logic. 2.2 Defines how data flows.
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

class DeepNarrowNet(nn.Module):
    """
    Model 2: Deep and Narrow.
    Structure: Input -> 32 -> 32 -> 32 -> 32 -> 1
    """
    def __init__(self, input_dim):
        super(DeepNarrowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32), # Extra depth
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class HybridNet(nn.Module):
    """
    Model 3: Hybrid Activation.
    Structure: Input -> 64(ReLU) -> 32(ReLU) -> 16(Tanh) -> 1
    """
    def __init__(self, input_dim):
        super(HybridNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.act1 = nn.ReLU()
        
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        
        self.fc3 = nn.Linear(32, 16)
        self.act3 = nn.Tanh() # Hybrid part
        
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.sigmoid(self.output(x))
        return x

class CustomNet(nn.Module):
    """
    Model 4: Custom Design with Dropout for regularization.
    """
    def __init__(self, input_dim):
        super(CustomNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), # Batch Norm for stability
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),     # Dropout to prevent overfitting
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Training & Evaluation Engine
# ==========================================

def train_model(model, dataloaders, device, name="Model"):
    """
    Trains a model with Early Stopping.
    """
    print(f"\nTraining {name}...")
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    history = {'train_loss': [], 'val_loss': []}

    model.to(device)

    for epoch in range(50): # Max 50 epochs
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()       # Clear old gradients
            outputs = model(inputs)     # Forward pass
            loss = criterion(outputs, labels) # Calc loss
            loss.backward()             # Backprop
            optimizer.step()            # Update weights
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        history['train_loss'].append(epoch_loss)

        # Validation Phase
        model.eval() # Stop dropout/batchnorm updates
        val_loss = 0.0
        with torch.no_grad(): # Save memory, no gradients needed
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_epoch_loss = val_loss / len(dataloaders['val'].dataset)
        history['val_loss'].append(val_epoch_loss)

        print(f"  Epoch {epoch+1}/50 - Train Loss: {epoch_loss:.4f} - Val Loss: {val_epoch_loss:.4f}")

        # Early Stopping Check
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            counter = 0
            torch.save(model.state_dict(), f"outputs/{name}_best.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(f"outputs/{name}_best.pth"))
    return history

def evaluate_model(model, dataloaders, device, name="Model"):
    """
    Evaluates model on Test set and returns metrics.
    """
    model.eval()
    y_true = []
    y_scores = []
    
    print(f"Evaluating {name}...")
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int) # Threshold 0.5

    # Calculate Metrics
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    
    # Store results for plotting
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    results = {
        'name': name,
        'cm': cm,
        'fpr': fpr, 'tpr': tpr, 'auc': roc_auc,
        'precision': precision, 'recall': recall
    }
    return results

def plot_results(all_results):
    """
    Plots ROC, PR Curves, and Confusion Matrices.
    """
    os.makedirs("outputs", exist_ok=True)
    
    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    for res in all_results:
        plt.plot(res['fpr'], res['tpr'], label=f"{res['name']} (AUC = {res['auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig("outputs/roc_curves.png")
    plt.close()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    for res in all_results:
        plt.plot(res['recall'], res['precision'], label=f"{res['name']}")
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig("outputs/pr_curves.png")
    plt.close()

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloaders, input_dim = load_and_preprocess_data()
    
    models = [
        (ShallowWideNet(input_dim), "ShallowWide"),
        (DeepNarrowNet(input_dim), "DeepNarrow"),
        (HybridNet(input_dim), "HybridActivation"),
        (CustomNet(input_dim), "CustomDesign")
    ]
    
    all_results = []
    
    for model, name in models:
        _ = train_model(model, dataloaders, device, name)
        results = evaluate_model(model, dataloaders, device, name)
        all_results.append(results)
        
    plot_results(all_results)
    print("\nAll done! Check 'outputs/' folder for results.")

if __name__ == "__main__":
    main()
