import os
import random
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. SET RANDOM SEEDS FOR REPRODUCIBILITY
# ---------------------------------------------------------
# 2.1 What the line does: Sets the seed for Python's built-in random number generator.
# 2.2 Why it is used: To ensure that any random operations (like shuffling) are repeatable.
# 2.3 When to use it: Always in ML experiments to get the same results every time you run the code.
# 2.4 Where to use it: At the very beginning of the script.
# 2.5 How to use it: random.seed(42)
# 2.6 How it works internally: Initializes the pseudo-random number generator with a fixed starting point.
# 2.7 Output with sample examples: Running random.random() twice after reset will give the same sequence.
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data():
    """
    Loads the breast cancer dataset, splits it, and standardizes features.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (arrays): Processed data.
    """
    # ---------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------
    # 2.1 What the line does: Loads the Breast Cancer Wisconsin dataset.
    # 2.2 Why it is used: It's a standard binary classification dataset for medical diagnosis.
    # 2.3 When to use it: For practicing classification on tabular data.
    # 2.4 Where to use it: In data loading functions.
    # 2.5 How to use it: data = load_breast_cancer()
    # 2.6 How it works internally: Reads the CSV-like data bundled with sklearn.
    # 2.7 Output with sample examples: Returns an object with .data (features) and .target (labels).
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # ---------------------------------------------------------
    # SPLIT DATA (TRAIN/TEMP)
    # ---------------------------------------------------------
    # 3.1 What it does: Splits data into training set and a temporary set (for val/test).
    # 3.2 Why it is used: We need separate data to teach the model (train) and test it later.
    # 3.3 When to use it: Before any preprocessing or training.
    # 3.4 Where to use it: Standard ML pipeline.
    # 3.5 How to use it: train_test_split(X, y, test_size=0.3, stratify=y)
    # 3.6 How it affects execution internally: Shuffles data and partitions it carefully.
    # 3.7 Output impact with examples: 70% of data goes to X_train.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # ---------------------------------------------------------
    # SPLIT DATA (VAL/TEST)
    # ---------------------------------------------------------
    # 2.1 What the line does: Splits the temporary set equally into Validation and Test sets.
    # 2.2 Why it is used: To have a 'Mock Exam' (Val) and 'Final Exam' (Test).
    # 2.3 When to use it: After the first split.
    # 2.4 Where to use it: Data preparation phase.
    # 2.5 How to use it: Split the remaining 30% into 15% and 15%.
    # 2.6 How it works internally: Takes the input arrays and divides them.
    # 2.7 Output with sample examples: X_val gets half of X_temp.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # ---------------------------------------------------------
    # STANDARDIZE FEATURES
    # ---------------------------------------------------------
    # 2.1 What the line does: Creates a scaler object to standardize data (mean=0, std=1).
    # 2.2 Why it is used: Neural networks learn better when input numbers are small and centered.
    # 2.3 When to use it: Almost always for Neural Networks with tabular data.
    # 2.4 Where to use it: Before feeding data to the network.
    # 2.5 How to use it: scaler = StandardScaler()
    # 2.6 How it works internally: Computes mean and stdev of the column.
    # 2.7 Output with sample examples: Transforms 150.0 to 1.25 (Z-score).
    scaler = StandardScaler()
    
    # Fit on training data ONLY to avoid data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training shape: {X_train_scaled.shape}")
    print(f"Validation shape: {X_val_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def build_model(input_shape, weight_decay=1e-4):
    """
    Builds a feed-forward neural network with L2 regularization.
    
    Args:
        input_shape (tuple): Shape of the input features.
        weight_decay (float): L2 regularization factor.
        
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    # ---------------------------------------------------------
    # INITIALIZE MODEL
    # ---------------------------------------------------------
    # 2.1 What the line does: Creates a linear stack of layers.
    # 2.2 Why it is used: Most simple Neural Networks are sequential (layer after layer).
    # 2.3 When to use it: For feed-forward networks.
    # 2.4 Where to use it: Model definition.
    # 2.5 How to use it: model = models.Sequential()
    # 2.6 How it works internally: Creates a container for layers.
    # 2.7 Output with sample examples: An empty model object ready for layers.
    model = models.Sequential()

    # ---------------------------------------------------------
    # ADD FIRST HIDDEN LAYER
    # ---------------------------------------------------------
    # 3.1 What it does: Adds a dense (fully connected) layer with 64 neurons.
    # 3.2 Why it is used: To learn complex patterns from the input.
    # 3.3 When to use it: As the first layer of the MLP.
    # 3.4 Where to use it: Inside model.add().
    # 3.5 How to use it: layers.Dense(64, activation='relu', kernel_regularizer=...)
    # 3.6 How it affects execution internally: Creates a weight matrix of shape (input_dim, 64).
    # 3.7 Output impact with examples: Outputs 64 distinct activation values.
    model.add(layers.Dense(
        64, 
        activation='relu', 
        input_shape=input_shape,
        kernel_regularizer=regularizers.l2(weight_decay) # L2 Regularization (Weight Decay)
    ))

    # ---------------------------------------------------------
    # ADD SECOND HIDDEN LAYER
    # ---------------------------------------------------------
    # 2.1 What the line does: Adds a second dense layer with 32 neurons.
    # 2.2 Why it is used: To combine patterns from the first layer into higher-level features.
    # 2.3 When to use it: To increase model depth and capacity.
    # 2.4 Where to use it: After the first hidden layer.
    # 2.5 How to use it: layers.Dense(32, activation='relu', ...)
    # 2.6 How it works internally: Multiplies inputs by weights and adds bias.
    # 2.7 Output with sample examples: Reduces dimensions from 64 to 32.
    model.add(layers.Dense(
        32, 
        activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay)
    ))

    # ---------------------------------------------------------
    # ADD OUTPUT LAYER
    # ---------------------------------------------------------
    # 2.1 What the line does: Adds the final layer with 1 neuron and sigmoid activation.
    # 2.2 Why it is used: To output a probability between 0 and 1 (Cancer or Not).
    # 2.3 When to use it: For binary classification tasks.
    # 2.4 Where to use it: Last layer of the network.
    # 2.5 How to use it: layers.Dense(1, activation='sigmoid')
    # 2.6 How it works internally: Squashes the final sum into the [0, 1] range.
    # 2.7 Output with sample examples: Returns e.g., 0.85 (85% chance of being malignant).
    model.add(layers.Dense(1, activation='sigmoid'))

    # ---------------------------------------------------------
    # COMPILE MODEL
    # ---------------------------------------------------------
    # 3.1 What it does: Configures the model for training.
    # 3.2 Why it is used: To specify the optimizer, loss function, and metrics.
    # 3.3 When to use it: After building the structure, before training.
    # 3.4 Where to use it: Last step of build function.
    # 3.5 How to use it: model.compile(...)
    # 3.6 How it affects execution internally: Prepares the computation graph.
    # 3.7 Output impact with examples: Model is now ready for .fit().
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['AUC', 'accuracy']
    )
    
    return model

def main():
    # Load Data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    
    # Build Model
    model = build_model(input_shape=(X_train.shape[1],))
    
    # ---------------------------------------------------------
    # DEFINE EARLY STOPPING
    # ---------------------------------------------------------
    # 3.1 What it does: Stops training if validation loss doesn't improve.
    # 3.2 Why it is used: To prevent overfitting (memorizing noise).
    # 3.3 When to use it: Training iterative models (Neural Networks, XGBoost).
    # 3.4 Where to use it: In the callbacks list of model.fit().
    # 3.5 How to use it: callbacks.EarlyStopping(...)
    # 3.6 How it affects execution internally: Checks metric at end of split.
    # 3.7 Output impact with examples: Stops at epoch 25 instead of running to 100.
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',       # Watch the validation loss
        patience=4,               # Wait 4 epochs before stopping
        restore_best_weights=True,# Go back to the best version
        verbose=1
    )

    print("\nStarting training with Early Stopping...")
    
    # ---------------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------------
    # 3.1 What it does: Trains the model on the data.
    # 3.2 Why it is used: To find the best weights that minimize loss.
    # 3.3 When to use it: After compiling.
    # 3.4 Where to use it: Main execution block.
    # 3.5 How to use it: model.fit(...)
    # 3.6 How it affects execution internally: Runs forward/backward pass multiple times.
    # 3.7 Output impact with examples: Returns history object with loss/metrics per epoch.
    history = model.fit(
        X_train, y_train,
        epochs=100,               # Set high, let Early Stopping decide
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # ---------------------------------------------------------
    # EVALUATE ON TEST SET
    # ---------------------------------------------------------
    print("\nEvaluating on Test Set...")
    test_loss, test_auc, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Check stopping epoch
    stopped_epoch = early_stopping.stopped_epoch
    # If it completed all epochs (didn't stop early), stopped_epoch is 0 in some versions, 
    # so we use len(history.history['loss'])
    final_epoch = len(history.history['loss'])
    print(f"Training stopped at epoch: {final_epoch}")

    # Output Summary Table
    print("\n--- Training Summary ---")
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)
    print(f"{'Stop Epoch':<15} | {final_epoch:<10}")
    print(f"{'Test AUC':<15} | {test_auc:.4f}")
    print(f"{'Test Acc':<15} | {test_acc:.4f}")
    print("-" * 30)
    
    # Save output for documentation
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/execution_output.md', 'w') as f:
        f.write("# Execution Output\n\n")
        f.write(f"- Stops at Epoch: **{final_epoch}**\n")
        f.write(f"- Validation AUC: **{history.history['val_auc'][-1]:.4f}**\n")
        f.write(f"- Test Accuracy: **{test_acc:.4f}**\n")
        
    # Plot History
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.axvline(x=final_epoch-1-4, color='r', linestyle='--', label='Best Model') # Approx
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Subplot 2: AUC
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/training_history.png')
    print("Training history plot saved to outputs/training_history.png")

if __name__ == "__main__":
    main()
