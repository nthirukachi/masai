import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. DATA GENERATION (Provided by User)
# ---------------------------------------------------------

# ### 🔹 Line Explanation
# #### 2.1 What the line does
# Defines a function to generate a stream of data batches with concept drift (shifting clusters).
# #### 2.2 Why it is used
# To simulate a non-stationary environment where the relationship between inputs (X) and output (y) changes over time, forcing the model to adapt.
# #### 2.3 When to use it
# Use this when you need synthetic data to test algorithms designed for online learning or concept drift adaptation.
# #### 2.4 Where to use it
# Used in research (testing drift detection), financial modeling (market regime changes), or IoT sensor data simulation.
# #### 2.5 How to use it
# batches = drifting_stream(seed=42)
# #### 2.6 How it works internally
# It loops through a list of 'shifts'. For each shift, it generates a classification dataset and adds the shift values to the features X[:,0] and X[:,1].
# #### 2.7 Output with sample examples
# Returns a list of tuples: [(X1, y1), (X2, y2), ...]. Each X is (500, 2).
def drifting_stream(seed=99):
    # ### 🔹 Line Explanation
    # #### 2.1 What the line does
    # Initializes a random number generator with a fixed seed.
    # #### 2.2 Why it is used
    # To ensure reproducibility. Real-world simulation requires control; we want the same "random" numbers every time we run the code.
    # #### 2.3 When to use it
    # Always in experiments, testing, and teaching to guarantee consistent results.
    # #### 2.4 Where to use it
    # At the start of any stochastic (random) process.
    # #### 2.5 How to use it
    # rng = np.random.default_rng(123)
    # #### 2.6 How it works internally
    # Creates a Generator instance with the PCG64 algorithm state initialized by the seed.
    # #### 2.7 Output with sample examples
    # A Generator object.
    rng = np.random.default_rng(seed)

    batches = []
    
    # ### 🔹 Line Explanation
    # #### 2.1 What the line does
    # Defines the sequence of coordinate shifts to apply to the data features.
    # #### 2.2 Why it is used
    # To artificially induce "drift". By moving the data points, we change their location relative to the decision boundary, simulating a changing world.
    # #### 2.3 When to use it
    # When simulating concept drift or covariate shift.
    # #### 2.4 Where to use it
    # In synthetic data generation for stress-testing models.
    # #### 2.5 How to use it
    # shifts = [(0,0), (1,1), (-1,-1)]
    # #### 2.6 How it works internally
    # A list of tuples. (0.0, 0.0) means no shift. (0.8, -0.6) means add 0.8 to x1 and subtract 0.6 from x2.
    # #### 2.7 Output with sample examples
    # A list of tuples: [(0.0, 0.0), (0.8, -0.6), (1.2, 0.9)]
    shifts = [(0.0, 0.0), (0.8, -0.6), (1.2, 0.9)]
    
    for drift_x, drift_y in shifts:
        # ### 🔹 Line Explanation
        # #### 2.1 What the line does
        # Generates a random classification dataset with 500 samples and 2 features.
        # #### 2.2 Why it is used
        # To create the base data structure (clouds of points) that we will then shift.
        # #### 2.3 When to use it
        # When you need a quick, labeled dataset for binary classification testing.
        # #### 2.4 Where to use it
        # Scikit-learn tutorials, unit tests, and rapid prototyping.
        # #### 2.5 How to use it
        # X, y = make_classification(n_samples=100)
        # #### 2.6 How it works internally
        # Creates clusters of points based on a hypercube vertices algorithm.
        # #### 2.7 Output with sample examples
        # X is (500, 2) array, y is (500,) array of 0s and 1s.
        X, y = make_classification(
            n_samples=500,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            class_sep=1.2,
            random_state=rng.integers(1000),
        )
        
        # ### 🔹 Line Explanation
        # #### 2.1 What the line does
        # Adds the drift value `drift_x` to the first feature (column 0) of all samples.
        # #### 2.2 Why it is used
        # To strictly implement the concept drift. It pushes the data cloud along the X-axis.
        # #### 2.3 When to use it
        # When modifying data distributions manually.
        # #### 2.4 Where to use it
        # Data augmentation or simulation.
        # #### 2.5 How to use it
        # X[:, 0] += 5.0
        # #### 2.6 How it works internally
        # Numpy vectorized addition. Adds scalar `drift_x` to every element in the 0th column of X in-place.
        # #### 2.7 Output with sample examples
        # If mean was 0, new mean is `drift_x`.
        X[:, 0] += drift_x

        # ### 🔹 Line Explanation
        # #### 2.1 What the line does
        # Adds the drift value `drift_y` to the second feature (column 1).
        # #### 2.2 Why it is used
        # To push the data cloud along the Y-axis.
        # #### 2.3 When to use it
        # Same as above.
        # #### 2.4 Where to use it
        # Same as above.
        # #### 2.5 How to use it
        # X[:, 1] += 2.0
        # #### 2.6 How it works internally
        # Vectorized addition in-place.
        # #### 2.7 Output with sample examples
        # Returns modified X.
        X[:, 1] += drift_y

        batches.append((X, y))
        
    return batches

# ---------------------------------------------------------
# 2. ADAPTIVE PERCEPTRON CLASS
# ---------------------------------------------------------

class AdaptivePerceptron:
    # ### ⚙️ Function / Method Arguments Explanation
    # #### 3.1 What it does
    # Initializes the AdaptivePerceptron model with hyperparameters.
    # #### 3.2 Why it is used
    # To set up the initial state of the object before any training happens.
    # #### 3.3 When to use it
    # When creating a new instance of the class: `model = AdaptivePerceptron()`.
    # #### 3.4 Where to use it
    # Standard Python class definition.
    # #### 3.5 How to use it
    # model = AdaptivePerceptron(learning_rate=0.1, decay_rate=0.9, decay_steps=5)
    # #### 3.6 How it affects execution internally
    # Sets `self.learning_rate`, `self.decay_rate`, etc., which control the training loop behavior.
    # #### 3.7 Output impact with examples
    # Returns a new object instance.
    def __init__(self, learning_rate=0.1, decay_rate=0.9, decay_steps=5):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = None
        self.bias = 0
        self.reset_count = 0 
        self.learning_rates_log = [] # To track LR over time for deliverables

    # ### ⚙️ Function / Method Arguments Explanation
    # #### 3.1 What it does
    # The sigmoid activation function.
    # #### 3.2 Why it is used
    # Maps any real number to a value between 0 and 1. Useful for probability estimation.
    # NOTE: Standard Perceptron uses Step function. We use Sigmoid here for smoother gradients or just probability output.
    # However, strict Perceptron uses Step. Let's use Step for prediction but Sigmoid concept is often taught.
    # Let's stick to standard Perceptron (Step function implementation) for "Perceptron" project.
    # #### 3.3 When to use it
    # In Logistic Regression or Neural Networks.
    # #### 3.4 Where to use it
    # Output layer for binary classification.
    # #### 3.5 How to use it
    # p = self.activation(z)
    # #### 3.6 How it affects execution internally
    # Computes 1 / (1 + exp(-z)).
    # #### 3.7 Output impact with examples
    # Input 0 -> 0.5. Input large pos -> ~1. Input large neg -> ~0.
    def activation(self, z):
        # Using a simple Heaviside Step Function for standard Perceptron
        return 1 if z >= 0 else 0

    # ### ⚙️ Function / Method Arguments Explanation
    # #### 3.1 What it does
    # Predicts class labels for samples in X.
    # #### 3.2 Why it is used
    # To use the trained model on new data.
    # #### 3.3 When to use it
    # During evaluation (validation/testing).
    # #### 3.4 Where to use it
    # `y_pred = model.predict(X_val)`
    # #### 3.5 How to use it
    # y_pred = model.predict(X)
    # #### 3.6 How it affects execution internally
    # Computes dot product w*x + b and applies activation.
    # #### 3.7 Output impact with examples
    # Returns array of 0s and 1s.
    def predict(self, X):
        if self.weights is None:
            # If not trained, return zeros or random
            return np.zeros(X.shape[0])
            
        linear_output = np.dot(X, self.weights) + self.bias
        # Vectorized application of step function
        y_predicted = np.array([self.activation(z) for z in linear_output])
        return y_predicted

    # ### ⚙️ Function / Method Arguments Explanation
    # #### 3.1 What it does
    # Updates weights based on a single sample (SGD).
    # #### 3.2 Why it is used
    # The core learning mechanism of Perceptron.
    # #### 3.3 When to use it
    # Inside the training loop.
    # #### 3.4 Where to use it
    # Called by `train_epoch`.
    # #### 3.5 How to use it
    # self.update_weights(x_i, y_i)
    # #### 3.6 How it affects execution internally
    # Calculates error, updates self.weights and self.bias.
    # #### 3.7 Output impact with examples
    # Weights change slightly in direction of the error.
    def update_weights(self, x_i, y_true):
        # Calculate prediction for this single sample
        linear_output = np.dot(x_i, self.weights) + self.bias
        y_pred = self.activation(linear_output)
        
        # Perceptron Update Rule: w = w + lr * (y_true - y_pred) * x
        error = y_true - y_pred
        
        # Only update if there is an error
        if error != 0:
            update = self.learning_rate * error
            self.weights += update * x_i
            self.bias += update

    # ### ⚙️ Function / Method Arguments Explanation
    # #### 3.1 What it does
    # Resets the model's weights and bias to initial random state.
    # #### 3.2 Why it is used
    # To "forget" the old concept when accuracy drops significantly (Drift detected).
    # #### 3.3 When to use it
    # When validation accuracy < 70%.
    # #### 3.4 Where to use it
    # Logic in the main stream loop.
    # #### 3.5 How to use it
    # model.reset_model(n_features)
    # #### 3.6 How it affects execution internally
    # Re-initializes self.weights with random small numbers. Increments reset_count.
    # #### 3.7 Output impact with examples
    # Model becomes "dumb" again (random) and must relearn.
    def reset_model(self, n_features):
        rng = np.random.default_rng(42)
        # Initialize small random weights
        self.weights = rng.random(n_features) * 0.01
        self.bias = 0
        self.learning_rate = self.initial_learning_rate # Optional: Reset LR too? Usually yes if restarting.
        self.reset_count += 1
        print("  [RESET] TRIGGERED: Weights re-initialized!")

# ---------------------------------------------------------
# 3. EXPERIMENT EXECUTION
# ---------------------------------------------------------

def run_experiment():
    # 1. Get Data Stream
    batches = drifting_stream(seed=99)
    print(f"Generated {len(batches)} batches of data.\n")
    
    # 2. Initialize Model
    # Decay LR by 10% (x 0.9) every 5 epochs
    model = AdaptivePerceptron(learning_rate=0.1, decay_rate=0.9, decay_steps=5)
    
    # 3. Storage for Metrics
    global_accuracies = []
    reset_points = [] # Store batch indices where reset happened
    
    # We will simulate "epochs" per batch. 
    # The prompt says "decays... every five epochs".
    # Since we have few batches, we assume multiple passes (epochs) per batch.
    EPOCHS_PER_BATCH = 15
    
    overall_epoch_counter = 0

    for batch_idx, (X, y) in enumerate(batches):
        print(f"=== BATCH {batch_idx + 1} ===")
        
        # A. Split into Train(300) and Validation(200) as per requirements
        # "evaluate accuracy on a 200 sample validation buffer"
        # Since stream is sequential, we can just split the batch.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=200, shuffle=False # Keep temporal order if inherent, though make_classification is i.i.d per batch
        )
        # Note: shuffle=False is good for streams, or shuffle=True since make_classifiction provides i.i.d samples within call.
        # Let's use shuffle=True to ensure train/val are representative of this specific batch distribution.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=200, shuffle=True, random_state=42)
        
        # Initialize weights if first batch
        if model.weights is None:
            model.reset_model(n_features=X.shape[1])
            model.reset_count = 0 # Don't count initialization as a reset
        
        # B. Train for N epochs on this batch
        for epoch in range(EPOCHS_PER_BATCH):
            overall_epoch_counter += 1
            
            # Decay Schedule Check
            if overall_epoch_counter % model.decay_steps == 0:
                # ### 🔹 Line Explanation
                # #### 2.1 What the line does
                # Multiplies current learning rate by decay rate (e.g. 0.9).
                # #### 2.2 Why it is used
                # To reduce the step size as training progresses. Small steps help converge to a precise minimum.
                # #### 2.3 When to use it
                # When using gradient descent based methods.
                # #### 2.4 Where to use it
                # Inside the epoch loop.
                # #### 2.5 How to use it
                # lr = lr * 0.9
                # #### 2.6 How it works internally
                # Floating point multiplication.
                # #### 2.7 Output with sample examples
                # 0.1 -> 0.09 -> 0.081
                model.learning_rate *= model.decay_rate
                # print(f"    [Epoch {overall_epoch_counter}] LR Decayed to {model.learning_rate:.6f}")
            
            model.learning_rates_log.append(model.learning_rate)

            # Iterate over samples (SGD)
            for i in range(len(X_train)):
                model.update_weights(X_train[i], y_train[i])
        
        # C. Evaluate on Validation Buffer
        # ### 🔹 Line Explanation
        # #### 2.1 What the line does
        # Uses the model to predict labels for the validation set.
        # #### 2.2 Why it is used
        # To measure how well the model generalizes to unseen data from the same distribution (the same batch).
        # #### 2.3 When to use it
        # After training on the current batch is done.
        # #### 2.4 Where to use it
        # Validation step.
        # #### 2.5 How to use it
        # preds = model.predict(X_val)
        # #### 2.6 How it works internally
        # Matrix multiplication followed by step function.
        # #### 2.7 Output with sample examples
        # Array of predictions.
        val_predictions = model.predict(X_val)
        
        # Calculate Accuracy
        val_acc = accuracy_score(y_val, val_predictions)
        global_accuracies.append(val_acc)
        print(f"  Batch {batch_idx+1} Validation Accuracy: {val_acc:.4f}")
        
        # D. Check for Concept Drift / Reset Condition
        # "resets weights only if accuracy drops below 70 percent"
        if val_acc < 0.70:
            print(f"  [DRIFT] Accuracy {val_acc:.2f} < 0.70. Drift likely detected.")
            model.reset_model(n_features=X.shape[1])
            reset_points.append(batch_idx + 1)
            # IMPORTANT: After reset, do we retrain on this batch? 
            # The prompt doesn't specify. Usually, in online learning, you might retrain or just move on.
            # To "save" the performance, let's retrain on the CURRENT batch with new weights.
            # Otherwise next batch starts with random weights trained on nothing.
            print("  [RETRAIN] Retraining on current batch after reset...")
            model.learning_rate = model.initial_learning_rate # Reset LR as well? Yes.
            
            # Simple retraining (1 pass or N epochs? Let's do N epochs for fairness)
            for epoch in range(EPOCHS_PER_BATCH):
                 for i in range(len(X_train)):
                    model.update_weights(X_train[i], y_train[i])
            
            # Re-evaluate logic? Optional. Let's stick to the prompt's flow "Stream batches sequentially".
            # We record the accuracy *before* the fix effectively showing the drop.
    
    # ---------------------------------------------------------
    # 4. OUTPUT GENERATION
    # ---------------------------------------------------------
    print("\n=== FINAL ANALYSIS ===")
    print(f"Total Resets: {model.reset_count}")
    print(f"Final Batch Accuracy: {global_accuracies[-1]:.4f}")
    
    # Generate Output 1: Accuracy Timeline
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(batches) + 1), global_accuracies, marker='o', linestyle='-', label='Validation Accuracy')
    plt.axhline(y=0.70, color='r', linestyle='--', label='Reset Threshold (0.70)')
    plt.axhline(y=0.80, color='g', linestyle='--', label='Success Criteria (0.80)')
    
    # Mark reset points
    for rp in reset_points:
        plt.axvline(x=rp, color='orange', linestyle=':', label='Weight Reset' if rp == reset_points[0] else "")
        
    plt.title('Adaptive Perceptron Accuracy over Drifting Batches')
    plt.xlabel('Batch Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/accuracy_timeline.png')
    print("Saved accuracy_timeline.png to outputs/")
    
    # Generate Output 2: Learning Rate Table
    # We will save first 50 LR values or sampled values
    lr_df = pd.DataFrame({'Epoch': range(1, len(model.learning_rates_log) + 1), 'Learning_Rate': model.learning_rates_log})
    lr_df.to_csv('outputs/learning_rate_log.csv', index=False)
    print("Saved learning_rate_log.csv to outputs/")

if __name__ == "__main__":
    run_experiment()
