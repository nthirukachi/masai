import numpy as np  # 2.1 What: Import NumPy library. 2.2 Why: For efficient array operations (vectorization).
import matplotlib.pyplot as plt  # 2.1 What: Import Matplotlib.pyplot. 2.2 Why: To visualize data and decision boundaries.

# 2.1 What: Set random seed. 2.2 Why: Ensures reproducibility of random numbers (same results every run).
np.random.seed(42)

def generate_data(n_samples=100):
    """
    Generates synthetic dataset for student pass/fail classification.
    
    ⚙️ Arguments (Rule 3.1-3.7):
    - n_samples:
        3.1 What: Number of data points to generate.
        3.2 Why: Controls dataset size.
        3.3 When: Generating synthetic data.
        3.5 How: generate_data(100)
        3.7 Output Impact: Returns n_samples rows.
    """
    # 2.1 What: Generate random integer study hours (0-100).
    # 2.2 Why: Simulates different student efforts.
    # 2.6 Internal: Creates array of size n_samples.
    study_hours = np.random.randint(0, 100, n_samples)
    
    # 2.1 What: Generate random integer attendance (40-100).
    # 2.2 Why: Simulates class participation.
    attendance = np.random.randint(40, 100, n_samples)
    
    # 2.1 What: Create Ground Truth labels.
    # 2.2 Why: We need known answers to train the model.
    # 2.6 Internal: Checks condition, returns boolean, converts to int (0/1).
    labels = ((study_hours + 0.5 * attendance) > 75).astype(int)
    
    # 2.1 What: Stack features into a single matrix X.
    # 2.2 Why: ML models expect a 2D feature matrix (samples x features).
    X = np.column_stack((study_hours, attendance))
    
    # 2.1 What: Assign labels to y.
    y = labels
    
    return X, y

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Initializes the Perceptron.
        
        ⚙️ Arguments:
        - learning_rate:
            3.1 What: Step size for weight updates.
            3.2 Why: Controls stability vs speed.
        - epochs:
            3.1 What: Number of passes through dataset.
            3.2 Why: Repeated exposure is needed to learn.
        """
        # 2.1 What: Store learning rate.
        self.learning_rate = learning_rate
        # 2.1 What: Store epochs.
        self.epochs = epochs
        # 2.1 What: Initialize weights placeholder.
        self.weights = None
        # 2.1 What: Initialize bias placeholder.
        self.bias = None
        # 2.1 What: List to track errors per epoch.
        self.errors_ = []

    def step_function(self, z):
        """
        Step activation function.
        
        ⚙️ Arguments:
        - z:
            3.1 What: The linear weighted sum.
            3.6 Internal: Input float or array.
        """
        # 2.1 What: Return 1 if positive, else 0.
        # 2.2 Why: Converts continuous score to binary decision.
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        """
        Trains the model.
        
        ⚙️ Arguments:
        - X: Feature matrix.
        - y: Target labels.
        """
        # 2.1 What: Get shape of data.
        n_samples, n_features = X.shape
        
        # 2.1 What: Initialize weights to zeros.
        # 2.6 Behavior: Starting from neutral.
        self.weights = np.zeros(n_features)
        # 2.1 What: Initialize bias to zero.
        self.bias = 0
        # 2.1 What: Reset error history.
        self.errors_ = []
        
        # 2.1 What: Loop over epochs.
        for epoch in range(self.epochs):
            errors_in_epoch = 0  # 2.1 What: Counter for mistakes this round.
            
            # 2.1 What: Loop through every student data point.
            for fn_idx, x_i in enumerate(X):
                # 2.1 What: Calculate Linear Sum (Dot Product).
                # 2.6 Internal: w1*x1 + w2*x2 + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # 2.1 What: Apply Step Function.
                # 2.2 Why: To get prediction (0 or 1).
                y_predicted = self.step_function(linear_output)
                
                # 2.1 What: Calculate Update term.
                # 2.2 Why: Learning Rule (Lr * Error).
                update = self.learning_rate * (y[fn_idx] - y_predicted)
                
                # 2.1 What: Update Weights.
                # 2.6 Internal: Move weight vector towards/away from input.
                self.weights += update * x_i
                
                # 2.1 What: Update Bias.
                self.bias += update
                
                # 2.1 What: Track if we made a mistake.
                if update != 0:
                    errors_in_epoch += 1
            
            # 2.1 What: Save total errors for this epoch.
            self.errors_.append(errors_in_epoch)
            
        return self

    def predict(self, X):
        """
        Predicts labels for new data.
        
        ⚙️ Arguments:
        - X: New data matrix.
        """
        # 2.1 What: Calculate linear score for all inputs.
        linear_output = np.dot(X, self.weights) + self.bias
        # 2.1 What: Apply step function and return.
        return self.step_function(linear_output)

def plot_decision_boundary(X, y, classifier, title="Perceptron Decision Boundary"):
    """
    Plots decision boundary.
    """
    # 2.1 What: Create a figure.
    plt.figure(figsize=(10, 6))
    
    # 2.1 What: Scatter plot for Fail (0) - Red.
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', marker='o', label='Fail (0)')
    # 2.1 What: Scatter plot for Pass (1) - Blue.
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='x', label='Pass (1)')
    
    # 2.1 What: Define x-axis range for the line.
    x1_min, x1_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    x1_values = np.linspace(x1_min, x1_max, 100)
    
    # 2.1 What: Calculate corresponding y-axis values (x2) using Line Equation.
    # 2.6 Internal: x2 = -(w1*x1 + b) / w2
    if classifier.weights[1] != 0:
        x2_values = -(classifier.weights[0] * x1_values + classifier.bias) / classifier.weights[1]
        plt.plot(x1_values, x2_values, 'k--', label='Decision Boundary')
    
    # 2.1 What: Add labels and legend.
    plt.xlabel('Study Hours')
    plt.ylabel('Attendance %')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 2.1 What: Save figure.
    plt.savefig('c:/masai/Perceptron_From_Scratch/outputs/decision_boundary.png')
    # plt.show() # Commented for auto-execution

def plot_convergence(errors):
    """
    Plots error convergence.
    """
    plt.figure(figsize=(8, 5))
    # 2.1 What: Plot errors vs epochs.
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Updates (Errors)')
    plt.title('Convergence Analysis')
    plt.grid(True)
    plt.savefig('c:/masai/Perceptron_From_Scratch/outputs/convergence_plot.png')
    # plt.show()

def main():
    print("Generating data...") # 2.1 What: User feedback.
    X, y = generate_data(n_samples=100)
    
    print("\nTraining Perceptron...")
    model = Perceptron(learning_rate=0.01, epochs=100)
    model.fit(X, y) # 2.1 What: Train the model.
    print(f"Learned Weights: {model.weights}")
    
    print("\nEvaluating...")
    predictions = model.predict(X)
    # 2.1 What: Calculate accuracy.
    accuracy = np.mean(predictions == y)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    
    print("\nPlotting...")
    plot_decision_boundary(X, y, model)
    plot_convergence(model.errors_)
    print("Plots saved.")

if __name__ == "__main__":
    main()

