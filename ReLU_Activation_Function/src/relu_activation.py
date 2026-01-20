# ================================================
# RELU ACTIVATION FUNCTION - FROM SCRATCH
# ================================================
# Question 14: Compare Activation Functions Mathematically and Visually
# 
# This file implements the ReLU (Rectified Linear Unit) activation function
# and its derivative without using any built-in activation functions.
# ================================================

import numpy as np  # 2.1 What: Imports NumPy library for numerical operations
                    # 2.2 Why: NumPy provides efficient array operations and mathematical functions
                    # 2.3 When: Always imported when working with numerical data in Python
                    # 2.4 Where: Used in ML, data science, scientific computing
                    # 2.5 How: import numpy as np (alias 'np' is standard convention)
                    # 2.6 Internal: Loads compiled C code for fast vectorized operations

import matplotlib.pyplot as plt  # 2.1 What: Imports Matplotlib's pyplot module
                                  # 2.2 Why: For creating visualizations (plots, graphs)
                                  # 2.3 When: Whenever you need to visualize data
                                  # 2.4 Where: Data analysis, ML model evaluation, research papers
                                  # 2.5 How: import matplotlib.pyplot as plt
                                  # 2.6 Internal: Creates figure and axes objects for rendering

# ================================================
# RELU FUNCTION IMPLEMENTATION
# ================================================

def relu(z):
    """
    Calculates the ReLU (Rectified Linear Unit) activation function.
    
    Formula: f(z) = max(0, z)
    
    Arguments (Rule 3.1-3.7):
    ----------------------------
    - z:
        3.1 What: Input value(s) - can be a single number or NumPy array.
        3.2 Why: The weighted sum from previous layer that needs transformation.
             ReLU is the simplest non-linear activation; alternatives include
             LeakyReLU, ELU, but ReLU is most popular for its simplicity.
        3.3 When: During forward propagation in neural networks.
        3.4 Where: Hidden layers in CNNs, deep networks, modern architectures.
        3.5 How: relu(-2) returns 0, relu(3) returns 3.
        3.6 Internal: Uses np.maximum which compares element-wise with 0.
        3.7 Output Impact: Returns 0 for negatives, input value for positives.
    
    Returns:
    --------
    float or np.ndarray: ReLU of input, always >= 0.
    
    Example:
    --------
    >>> relu(-5)
    0
    >>> relu(3)
    3
    """
    # 2.1 What: Calculate ReLU using the max(0, z) formula.
    # 2.2 Why: Simplest non-linear activation with NO vanishing gradient for positive inputs.
    #      Alternative: LeakyReLU allows small negatives, but ReLU is simpler.
    # 2.3 When: Default choice for hidden layers in modern deep networks.
    # 2.4 Where: CNNs, ResNets, Transformers, most modern architectures.
    # 2.5 How: result = relu(weighted_sum)
    # 2.6 Internal: np.maximum compares z with 0 element-wise, returns larger.
    # 2.7 Output: For z=-5, returns 0. For z=5, returns 5.
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Calculates the derivative of the ReLU function.
    
    Formula: f'(z) = 1 if z > 0, else 0
    
    Note: Derivative is technically undefined at z=0, but conventionally set to 0.
    
    Arguments (Rule 3.1-3.7):
    ----------------------------
    - z:
        3.1 What: Input value(s) at which to compute the derivative.
        3.2 Why: Needed for backpropagation to compute gradients.
             The binary nature (0 or 1) makes gradient computation very fast.
        3.3 When: During backward pass (training) in neural networks.
        3.4 Where: Used in gradient descent optimization.
        3.5 How: relu_derivative(2) returns 1, relu_derivative(-3) returns 0.
        3.6 Internal: Uses comparison operation, returns 1 or 0.
        3.7 Output Impact: Returns 1 for positive (gradient flows), 0 for negative (blocked).
    
    Returns:
    --------
    float or np.ndarray: Derivative of ReLU at input z (either 0 or 1).
    
    Example:
    --------
    >>> relu_derivative(2)
    1
    >>> relu_derivative(-3)
    0
    """
    # 2.1 What: Return 1 if z > 0, else 0.
    # 2.2 Why: This is the mathematically derived gradient (step function).
    #      Advantage: Gradient is 1 for positive inputs = NO vanishing gradient!
    # 2.6 Internal: Comparison returns boolean, astype converts to 0/1.
    # 2.7 Output: Either 0 (gradient blocked) or 1 (gradient flows).
    return np.where(z > 0, 1, 0).astype(float)


# ================================================
# VISUALIZATION FUNCTIONS
# ================================================

def plot_relu_function(z_range, output_dir):
    """
    Plots the ReLU function.
    
    Arguments:
    - z_range: NumPy array of input values for x-axis.
    - output_dir: Directory path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # 2.1 What: Compute ReLU values for all inputs.
    y = relu(z_range)
    
    # 2.1 What: Plot the ReLU curve.
    plt.plot(z_range, y, 'r-', linewidth=2, label='ReLU(z) = max(0, z)')
    
    # 2.1 What: Add reference lines.
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2.1 What: Add labels and title.
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel('Output ReLU(z)', fontsize=12)
    plt.title('ReLU Activation Function: f(z) = max(0, z)', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2.1 What: Add annotations.
    plt.annotate('Dead Region\n(gradient = 0)', xy=(-4, 0.5), fontsize=9, color='red')
    plt.annotate('Linear Region\n(gradient = 1)', xy=(2, 4), fontsize=9, color='green')
    plt.annotate('Kink at z=0', xy=(0.2, 0.5), fontsize=9, color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.ylim(-1, 7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/relu_function.png', dpi=150)
    plt.close()
    print(f"[OK] ReLU function plot saved to {output_dir}/relu_function.png")


def plot_relu_derivative(z_range, output_dir):
    """
    Plots the ReLU derivative.
    
    Arguments:
    - z_range: NumPy array of input values for x-axis.
    - output_dir: Directory path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # 2.1 What: Compute derivative values.
    y = relu_derivative(z_range)
    
    # 2.1 What: Plot the derivative (step function).
    plt.plot(z_range, y, 'b-', linewidth=2, label="ReLU Derivative")
    
    # 2.1 What: Add reference lines.
    plt.axhline(y=1, color='green', linestyle=':', alpha=0.7, label='Gradient = 1')
    plt.axhline(y=0, color='red', linestyle=':', alpha=0.7, label='Gradient = 0 (dead)')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel("Derivative ReLU'(z)", fontsize=12)
    plt.title("ReLU Derivative: f'(z) = 1 if z > 0, else 0", fontsize=14)
    plt.legend(loc='right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2.1 What: Annotate key insight.
    plt.annotate('No vanishing gradient\nfor positive inputs!', xy=(3, 1), xytext=(4, 0.7),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    plt.annotate('Dead neurons\n(gradient = 0)', xy=(-3, 0), xytext=(-4.5, 0.3),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    
    plt.ylim(-0.2, 1.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/relu_derivative.png', dpi=150)
    plt.close()
    print(f"[OK] ReLU derivative plot saved to {output_dir}/relu_derivative.png")


def plot_combined(z_range, output_dir):
    """
    Plots both ReLU function and its derivative on the same figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: ReLU function
    y1 = relu(z_range)
    ax1.plot(z_range, y1, 'r-', linewidth=2, label='ReLU(z)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Input (z)', fontsize=11)
    ax1.set_ylabel('Output ReLU(z)', fontsize=11)
    ax1.set_title('ReLU Function', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 7)
    
    # Right plot: Derivative
    y2 = relu_derivative(z_range)
    ax2.plot(z_range, y2, 'b-', linewidth=2, label="ReLU Derivative")
    ax2.axhline(y=1, color='green', linestyle=':', alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle=':', alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Input (z)', fontsize=11)
    ax2.set_ylabel("Derivative", fontsize=11)
    ax2.set_title('ReLU Derivative (Step Function)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 1.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/relu_combined.png', dpi=150)
    plt.close()
    print(f"[OK] Combined plot saved to {output_dir}/relu_combined.png")


# ================================================
# NUMERICAL ANALYSIS
# ================================================

def numerical_analysis(output_dir):
    """
    Creates numerical analysis table and gradient analysis.
    """
    # 2.1 What: Define test inputs as specified in the problem.
    test_inputs = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])
    
    # 2.1 What: Compute ReLU and derivative for each input.
    relu_values = relu(test_inputs)
    derivative_values = relu_derivative(test_inputs)
    
    print("\n" + "=" * 60)
    print("NUMERICAL ANALYSIS TABLE")
    print("=" * 60)
    print(f"{'Input (z)':<12} {'ReLU(z)':<15} {'Derivative':<15}")
    print("-" * 42)
    
    for z, r, d in zip(test_inputs, relu_values, derivative_values):
        print(f"{z:<12.1f} {r:<15.1f} {d:<15.1f}")
    
    # 2.1 What: Gradient analysis at specific points.
    print("\n" + "=" * 60)
    print("GRADIENT ANALYSIS AT x = -2, 0, 2")
    print("=" * 60)
    
    gradient_points = [-2, 0, 2]
    for x in gradient_points:
        grad = relu_derivative(x)
        if grad == 1:
            strength = "PERFECT (= 1.0, no vanishing!)"
        else:
            strength = "DEAD (= 0, gradient blocked!)"
        print(f"At x = {x:>2}: Gradient = {grad:.1f} -> {strength}")
    
    # 2.1 What: Identify strongest gradient region.
    print("\n" + "=" * 60)
    print("STRONGEST GRADIENT REGION")
    print("=" * 60)
    print("Gradients are strongest (= 1.0) for ALL z > 0")
    print("NO vanishing gradient problem for positive inputs!")
    print("Gradient is constant 1 (unlike sigmoid/tanh which decay)")
    
    # 2.1 What: Dead neuron analysis.
    print("\n" + "=" * 60)
    print("DEAD NEURON PROBLEM")
    print("=" * 60)
    print("* For z <= 0: Gradient = 0 (completely blocked)")
    print("* If a neuron always receives negative input, it 'dies'")
    print("* Dead neurons never learn - a key limitation of ReLU")
    print("* Solution: Use LeakyReLU or careful weight initialization")
    
    # 2.1 What: Save analysis to file.
    with open(f'{output_dir}/numerical_analysis.md', 'w') as f:
        f.write("# ReLU Numerical Analysis\n\n")
        f.write("## Output Table\n\n")
        f.write("| Input (z) | ReLU(z) | Derivative |\n")
        f.write("|-----------|---------|------------|\n")
        for z, r, d in zip(test_inputs, relu_values, derivative_values):
            f.write(f"| {z:.1f} | {r:.1f} | {d:.1f} |\n")
        f.write("\n## Gradient Analysis\n\n")
        f.write("| Point | Gradient | Status |\n")
        f.write("|-------|----------|--------|\n")
        for x in gradient_points:
            grad = relu_derivative(x)
            status = "ACTIVE" if grad == 1 else "DEAD"
            f.write(f"| x = {x} | {grad:.1f} | {status} |\n")
    
    print(f"\n[OK] Numerical analysis saved to {output_dir}/numerical_analysis.md")


# ================================================
# WRITTEN ANALYSIS
# ================================================

def written_analysis():
    """
    Prints written analysis about ReLU activation.
    """
    print("\n" + "=" * 60)
    print("WRITTEN ANALYSIS: RELU ACTIVATION FUNCTION")
    print("=" * 60)
    
    analysis = """
NO VANISHING GRADIENT (FOR POSITIVE INPUTS):
---------------------------------------------
ReLU's biggest advantage is that the gradient is EXACTLY 1 for all 
positive inputs. This means:
- No gradient decay during backpropagation
- Deep networks can train effectively  
- Learning is fast and stable

Compare with sigmoid (max 0.25) and tanh (max 1.0 but decays quickly).
ReLU maintains gradient = 1 for ALL positive inputs, not just at one point.

DEAD NEURON PROBLEM:
--------------------
ReLU's major limitation is the "dying ReLU" problem:
- For z <= 0, gradient = 0 (completely blocked)
- If weights push a neuron to always output 0, it never recovers
- The neuron becomes "dead" and stops learning

Solutions:
+ LeakyReLU: f(z) = max(0.01*z, z) - small gradient for negatives
+ ELU: Smooth curve for negatives
+ Careful initialization (He initialization)
+ Lower learning rates

WHEN TO USE RELU:
-----------------
+ Hidden layers in deep networks (CNNs, etc.)
+ When training speed is important
+ When vanishing gradient is a concern
+ Modern architectures (ResNets, Transformers)

WHEN NOT TO USE RELU:
---------------------
- Output layer (use sigmoid/softmax instead)
- When bounded output is needed
- RNNs (tanh often preferred for stability)
- When dead neurons are a critical concern

COMPARISON TABLE:
-----------------
| Activation | Max Gradient | Vanishing? | Dead Neurons? |
|------------|--------------|------------|---------------|
| Sigmoid    | 0.25         | YES        | No            |
| Tanh       | 1.0          | YES        | No            |
| ReLU       | 1.0 always   | NO         | YES           |
"""
    print(analysis)


# ================================================
# MAIN EXECUTION
# ================================================

def main():
    """
    Main function to run all ReLU activation analyses.
    """
    print("=" * 60)
    print("RELU ACTIVATION FUNCTION - FROM SCRATCH")
    print("=" * 60)
    
    # 2.1 What: Define output directory.
    output_dir = 'c:/masai/ReLU_Activation_Function/outputs'
    
    # 2.1 What: Create input range for plotting.
    z_range = np.linspace(-6, 6, 200)  # 200 points from -6 to 6
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_relu_function(z_range, output_dir)
    plot_relu_derivative(z_range, output_dir)
    plot_combined(z_range, output_dir)
    
    # Numerical analysis
    numerical_analysis(output_dir)
    
    # Written analysis
    written_analysis()
    
    print("\n" + "=" * 60)
    print("RELU ACTIVATION ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
