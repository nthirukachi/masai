# ================================================
# TANH ACTIVATION FUNCTION - FROM SCRATCH
# ================================================
# Question 14: Compare Activation Functions Mathematically and Visually
# 
# This file implements the Tanh (Hyperbolic Tangent) activation function
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
# TANH FUNCTION IMPLEMENTATION
# ================================================

def tanh(z):
    """
    Calculates the Tanh (Hyperbolic Tangent) activation function from scratch.
    
    Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    
    Alternative: tanh(z) = 2 * sigmoid(2z) - 1
    
    Arguments (Rule 3.1-3.7):
    ----------------------------
    - z:
        3.1 What: Input value(s) - can be a single number or NumPy array.
        3.2 Why: The weighted sum from previous layer that needs to be transformed.
             We use the exponential formula for direct implementation from scratch.
             Alternative: Could use np.tanh() but that's built-in.
        3.3 When: During forward propagation in neural networks.
        3.4 Where: Used in hidden layers, RNNs, LSTMs, and when zero-centered output is needed.
        3.5 How: tanh(0) returns 0, tanh(np.array([-1, 0, 1])) returns [-0.76, 0, 0.76]
        3.6 Internal: Computes e^z and e^(-z), then calculates the ratio.
        3.7 Output Impact: Returns values between -1 and 1.
    
    Returns:
    --------
    float or np.ndarray: Tanh of input, always in range (-1, 1).
    
    Example:
    --------
    >>> tanh(0)
    0.0
    >>> tanh(1)
    0.7615941559557649
    """
    # 2.1 What: Calculate tanh using the mathematical formula.
    # 2.2 Why: Squashes any real number to range (-1, 1) - zero-centered unlike sigmoid.
    #      This is the direct implementation; alternative is np.tanh() but we avoid built-ins.
    # 2.3 When: When we need zero-centered outputs (common in hidden layers).
    # 2.4 Where: RNNs, LSTMs, older neural networks, image processing.
    # 2.5 How: result = tanh(weighted_sum)
    # 2.6 Internal: np.exp(z) computes e^z, np.exp(-z) computes e^(-z).
    # 2.7 Output: For z=0, returns 0. For z=2, returns ~0.964.
    
    exp_z = np.exp(z)      # 2.1 What: Compute e^z
    exp_neg_z = np.exp(-z) # 2.1 What: Compute e^(-z)
    
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)


def tanh_derivative(z):
    """
    Calculates the derivative of the Tanh function.
    
    Formula: tanh'(z) = 1 - tanh^2(z)
    
    Arguments (Rule 3.1-3.7):
    ----------------------------
    - z:
        3.1 What: Input value(s) at which to compute the derivative.
        3.2 Why: Needed for backpropagation to compute gradients.
             This formula is mathematically derived; no simpler alternative.
        3.3 When: During backward pass (training) in neural networks.
        3.4 Where: Used in gradient descent optimization.
        3.5 How: tanh_derivative(0) returns 1.0 (maximum gradient).
        3.6 Internal: First computes tanh(z), then computes 1 - tanh^2(z).
        3.7 Output Impact: Returns values between 0 and 1 (maximum at z=0).
    
    Returns:
    --------
    float or np.ndarray: Derivative of tanh at input z.
    
    Example:
    --------
    >>> tanh_derivative(0)
    1.0
    >>> tanh_derivative(2)
    0.07065082485316443
    """
    # 2.1 What: First compute the tanh value.
    # 2.2 Why: The derivative formula requires the tanh output.
    # 2.6 Internal: Stores tanh(z) to avoid computing it twice.
    t = tanh(z)
    
    # 2.1 What: Apply the derivative formula 1 - tanh^2(z).
    # 2.2 Why: This is the mathematically derived gradient formula.
    # 2.6 Internal: Element-wise operations for arrays.
    # 2.7 Output: Maximum value 1.0 occurs at z=0; decreases towards 0 at extremes.
    return 1 - t ** 2


# ================================================
# VISUALIZATION FUNCTIONS
# ================================================

def plot_tanh_function(z_range, output_dir):
    """
    Plots the Tanh function.
    
    Arguments:
    - z_range: NumPy array of input values for x-axis.
    - output_dir: Directory path to save the plot.
    """
    # 2.1 What: Create a new figure with specified size.
    plt.figure(figsize=(10, 6))
    
    # 2.1 What: Compute tanh values for all inputs.
    y = tanh(z_range)
    
    # 2.1 What: Plot the tanh curve.
    plt.plot(z_range, y, 'g-', linewidth=2, label='Tanh(z)')
    
    # 2.1 What: Add horizontal reference lines.
    plt.axhline(y=0, color='red', linestyle=':', alpha=0.7, label='y=0 (center)')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    
    # 2.1 What: Add vertical reference line at z=0.
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2.1 What: Add labels and title.
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel('Output tanh(z)', fontsize=12)
    plt.title('Tanh Activation Function: tanh(z) = (e^z - e^-z) / (e^z + e^-z)', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2.1 What: Add annotations for saturation regions.
    plt.annotate('Saturation Region\n(gradient -> 0)', xy=(-5, -0.95), fontsize=9, color='red')
    plt.annotate('Saturation Region\n(gradient -> 0)', xy=(3.5, 0.95), fontsize=9, color='red')
    plt.annotate('Zero-centered!', xy=(0.2, 0.1), fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tanh_function.png', dpi=150)
    plt.close()
    print(f"[OK] Tanh function plot saved to {output_dir}/tanh_function.png")


def plot_tanh_derivative(z_range, output_dir):
    """
    Plots the Tanh derivative.
    
    Arguments:
    - z_range: NumPy array of input values for x-axis.
    - output_dir: Directory path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # 2.1 What: Compute derivative values.
    y = tanh_derivative(z_range)
    
    # 2.1 What: Plot the derivative curve.
    plt.plot(z_range, y, 'purple', linewidth=2, label="Tanh Derivative tanh'(z)")
    
    # 2.1 What: Add reference lines.
    plt.axhline(y=1.0, color='green', linestyle=':', alpha=0.7, label='Max gradient = 1.0')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel("Derivative tanh'(z)", fontsize=12)
    plt.title("Tanh Derivative: tanh'(z) = 1 - tanh^2(z)", fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2.1 What: Annotate maximum gradient point.
    plt.annotate('Maximum gradient\nat z=0 (value=1.0)', xy=(0, 1.0), xytext=(2, 0.8),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tanh_derivative.png', dpi=150)
    plt.close()
    print(f"[OK] Tanh derivative plot saved to {output_dir}/tanh_derivative.png")


def plot_combined(z_range, output_dir):
    """
    Plots both tanh function and its derivative on the same figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Tanh function
    y1 = tanh(z_range)
    ax1.plot(z_range, y1, 'g-', linewidth=2, label='Tanh(z)')
    ax1.axhline(y=0, color='red', linestyle=':', alpha=0.7)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Input (z)', fontsize=11)
    ax1.set_ylabel('Output tanh(z)', fontsize=11)
    ax1.set_title('Tanh Function', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.2, 1.2)
    
    # Right plot: Derivative
    y2 = tanh_derivative(z_range)
    ax2.plot(z_range, y2, 'purple', linewidth=2, label="Derivative tanh'(z)")
    ax2.axhline(y=1.0, color='green', linestyle=':', alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Input (z)', fontsize=11)
    ax2.set_ylabel("Derivative tanh'(z)", fontsize=11)
    ax2.set_title('Tanh Derivative', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tanh_combined.png', dpi=150)
    plt.close()
    print(f"[OK] Combined plot saved to {output_dir}/tanh_combined.png")


# ================================================
# NUMERICAL ANALYSIS
# ================================================

def numerical_analysis(output_dir):
    """
    Creates numerical analysis table and gradient analysis.
    """
    # 2.1 What: Define test inputs as specified in the problem.
    test_inputs = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])
    
    # 2.1 What: Compute tanh and derivative for each input.
    tanh_values = tanh(test_inputs)
    derivative_values = tanh_derivative(test_inputs)
    
    print("\n" + "=" * 60)
    print("NUMERICAL ANALYSIS TABLE")
    print("=" * 60)
    print(f"{'Input (z)':<12} {'Tanh(z)':<15} {'Derivative':<15}")
    print("-" * 42)
    
    for z, t, d in zip(test_inputs, tanh_values, derivative_values):
        print(f"{z:<12.1f} {t:<15.6f} {d:<15.6f}")
    
    # 2.1 What: Gradient analysis at specific points.
    print("\n" + "=" * 60)
    print("GRADIENT ANALYSIS AT x = -2, 0, 2")
    print("=" * 60)
    
    gradient_points = [-2, 0, 2]
    for x in gradient_points:
        grad = tanh_derivative(x)
        strength = "STRONG (> 0.1)" if grad > 0.1 else "WEAK (vanishing)"
        print(f"At x = {x:>2}: Gradient = {grad:.6f} -> {strength}")
    
    # 2.1 What: Identify strongest gradient region.
    print("\n" + "=" * 60)
    print("STRONGEST GRADIENT REGION")
    print("=" * 60)
    print("Gradients are strongest (> 0.1) in the range: approximately -2.5 < z < 2.5")
    print("Maximum gradient of 1.0 occurs at z = 0 (4x better than sigmoid!)")
    
    # 2.1 What: Identify saturation regions.
    print("\n" + "=" * 60)
    print("SATURATION REGIONS (Vanishing Gradient)")
    print("=" * 60)
    print("* Left saturation:  z < -3 (output approx -1, gradient approx 0)")
    print("* Right saturation: z > 3  (output approx +1, gradient approx 0)")
    print("* Still has vanishing gradient, but 4x better max than sigmoid!")
    
    # 2.1 What: Save analysis to file.
    with open(f'{output_dir}/numerical_analysis.md', 'w') as f:
        f.write("# Tanh Numerical Analysis\n\n")
        f.write("## Output Table\n\n")
        f.write("| Input (z) | Tanh(z) | Derivative |\n")
        f.write("|-----------|---------|------------|\n")
        for z, t, d in zip(test_inputs, tanh_values, derivative_values):
            f.write(f"| {z:.1f} | {t:.6f} | {d:.6f} |\n")
        f.write("\n## Gradient Analysis\n\n")
        f.write("| Point | Gradient | Strength |\n")
        f.write("|-------|----------|----------|\n")
        for x in gradient_points:
            grad = tanh_derivative(x)
            strength = "STRONG" if grad > 0.1 else "WEAK"
            f.write(f"| x = {x} | {grad:.6f} | {strength} |\n")
    
    print(f"\n[OK] Numerical analysis saved to {output_dir}/numerical_analysis.md")


# ================================================
# WRITTEN ANALYSIS
# ================================================

def written_analysis():
    """
    Prints written analysis about Tanh activation.
    """
    print("\n" + "=" * 60)
    print("WRITTEN ANALYSIS: TANH ACTIVATION FUNCTION")
    print("=" * 60)
    
    analysis = """
VANISHING GRADIENT PROBLEM:
---------------------------
Tanh also suffers from the vanishing gradient problem, but it's 
BETTER than sigmoid. The maximum gradient is 1.0 (at z=0), 
compared to sigmoid's 0.25. However, for |z| > 3, the gradient 
still approaches zero, causing learning to slow down.

For deep networks, vanishing gradient remains a problem:
- After 10 layers with sigmoid: 0.25^10 = 0.00000095
- After 10 layers with tanh: Even with max=1, quick decay at saturation

COMPARISON WITH SIGMOID:
------------------------
| Property         | Sigmoid     | Tanh        |
|------------------|-------------|-------------|
| Output Range     | (0, 1)      | (-1, 1)     |
| Zero-centered    | No          | YES         |
| Max Gradient     | 0.25        | 1.0         |
| Saturation       | |z| > 4     | |z| > 3     |

The key advantage of tanh is ZERO-CENTERED OUTPUT, which helps
with optimization because gradients don't all push in same direction.

WHEN TO USE TANH:
-----------------
+ RNNs and LSTMs (often used with gates)
+ When zero-centered output is important
+ Hidden layers in shallow networks
+ Normalization between -1 and 1

WHEN NOT TO USE TANH:
---------------------
- Deep networks (still vanishing gradient)
- Modern CNNs (use ReLU instead)
- When training speed is critical

SATURATION REGIONS:
-------------------
* z < -3: Output saturates near -1, gradient approx 0
* z > +3: Output saturates near +1, gradient approx 0
* These regions still "kill" gradients, but less severely than sigmoid.
"""
    print(analysis)


# ================================================
# MAIN EXECUTION
# ================================================

def main():
    """
    Main function to run all tanh activation analyses.
    """
    print("=" * 60)
    print("TANH ACTIVATION FUNCTION - FROM SCRATCH")
    print("=" * 60)
    
    # 2.1 What: Define output directory.
    output_dir = 'c:/masai/Tanh_Activation_Function/outputs'
    
    # 2.1 What: Create input range for plotting.
    z_range = np.linspace(-6, 6, 200)  # 200 points from -6 to 6
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_tanh_function(z_range, output_dir)
    plot_tanh_derivative(z_range, output_dir)
    plot_combined(z_range, output_dir)
    
    # Numerical analysis
    numerical_analysis(output_dir)
    
    # Written analysis
    written_analysis()
    
    print("\n" + "=" * 60)
    print("TANH ACTIVATION ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
