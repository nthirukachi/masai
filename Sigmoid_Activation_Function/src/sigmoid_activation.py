# ================================================
# SIGMOID ACTIVATION FUNCTION - FROM SCRATCH
# ================================================
# Question 14: Compare Activation Functions Mathematically and Visually
# 
# This file implements the Sigmoid activation function and its derivative
# without using any built-in activation functions.
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
# SIGMOID FUNCTION IMPLEMENTATION
# ================================================

def sigmoid(z):
    """
    Calculates the Sigmoid activation function.
    
    Formula: σ(z) = 1 / (1 + e^(-z))
    
    ⚙️ Arguments (Rule 3.1-3.7):
    ----------------------------
    - z:
        3.1 What: Input value(s) - can be a single number or NumPy array.
        3.2 Why: The weighted sum from previous layer that needs to be "squashed".
             This is the standard way; alternatives like hardcoded lookup tables
             exist but are less accurate.
        3.3 When: During forward propagation in neural networks.
        3.4 Where: Used in binary classification, logistic regression, 
             hidden layers of older neural networks.
        3.5 How: sigmoid(2.5) returns ~0.924, sigmoid(np.array([-1, 0, 1]))
        3.6 Internal: Computes e^(-z), adds 1, then takes reciprocal.
        3.7 Output Impact: Returns values between 0 and 1.
    
    Returns:
    --------
    float or np.ndarray: Sigmoid of input, always in range (0, 1).
    
    Example:
    --------
    >>> sigmoid(0)
    0.5
    >>> sigmoid(2)
    0.8807970779778823
    """
    # 2.1 What: Calculate sigmoid using the mathematical formula.
    # 2.2 Why: Squashes any real number to range (0, 1) - perfect for probabilities.
    #      This is the standard formula; no better mathematical alternative exists.
    # 2.3 When: Whenever we need probability-like outputs.
    # 2.4 Where: Output layer of binary classifiers, logistic regression.
    # 2.5 How: result = sigmoid(weighted_sum)
    # 2.6 Internal: np.exp(-z) computes e^(-z), then we add 1 and divide 1 by it.
    # 2.7 Output: For z=0, returns 0.5. For z=5, returns ~0.993.
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Calculates the derivative of the Sigmoid function.
    
    Formula: σ'(z) = σ(z) × (1 - σ(z))
    
    ⚙️ Arguments (Rule 3.1-3.7):
    ----------------------------
    - z:
        3.1 What: Input value(s) at which to compute the derivative.
        3.2 Why: Needed for backpropagation to compute gradients.
             This formula is derived mathematically; no alternative.
        3.3 When: During backward pass (training) in neural networks.
        3.4 Where: Used in gradient descent optimization.
        3.5 How: sigmoid_derivative(0) returns 0.25 (maximum gradient).
        3.6 Internal: First computes sigmoid(z), then multiplies by (1 - sigmoid(z)).
        3.7 Output Impact: Returns values between 0 and 0.25 (maximum at z=0).
    
    Returns:
    --------
    float or np.ndarray: Derivative of sigmoid at input z.
    
    Example:
    --------
    >>> sigmoid_derivative(0)
    0.25
    >>> sigmoid_derivative(5)
    0.006648056670790155
    """
    # 2.1 What: First compute the sigmoid value.
    # 2.2 Why: The derivative formula requires the sigmoid output.
    # 2.6 Internal: Stores sigmoid(z) to avoid computing it twice.
    s = sigmoid(z)
    
    # 2.1 What: Apply the derivative formula σ(z) × (1 - σ(z)).
    # 2.2 Why: This is the mathematically derived gradient formula.
    # 2.6 Internal: Element-wise multiplication for arrays.
    # 2.7 Output: Maximum value 0.25 occurs at z=0; decreases towards 0 at extremes.
    return s * (1 - s)


# ================================================
# VISUALIZATION FUNCTIONS
# ================================================

def plot_sigmoid_function(z_range, output_dir):
    """
    Plots the Sigmoid function.
    
    ⚙️ Arguments:
    - z_range: NumPy array of input values for x-axis.
    - output_dir: Directory path to save the plot.
    """
    # 2.1 What: Create a new figure (canvas) with specified size.
    #      plt.figure() creates an empty canvas for drawing plots.
    #      figsize=(10, 6) means 10 inches wide x 6 inches tall.
    #
    # 2.2 Why: Controls the dimensions of the output image.
    #      - Default figsize is (6.4, 4.8) if not specified.
    #      - We use (10, 6) for wide, professional-looking plots.
    #      - Alternative: plt.figure(figsize=(4, 3)) for smaller thumbnails.
    #
    # 2.3 When: Always call before drawing any plot if you need custom size.
    #
    # 2.4 Where: Used in data visualization, reports, presentations.
    #
    # 2.5 How to use (Examples):
    #      plt.figure(figsize=(10, 6))  # Wide figure (our choice)
    #      plt.figure(figsize=(8, 8))   # Square figure
    #      plt.figure(figsize=(14, 5))  # Very wide for side-by-side
    #
    # 2.6 How it works internally:
    #      1. Matplotlib creates a Figure object in memory
    #      2. figsize sets the physical dimensions in inches
    #      3. dpi (dots per inch) determines pixel resolution when saved
    #      4. Default dpi=100, so (10, 6) = 1000x600 pixels
    #
    # 2.7 Output: No visible output, but creates canvas ready for drawing.
    plt.figure(figsize=(10, 6))
    
    # 2.1 What: Compute sigmoid values for all inputs.
    y = sigmoid(z_range)
    
    # 2.1 What: Plots x-y data as a line graph on the current figure.
    #      This draws the sigmoid curve using z_range as x-axis and y as y-axis.
    #
    # 2.2 Why: To visualize how sigmoid transforms inputs to outputs.
    #      Visual representation helps understand the S-curve shape.
    #      Alternative: plt.scatter() for points, but line is better for continuous functions.
    #
    # 2.3 When: After creating a figure and computing y values.
    #
    # 2.4 Where: Data visualization, function plotting, ML model analysis.
    #
    # 2.5 How to use (syntax):
    #      plt.plot(x_data, y_data, format_string, **kwargs)
    #
    # 2.6 How it works internally:
    #      1. Matplotlib takes x,y coordinate pairs
    #      2. Connects them with line segments
    #      3. Applies styling (color, width, etc.)
    #      4. Adds to current axes object
    #
    # 2.7 Output: Draws line on canvas; returns list of Line2D objects.
    #
    # ⚙️ Arguments Explanation (3.1-3.7):
    # -----------------------------------
    # ARGUMENT 1: z_range (x-axis data)
    #   3.1 What: NumPy array of x-coordinates (our input values)
    #   3.2 Why: Defines where to plot points along horizontal axis
    #   3.5 Example: np.linspace(-6, 6, 200) creates 200 points from -6 to 6
    #   3.7 Impact: More points = smoother curve
    #
    # ARGUMENT 2: y (y-axis data)
    #   3.1 What: NumPy array of y-coordinates (sigmoid outputs)
    #   3.2 Why: Defines vertical position of each point
    #   3.5 Example: y = sigmoid(z_range)
    #   3.7 Impact: Shape of the curve depends on this
    #
    # ARGUMENT 3: 'b-' (format string)
    #   3.1 What: Shorthand for color and line style
    #   3.2 Why: Quick way to specify appearance
    #   3.5 Examples:
    #        'b-'  = blue solid line
    #        'r--' = red dashed line
    #        'g.'  = green dots
    #        'ko'  = black circles
    #   3.6 Format: '[color][marker][linestyle]'
    #        Colors: b=blue, r=red, g=green, k=black, m=magenta
    #        Markers: o=circle, s=square, ^=triangle, .=point
    #        Lines: -=solid, --=dashed, :=dotted, -.=dash-dot
    #
    # ARGUMENT 4: linewidth=2
    #   3.1 What: Thickness of the line in points
    #   3.2 Why: Thicker lines are more visible in presentations
    #   3.5 Examples: linewidth=1 (thin), linewidth=2 (medium), linewidth=4 (thick)
    #   3.7 Default: 1.5 if not specified
    #
    # ARGUMENT 5: label='Sigmoid σ(z)'
    #   3.1 What: Text label for this line (used in legend)
    #   3.2 Why: Identifies this line when plt.legend() is called
    #   3.5 Example: label='My Data' then call plt.legend() to show it
    #   3.7 Impact: Won't appear unless plt.legend() is called
    plt.plot(z_range, y, 'b-', linewidth=2, label='Sigmoid σ(z)')
    
    # 2.1 What: Add horizontal reference lines.
    # 2.2 Why: Show the asymptotic bounds (0 and 1).
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # 2.1 What: Draws a horizontal line across the entire width of the plot.
    #      "axhline" = "axes horizontal line" - spans full x-axis range.
    #
    # 2.2 Why: To show the sigmoid midpoint at y=0.5 (where sigmoid(0) = 0.5).
    #      This is an important reference because:
    #      - sigmoid(0) = 0.5 exactly (the center of the S-curve)
    #      - Values above 0.5 indicate positive input, below indicate negative
    #      Alternative: plt.plot() with specific coordinates, but axhline is simpler.
    #
    # 2.3 When: After plotting main data, to add reference markers.
    #
    # 2.4 Where: Used to show thresholds, baselines, decision boundaries.
    #
    # 2.5 How to use (syntax):
    #      plt.axhline(y=value, color='...', linestyle='...', alpha=..., label='...')
    #
    # 2.6 How it works internally:
    #      1. Creates a Line2D object from x=0 to x=1 (full axes width)
    #      2. Transforms to data coordinates
    #      3. Applies styling and adds to axes
    #
    # 2.7 Output: Draws horizontal line; returns Line2D object.
    #
    # ⚙️ Arguments Explanation (3.1-3.7):
    # -----------------------------------
    # ARGUMENT 1: y=0.5
    #   3.1 What: Y-coordinate where the horizontal line is drawn
    #   3.2 Why: 0.5 is the midpoint of sigmoid output range (0 to 1)
    #   3.5 Examples: y=0 (bottom), y=0.5 (middle), y=1 (top)
    #   3.7 Impact: Determines vertical position of the line
    #
    # ARGUMENT 2: color='red'
    #   3.1 What: Color of the line
    #   3.2 Why: Red stands out against blue sigmoid curve
    #   3.5 Examples: 'red', 'blue', 'green', '#FF5733' (hex), (1,0,0) (RGB)
    #   3.7 Default: Current color cycle if not specified
    #
    # ARGUMENT 3: linestyle=':'
    #   3.1 What: Style of the line (dotted in this case)
    #   3.2 Why: Dotted line indicates it's a reference, not data
    #   3.5 Examples:
    #        '-'  = solid line
    #        '--' = dashed line
    #        ':'  = dotted line (our choice)
    #        '-.' = dash-dot line
    #   3.7 Default: '-' (solid) if not specified
    #
    # ARGUMENT 4: alpha=0.7
    #   3.1 What: Transparency level (0=invisible, 1=opaque)
    #   3.2 Why: Semi-transparent so it doesn't obscure the main curve
    #   3.5 Examples: alpha=0.3 (very faint), alpha=0.7 (visible), alpha=1.0 (solid)
    #   3.7 Default: 1.0 (fully opaque) if not specified
    #
    # ARGUMENT 5: label='y=0.5 (midpoint)'
    #   3.1 What: Text to show in the legend for this line
    #   3.2 Why: Explains what this reference line represents
    #   3.5 Example: Appears in legend when plt.legend() is called
    #   3.7 Impact: Won't show unless plt.legend() is called
    plt.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, label='y=0.5 (midpoint)')
    
    # 2.1 What: Add vertical reference line at z=0.
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2.1 What: Add labels and title.
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel('Output σ(z)', fontsize=12)
    plt.title('Sigmoid Activation Function: σ(z) = 1 / (1 + e⁻ᶻ)', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2.1 What: Add annotations for saturation regions.
    plt.annotate('Saturation Region\n(gradient ≈ 0)', xy=(-5, 0.05), fontsize=9, color='red')
    plt.annotate('Saturation Region\n(gradient ≈ 0)', xy=(3.5, 0.95), fontsize=9, color='red')
    
    # 2.1 What: Save the figure.
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sigmoid_function.png', dpi=150)
    plt.close()
    print(f"[OK] Sigmoid function plot saved to {output_dir}/sigmoid_function.png")


def plot_sigmoid_derivative(z_range, output_dir):
    """
    Plots the Sigmoid derivative.
    
    ⚙️ Arguments:
    - z_range: NumPy array of input values for x-axis.
    - output_dir: Directory path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # 2.1 What: Compute derivative values.
    y = sigmoid_derivative(z_range)
    
    # 2.1 What: Plot the derivative curve.
    plt.plot(z_range, y, 'r-', linewidth=2, label="Sigmoid Derivative σ'(z)")
    
    # 2.1 What: Add reference lines.
    plt.axhline(y=0.25, color='green', linestyle=':', alpha=0.7, label='Max gradient = 0.25')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel("Derivative σ'(z)", fontsize=12)
    plt.title("Sigmoid Derivative: σ'(z) = σ(z) × (1 - σ(z))", fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2.1 What: Annotate maximum gradient point.
    plt.annotate('Maximum gradient\nat z=0', xy=(0, 0.25), xytext=(2, 0.22),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sigmoid_derivative.png', dpi=150)
    plt.close()
    print(f"[OK] Sigmoid derivative plot saved to {output_dir}/sigmoid_derivative.png")


def plot_combined(z_range, output_dir):
    """
    Plots both sigmoid function and its derivative on the same figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Sigmoid function
    y1 = sigmoid(z_range)
    ax1.plot(z_range, y1, 'b-', linewidth=2, label='Sigmoid σ(z)')
    ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.7)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Input (z)', fontsize=11)
    ax1.set_ylabel('Output σ(z)', fontsize=11)
    ax1.set_title('Sigmoid Function', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Right plot: Derivative
    y2 = sigmoid_derivative(z_range)
    ax2.plot(z_range, y2, 'r-', linewidth=2, label="Derivative σ'(z)")
    ax2.axhline(y=0.25, color='green', linestyle=':', alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Input (z)', fontsize=11)
    ax2.set_ylabel("Derivative σ'(z)", fontsize=11)
    ax2.set_title('Sigmoid Derivative', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sigmoid_combined.png', dpi=150)
    plt.close()
    print(f"[OK] Combined plot saved to {output_dir}/sigmoid_combined.png")


# ================================================
# NUMERICAL ANALYSIS
# ================================================

def numerical_analysis(output_dir):
    """
    Creates numerical analysis table and gradient analysis.
    """
    # 2.1 What: Define test inputs as specified in the problem.
    test_inputs = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])
    
    # 2.1 What: Compute sigmoid and derivative for each input.
    sigmoid_values = sigmoid(test_inputs)
    derivative_values = sigmoid_derivative(test_inputs)
    
    print("\n" + "=" * 60)
    print("NUMERICAL ANALYSIS TABLE")
    print("=" * 60)
    print(f"{'Input (z)':<12} {'Sigmoid(z)':<15} {'Derivative':<15}")
    print("-" * 42)
    
    for z, s, d in zip(test_inputs, sigmoid_values, derivative_values):
        print(f"{z:<12.1f} {s:<15.6f} {d:<15.6f}")
    
    # 2.1 What: Gradient analysis at specific points.
    print("\n" + "=" * 60)
    print("GRADIENT ANALYSIS AT x = -2, 0, 2")
    print("=" * 60)
    
    gradient_points = [-2, 0, 2]
    for x in gradient_points:
        grad = sigmoid_derivative(x)
        strength = "STRONG (> 0.1)" if grad > 0.1 else "WEAK (vanishing)"
        print(f"At x = {x:>2}: Gradient = {grad:.6f} -> {strength}")
    
    # 2.1 What: Identify strongest gradient region.
    print("\n" + "=" * 60)
    print("STRONGEST GRADIENT REGION")
    print("=" * 60)
    print("Gradients are strongest (> 0.1) in the range: approximately -2 < z < 2")
    print("Maximum gradient of 0.25 occurs at z = 0")
    
    # 2.1 What: Identify saturation regions.
    print("\n" + "=" * 60)
    print("SATURATION REGIONS (Vanishing Gradient)")
    print("=" * 60)
    print("* Left saturation:  z < -4 (output approx 0, gradient approx 0)")
    print("* Right saturation: z > 4  (output approx 1, gradient approx 0)")
    print("* In these regions, learning becomes extremely slow!")
    
    # 2.1 What: Save analysis to file.
    with open(f'{output_dir}/numerical_analysis.md', 'w') as f:
        f.write("# Sigmoid Numerical Analysis\n\n")
        f.write("## Output Table\n\n")
        f.write("| Input (z) | Sigmoid(z) | Derivative |\n")
        f.write("|-----------|------------|------------|\n")
        for z, s, d in zip(test_inputs, sigmoid_values, derivative_values):
            f.write(f"| {z:.1f} | {s:.6f} | {d:.6f} |\n")
        f.write("\n## Gradient Analysis\n\n")
        f.write("| Point | Gradient | Strength |\n")
        f.write("|-------|----------|----------|\n")
        for x in gradient_points:
            grad = sigmoid_derivative(x)
            strength = "STRONG" if grad > 0.1 else "WEAK"
            f.write(f"| x = {x} | {grad:.6f} | {strength} |\n")
    
    print(f"\n[OK] Numerical analysis saved to {output_dir}/numerical_analysis.md")


# ================================================
# WRITTEN ANALYSIS
# ================================================

def written_analysis():
    """
    Prints written analysis about Sigmoid activation.
    """
    print("\n" + "=" * 60)
    print("WRITTEN ANALYSIS: SIGMOID ACTIVATION FUNCTION")
    print("=" * 60)
    
    analysis = """
VANISHING GRADIENT PROBLEM:
    ---------------------------
    The sigmoid function suffers from the "vanishing gradient" problem. 
    Looking at the derivative plot, we can see that the maximum gradient 
    is only 0.25 (at z=0), and it quickly approaches 0 as we move away 
    from the origin. For |z| > 4, the gradient is essentially zero.
    
    This means during backpropagation, when gradients are multiplied 
    across many layers, they become exponentially smaller. This makes 
    training deep networks extremely slow or impossible with sigmoid.
    
WHEN TO USE SIGMOID:
--------------------
+ Binary classification (output layer) - probability interpretation
+ Logistic regression
+ Gating mechanisms (LSTM, GRU) - values between 0 and 1 are useful
+ Shallow networks (1-2 hidden layers)

WHEN NOT TO USE SIGMOID:
------------------------
- Hidden layers of deep networks (use ReLU instead)
- When inputs are large (saturation occurs)
- When fast training is required

SATURATION REGIONS:
    -------------------
    * z < -4: Output saturates near 0, gradient approx 0
    * z > +4: Output saturates near 1, gradient approx 0
    * These regions "kill" gradients, stopping learning.
    """
    print(analysis)


# ================================================
# MAIN EXECUTION
# ================================================

def main():
    """
    Main function to run all sigmoid activation analyses.
    """
    print("=" * 60)
    print("SIGMOID ACTIVATION FUNCTION - FROM SCRATCH")
    print("=" * 60)
    
    # 2.1 What: Define output directory.
    output_dir = 'c:/masai/Sigmoid_Activation_Function/outputs'
    
    # 2.1 What: Create input range for plotting.
    # 2.2 Why: We need to visualize the function over a range.
    z_range = np.linspace(-6, 6, 200)  # 200 points from -6 to 6
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_sigmoid_function(z_range, output_dir)
    plot_sigmoid_derivative(z_range, output_dir)
    plot_combined(z_range, output_dir)
    
    # Numerical analysis
    numerical_analysis(output_dir)
    
    # Written analysis
    written_analysis()
    
    print("\n" + "=" * 60)
    print("SIGMOID ACTIVATION ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
