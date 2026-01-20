# ================================================
# ACTIVATION FUNCTIONS COMPARISON - FROM SCRATCH
# ================================================
# Question 14: Compare Activation Functions Mathematically and Visually
# 
# This file compares all three activation functions:
# Sigmoid, Tanh, and ReLU side-by-side.
# ================================================

import numpy as np  # 2.1 What: Imports NumPy for numerical operations
                    # 2.2 Why: Efficient array operations for all calculations

import matplotlib.pyplot as plt  # 2.1 What: Imports Matplotlib for visualization
                                  # 2.2 Why: Creating comparison plots

# ================================================
# ALL ACTIVATION FUNCTIONS (FROM SCRATCH)
# ================================================

def sigmoid(z):
    """
    Sigmoid activation: sigma(z) = 1 / (1 + e^(-z))
    Output range: (0, 1)
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Sigmoid derivative: sigma'(z) = sigma(z) * (1 - sigma(z))
    Max gradient: 0.25 at z=0
    """
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    """
    Tanh activation: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    Output range: (-1, 1), zero-centered
    """
    exp_z = np.exp(z)
    exp_neg_z = np.exp(-z)
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)

def tanh_derivative(z):
    """
    Tanh derivative: tanh'(z) = 1 - tanh^2(z)
    Max gradient: 1.0 at z=0
    """
    t = tanh(z)
    return 1 - t ** 2

def relu(z):
    """
    ReLU activation: f(z) = max(0, z)
    Output range: [0, infinity)
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    ReLU derivative: f'(z) = 1 if z > 0, else 0
    Gradient: 1 for all positive inputs (no vanishing!)
    """
    return np.where(z > 0, 1, 0).astype(float)


# ================================================
# COMPARISON VISUALIZATIONS
# ================================================

def plot_all_activations(z_range, output_dir):
    """
    Plots all three activation functions on the same graph.
    """
    plt.figure(figsize=(12, 7))
    
    # Compute all outputs
    y_sigmoid = sigmoid(z_range)
    y_tanh = tanh(z_range)
    y_relu = relu(z_range)
    
    # Plot all three
    plt.plot(z_range, y_sigmoid, 'b-', linewidth=2, label='Sigmoid')
    plt.plot(z_range, y_tanh, 'g-', linewidth=2, label='Tanh')
    plt.plot(z_range, y_relu, 'r-', linewidth=2, label='ReLU')
    
    # Reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel('Output', fontsize=12)
    plt.title('Comparison of Activation Functions', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.5, 6)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_activations.png', dpi=150)
    plt.close()
    print(f"[OK] All activations plot saved to {output_dir}/all_activations.png")


def plot_all_derivatives(z_range, output_dir):
    """
    Plots all three derivatives on the same graph.
    """
    plt.figure(figsize=(12, 7))
    
    # Compute all derivatives
    y_sigmoid_d = sigmoid_derivative(z_range)
    y_tanh_d = tanh_derivative(z_range)
    y_relu_d = relu_derivative(z_range)
    
    # Plot all three
    plt.plot(z_range, y_sigmoid_d, 'b-', linewidth=2, label="Sigmoid' (max=0.25)")
    plt.plot(z_range, y_tanh_d, 'g-', linewidth=2, label="Tanh' (max=1.0)")
    plt.plot(z_range, y_relu_d, 'r-', linewidth=2, label="ReLU' (0 or 1)")
    
    # Reference lines
    plt.axhline(y=0.25, color='blue', linestyle=':', alpha=0.5)
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel('Gradient', fontsize=12)
    plt.title('Comparison of Activation Derivatives (Gradients)', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.2)
    
    # Annotations
    plt.annotate('Sigmoid max = 0.25\n(Vanishing!)', xy=(0, 0.25), xytext=(2, 0.35),
                 fontsize=9, color='blue')
    plt.annotate('ReLU = 1 for all positive\n(No vanishing!)', xy=(3, 1), xytext=(3.5, 0.8),
                 fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_derivatives.png', dpi=150)
    plt.close()
    print(f"[OK] All derivatives plot saved to {output_dir}/all_derivatives.png")


def plot_side_by_side(z_range, output_dir):
    """
    Creates side-by-side comparison of functions and derivatives.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sigmoid
    axes[0, 0].plot(z_range, sigmoid(z_range), 'b-', linewidth=2)
    axes[0, 0].axhline(y=0.5, color='red', linestyle=':', alpha=0.7)
    axes[0, 0].set_title('Sigmoid: 1/(1+e^-z)', fontsize=11)
    axes[0, 0].set_ylabel('Output')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    axes[1, 0].plot(z_range, sigmoid_derivative(z_range), 'b-', linewidth=2)
    axes[1, 0].axhline(y=0.25, color='red', linestyle=':', alpha=0.7)
    axes[1, 0].set_title("Sigmoid' (max=0.25)", fontsize=11)
    axes[1, 0].set_ylabel('Gradient')
    axes[1, 0].set_xlabel('Input (z)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Tanh
    axes[0, 1].plot(z_range, tanh(z_range), 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='red', linestyle=':', alpha=0.7)
    axes[0, 1].set_title('Tanh: (e^z-e^-z)/(e^z+e^-z)', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-1.2, 1.2)
    
    axes[1, 1].plot(z_range, tanh_derivative(z_range), 'g-', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='red', linestyle=':', alpha=0.7)
    axes[1, 1].set_title("Tanh' (max=1.0)", fontsize=11)
    axes[1, 1].set_xlabel('Input (z)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # ReLU
    axes[0, 2].plot(z_range, relu(z_range), 'r-', linewidth=2)
    axes[0, 2].axhline(y=0, color='gray', linestyle=':', alpha=0.7)
    axes[0, 2].set_title('ReLU: max(0, z)', fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(-1, 7)
    
    axes[1, 2].plot(z_range, relu_derivative(z_range), 'r-', linewidth=2)
    axes[1, 2].axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    axes[1, 2].set_title("ReLU' (0 or 1)", fontsize=11)
    axes[1, 2].set_xlabel('Input (z)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(-0.2, 1.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/side_by_side_comparison.png', dpi=150)
    plt.close()
    print(f"[OK] Side-by-side comparison saved to {output_dir}/side_by_side_comparison.png")


# ================================================
# NUMERICAL ANALYSIS
# ================================================

def numerical_comparison(output_dir):
    """
    Creates complete numerical comparison table.
    """
    test_inputs = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])
    
    print("\n" + "=" * 80)
    print("COMPLETE NUMERICAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Input':<8} {'Sigmoid':<10} {'Sig.Grad':<10} {'Tanh':<10} {'Tanh.Grad':<10} {'ReLU':<10} {'ReLU.Grad':<10}")
    print("-" * 78)
    
    for z in test_inputs:
        print(f"{z:<8.1f} {sigmoid(z):<10.4f} {sigmoid_derivative(z):<10.4f} {tanh(z):<10.4f} {tanh_derivative(z):<10.4f} {relu(z):<10.1f} {relu_derivative(z):<10.1f}")
    
    # Gradient analysis
    print("\n" + "=" * 80)
    print("GRADIENT ANALYSIS AT x = -2, 0, 2")
    print("=" * 80)
    
    gradient_points = [-2, 0, 2]
    for x in gradient_points:
        sig_g = sigmoid_derivative(x)
        tanh_g = tanh_derivative(x)
        relu_g = relu_derivative(x)
        
        print(f"\nAt x = {x}:")
        print(f"  Sigmoid gradient: {sig_g:.4f} -> {'WEAK' if sig_g < 0.2 else 'OK'}")
        print(f"  Tanh gradient:    {tanh_g:.4f} -> {'WEAK' if tanh_g < 0.5 else 'STRONG' if tanh_g > 0.9 else 'OK'}")
        print(f"  ReLU gradient:    {relu_g:.1f} -> {'DEAD' if relu_g == 0 else 'PERFECT'}")
    
    # Save to markdown
    with open(f'{output_dir}/numerical_comparison_table.md', 'w') as f:
        f.write("# Numerical Comparison of All Activation Functions\n\n")
        f.write("## Complete Comparison Table\n\n")
        f.write("| Input | Sigmoid | Sig.Grad | Tanh | Tanh.Grad | ReLU | ReLU.Grad |\n")
        f.write("|-------|---------|----------|------|-----------|------|----------|\n")
        for z in test_inputs:
            f.write(f"| {z:.1f} | {sigmoid(z):.4f} | {sigmoid_derivative(z):.4f} | {tanh(z):.4f} | {tanh_derivative(z):.4f} | {relu(z):.1f} | {relu_derivative(z):.1f} |\n")
        
        f.write("\n## Key Insights\n\n")
        f.write("| Property | Sigmoid | Tanh | ReLU | Winner |\n")
        f.write("|----------|---------|------|------|--------|\n")
        f.write("| Max Gradient | 0.25 | 1.0 | 1.0 | ReLU/Tanh |\n")
        f.write("| Gradient Decay | Yes | Yes | No | **ReLU** |\n")
        f.write("| Zero-Centered | No | Yes | No | Tanh |\n")
        f.write("| Dead Neurons | No | No | Yes | Sigmoid/Tanh |\n")
        f.write("| Speed | Slow | Slow | Fast | **ReLU** |\n")
    
    print(f"\n[OK] Numerical comparison saved to {output_dir}/numerical_comparison_table.md")


# ================================================
# WRITTEN ANALYSIS
# ================================================

def written_analysis():
    """
    Comprehensive written analysis (200-300 words).
    """
    print("\n" + "=" * 80)
    print("WRITTEN ANALYSIS: ACTIVATION FUNCTIONS COMPARISON (200-300 WORDS)")
    print("=" * 80)
    
    analysis = """
VANISHING GRADIENT PROBLEM:
---------------------------
The derivative plots clearly show why vanishing gradient occurs. Sigmoid's 
maximum gradient is only 0.25 (at z=0), and it rapidly approaches zero for 
|z| > 4. Tanh is slightly better with a maximum of 1.0, but also decays 
quickly. When these gradients are multiplied across many layers during 
backpropagation, they become exponentially small, effectively stopping 
learning in early layers.

ReLU solves this by having a constant gradient of 1 for all positive inputs.
No matter how deep the network, gradients don't shrink (for positive paths).
This single property enabled training of networks with 100+ layers.

WHEN TO USE EACH ACTIVATION:
----------------------------
SIGMOID: Use for binary classification OUTPUT layers where probability 
interpretation is needed. Example: spam detection, medical diagnosis.

TANH: Use when zero-centered output is important, such as RNNs/LSTMs or 
when normalizing values to [-1, 1]. Better than sigmoid for hidden layers
but still suffers from vanishing gradient.

RELU: Default choice for hidden layers in modern deep networks including
CNNs, Transformers, and most architectures. Use unless you have specific
reasons not to.

SATURATION REGIONS:
-------------------
Sigmoid saturates for |z| > 4 (output near 0 or 1).
Tanh saturates for |z| > 3 (output near -1 or 1).
ReLU has no saturation for positive inputs, but the "dead zone" (z <= 0)
where output is always 0 creates dead neurons that never learn.

RECOMMENDATION: Use ReLU for hidden layers, sigmoid for binary output,
softmax for multi-class output. Consider LeakyReLU if dead neurons are 
a concern.
"""
    print(analysis)


# ================================================
# MAIN EXECUTION
# ================================================

def main():
    """
    Main function to run complete comparison analysis.
    """
    print("=" * 80)
    print("ACTIVATION FUNCTIONS COMPARISON - FROM SCRATCH")
    print("Sigmoid vs Tanh vs ReLU")
    print("=" * 80)
    
    output_dir = 'c:/masai/Activation_Functions_Comparison/outputs'
    z_range = np.linspace(-6, 6, 200)
    
    # Generate comparison plots
    print("\nGenerating comparison visualizations...")
    plot_all_activations(z_range, output_dir)
    plot_all_derivatives(z_range, output_dir)
    plot_side_by_side(z_range, output_dir)
    
    # Numerical analysis
    numerical_comparison(output_dir)
    
    # Written analysis
    written_analysis()
    
    print("\n" + "=" * 80)
    print("ACTIVATION FUNCTIONS COMPARISON COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
