# Problem Statement: Sigmoid Activation Function

## What Problem is Being Solved?

In neural networks, we need a way to introduce **non-linearity** into the model. Without activation functions, no matter how many layers we stack, the network would behave like a single linear transformation - unable to learn complex patterns.

The **Sigmoid activation function** solves this by:
1. **Squashing** any real-valued input to a bounded range (0, 1)
2. **Introducing non-linearity** that allows networks to learn complex patterns
3. **Providing probability-like outputs** useful for binary classification

## Why It Matters

| Use Case | Why Sigmoid is Relevant |
|----------|------------------------|
| Binary Classification | Output represents probability of class 1 |
| Logistic Regression | Core mathematical formulation |
| Gate Mechanisms (LSTM/GRU) | Values between 0-1 control information flow |
| Shallow Networks | Works well for 1-2 hidden layers |

## Real-World Relevance

- **Spam Detection**: Output probability (0.8 = 80% spam)
- **Medical Diagnosis**: Probability of disease presence
- **Credit Scoring**: Likelihood of loan default
- **Sentiment Analysis**: Probability of positive sentiment

---

## Steps to Solve the Problem

### Step 1: Implement Sigmoid Function
- Mathematical formula: σ(z) = 1 / (1 + e^(-z))
- Handle both scalar and array inputs using NumPy

### Step 2: Implement Sigmoid Derivative
- Formula: σ'(z) = σ(z) × (1 - σ(z))
- Needed for backpropagation

### Step 3: Create Visualizations
- Plot sigmoid function over range [-6, 6]
- Plot derivative to show gradient behavior
- Annotate saturation regions

### Step 4: Numerical Analysis
- Calculate outputs for test inputs [-5, -2, -0.5, 0, 0.5, 2, 5]
- Analyze gradient strength at key points
- Identify where gradients are strong (> 0.1)

### Step 5: Written Analysis
- Explain vanishing gradient problem
- Identify saturation regions
- Recommend use cases

---

## Expected Output

### Visualizations
1. **sigmoid_function.png**: S-shaped curve from 0 to 1
2. **sigmoid_derivative.png**: Bell-shaped curve, max at z=0
3. **sigmoid_combined.png**: Side-by-side comparison

### Numerical Table

| Input (z) | Sigmoid(z) | Derivative |
|-----------|------------|------------|
| -5.0 | 0.006693 | 0.006648 |
| -2.0 | 0.119203 | 0.104994 |
| -0.5 | 0.377541 | 0.235004 |
| 0.0 | 0.500000 | 0.250000 |
| 0.5 | 0.622459 | 0.235004 |
| 2.0 | 0.880797 | 0.104994 |
| 5.0 | 0.993307 | 0.006648 |

### Key Insights
- Maximum gradient of 0.25 occurs at z = 0
- Gradients become negligible for |z| > 4 (saturation)
- Output is never exactly 0 or 1, only asymptotically approaches them

---

## Exam Focus Points

1. **Formula**: σ(z) = 1 / (1 + e^(-z))
2. **Output Range**: Always between 0 and 1 (exclusive)
3. **Derivative**: σ'(z) = σ(z) × (1 - σ(z))
4. **Maximum Gradient**: 0.25 at z = 0
5. **Vanishing Gradient**: Occurs when |z| > 4
6. **Primary Use**: Binary classification output layer
