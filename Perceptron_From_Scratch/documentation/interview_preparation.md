# Interview Preparation: Perceptron From Scratch

## 1. High-Level Project Summary
*   **Problem**: We predicted whether a student Passes or Fails based on Study Hours and Attendance.
*   **Solution**: We built a **Perceptron** (Single-Layer Neural Network) from scratch using Python/NumPy.
*   **Outcome**: The model learned a linear decision boundary with ~84% accuracy, successfully separating most data points.

## 2. Core Concepts – Interview & Exam View

### Perceptron
*   **What**: Simplest ANN unit; a linear binary classifier.
*   **Why**: To learn a decision boundary automatically.
*   **When to use**: Linearly separable binary data.
*   **When NOT to use**: Non-linear data (XOR), complex patterns, multi-class problems.

### Activation Function (Step)
*   **What**: Functions that converts input to output (0 or 1).
*   **Why**: To make the hard decision/classification.
*   **When to use**: Binary output required.
*   **When NOT to use**: When you need probabilities (use Sigmoid) or in hidden layers (use ReLU).

## 3. Frequently Asked Interview Questions

### Q: Why does the Perceptron converge?
*   **Answer**: The **Perceptron Convergence Theorem** guarantees convergence *if and only if* the data is legally separable.
*   **Analogy**: If you can separate red and blue marbles with a single straight stick, the Perceptron will eventually find a place to put the stick.

### Q: What is the "Bias" intuition?
*   **Answer**: Bias allows the classifier to say "Yes" even when inputs are zero.
*   **Analogy**: A toll booth fee. Even if you drive 0 miles (input 0), you might still have a cost (bias). Or, a base score in a test.

## 4. Parameter & Argument Questions

### `learning_rate` (in `Perceptron`)
*   **Why exists?**: To scale the weight updates.
*   **If removed?**: Updates would be equal to the input values (huge steps), likely causing instability.
*   **Default vs Custom**: Default 0.01 is standard safe start. Custom tuning is needed for different data scales.

### `epochs`
*   **Why exists?**: To allow the model to see data multiple times.
*   **If removed?**: One pass (online learning) is essentially just one epoch. Usually not enough to converge.

## 5. Comparisons (Exam Critical)

### Perceptron vs Logistic Regression
| Feature | Perceptron | Logistic Regression |
| :--- | :--- | :--- |
| **Output** | Hard 0 or 1 | Probability (0.0 to 1.0) |
| **Activation** | Step Function | Sigmoid Function |
| **Update** | On error only | Every step (Gradient Descent) |
| **Separability** | Must be Linear | Handles overlap better (probabilistic) |

### Training vs Testing
| Feature | Training | Testing |
| :--- | :--- | :--- |
| **Goal** | Learn weights | Evaluate performance |
| **Data** | Known labels used to update | Known labels used to check |
| **Updates** | Weights change | Weights FROZEN |

## 6. Common Mistakes & Traps

### Beginner Mistakes
*   **Forgetting Bias**: The line will be forced through (0,0) and likely fail to fit.
*   **Large Learning Rate**: Weights explode or oscillate.
*   **Data Scaling**: Perceptrons are sensitive to scale (e.g., Input 1=0.01, Input 2=1000). Standardization helps.

### Interview Trick Questions
*   **Q**: Can a perceptron learn the XOR function?
*   **A**: **NO**. XOR is not linearly separable. This famously caused the "AI Winter".

## 7. Output Interpretation Questions

### Q: How do you explain the output graph?
*   **A**: The graph shows a 2D space of Study Hours vs Attendance. The line is the "wall" our model built. If you fall on one side, you are classified Pass; the other, Fail.

### Q: What does it mean if accuracy is 80%?
*   **A**: It means 20% of the students defied the simple linear rule. They are the "exceptions" (nonlinearities).

### Q: What would you do next?
*   **A**: Try a more complex model (Logistic Regression, SVM, or MLP) to handle the non-linear outliers.

## 8. One-Page Quick Revision
*   **Equation**: $y = 1 \text{ if } (w \cdot x + b > 0) \text{ else } 0$.
*   **Update**: $w = w + \eta(y - \hat{y})x$.
*   **Condition**: Works ONLY on Linearly Separable data.
*   **Code Key**: `np.dot` for calculation, `np.where` for step function.
*   **Visual**: A straight line separating two point clouds.
