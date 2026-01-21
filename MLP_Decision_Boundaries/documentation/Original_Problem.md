# Original Problem Statement

## Build a simple neural network and visualize how different activation functions create different decision boundaries on a non-linearly separable dataset.

---

### Dataset
Use sklearn's make_moons dataset:

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
```

---

### Tasks

1. **Build a simple neural network using sklearn's MLPClassifier with:**
   - Architecture: 1 hidden layer with 8 neurons
   - Three versions: one with relu, one with logistic (sigmoid), one with tanh
   - Same random_state for fair comparison

2. **Train all three models on the moons dataset**

3. **Create a comprehensive visualization showing:**
   - 3 subplots (one for each activation)
   - Each subplot shows the decision boundary (use contour plot)
   - Scatter plot of training data overlaid on decision boundary
   - Title showing activation function and training accuracy

4. **Analysis:**
   - Compare the shapes of decision boundaries
   - Which activation achieved highest accuracy?
   - Describe the visual differences in how each activation carves up the space
   - Explain why certain activations might perform better on this dataset

---

### Expected Deliverables

- Python code to train three models
- Visualization with 3 subplots showing decision boundaries
- Comparison table with accuracies
- Written analysis (250-350 words)
