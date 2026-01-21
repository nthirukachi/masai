# Original Problem Statement

Investigate the dying ReLU problem by implementing both ReLU and Leaky ReLU, then comparing their behavior.

**Dataset:** Generate data with some negative features:

```python
import numpy as np
np.random.seed(42)
X_train = np.random.randn(1000, 10)  # 1000 samples, 10 features (can be negative)
y_train = (X_train[:, 0] + X_train[:, 1] - X_train[:, 2] > 0).astype(int)
```

**Tasks:**

1. Implement from scratch:
   - relu(z) and relu_derivative(z)
   - leaky_relu(z, alpha=0.01) and leaky_relu_derivative(z, alpha=0.01)

2. Build a simple 2-layer neural network class with:
   - Input layer → Hidden layer (20 neurons) → Output layer (1 neuron)
   - Implement forward propagation
   - Implement backward propagation
   - Support choosing between ReLU and Leaky ReLU for hidden layer

3. Train two versions:
   - Version 1: Using standard ReLU in hidden layer
   - Version 2: Using Leaky ReLU in hidden layer
   - Both should train for 200 epochs with learning_rate=0.01

4. Analysis and visualization:
   - Plot training loss curves for both versions
   - Count and report percentage of dead neurons (neurons with zero activation across all training samples) for each version
   - Compare final accuracy on training data
   - Explain when Leaky ReLU might be advantageous

**Expected Deliverables:**
- Complete neural network implementation with both activation options
- Training loss comparison plot
- Dead neuron analysis
- Written comparison (200-300 words)
