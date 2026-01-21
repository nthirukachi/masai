# Original Problem Statement

Build a neural network to classify handwritten digits (MNIST) and analyze how activation function choice impacts performance, training time, and gradient flow.

**Dataset:** MNIST handwritten digits from keras/tensorflow:

```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 784) / 255.0  # Flatten and normalize
X_test = X_test.reshape(-1, 784) / 255.0
```

---

## Tasks:

### 1. Build three neural network variants using TensorFlow/Keras:

- Architecture: 784 input → 128 hidden → 64 hidden → 10 output
- Model A: Sigmoid activations in hidden layers
- Model B: Tanh activations in hidden layers
- Model C: ReLU activations in hidden layers
- All models: Use softmax activation in output layer, categorical cross-entropy loss

### 2. Training and monitoring:

- Train each model for 20 epochs with batch_size=128
- Track: training loss, validation loss, training accuracy, validation accuracy, training time per epoch
- Use the same optimizer (Adam with learning_rate=0.001) for all models

### 3. Comprehensive visualization:

- Plot training vs. validation accuracy for all three models (6 curves on one plot)
- Plot training vs. validation loss for all three models
- Create a bar chart comparing final test accuracies
- Plot training time per epoch for each model

### 4. Gradient analysis:

- After training, compute gradients of loss with respect to first layer weights for a batch of 32 samples
- Calculate mean absolute gradient magnitude for each model
- Visualize gradient magnitudes as a bar chart
- Explain which model suffers from vanishing gradients

### 5. Deep analysis (400-500 words):

- Compare convergence speed: Which model learned fastest?
- Compare final performance: Which achieved best test accuracy?
- Analyze gradient flow: Which model had strongest gradients in early layers?
- Explain the relationship between activation function derivatives and gradient vanishing
- Discuss practical implications: When might you choose sigmoid/tanh despite ReLU's advantages?
- Based on your results, provide recommendations for activation function selection

---

## Expected Deliverables:

- Python code for all three models with training loops
- 4-5 comprehensive visualization plots
- Comparison table with metrics (accuracy, loss, training time, gradient magnitude)
- Deep analysis report (400-500 words)
- Discussion of practical insights
