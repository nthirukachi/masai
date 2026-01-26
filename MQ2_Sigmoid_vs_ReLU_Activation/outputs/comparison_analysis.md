
## Comparison Analysis: Sigmoid vs ReLU Activation (200-250 words)

This experiment compared **Sigmoid (logistic)** and **ReLU** activation functions on the make_moons dataset using identical MLPClassifier architectures with hidden layers (20, 20).

### Convergence Speed

The ReLU model used **300 iterations** while Sigmoid used **273 iterations**. This difference stems from their gradient behavior. Sigmoid's output is bounded between 0 and 1, causing gradients to become very small (approach zero) when inputs are far from zero—a phenomenon called **vanishing gradients**. ReLU, outputting max(0, x), maintains a constant gradient of 1 for positive values, allowing faster weight updates.

### Accuracy Comparison

ReLU achieved **95.83%** accuracy compared to Sigmoid's **87.50%**. Both models successfully learned the non-linear moon-shaped decision boundaries, but ReLU's faster training allowed it to find a slightly better local minimum within the iteration budget.

### Loss Analysis

The final training loss was **0.1105** for ReLU and **0.3060** for Sigmoid. The loss curves show ReLU dropping more steeply initially, demonstrating its computational advantage in early training epochs.

### Gradient Behavior Impact

The key insight is that **gradient flow** directly impacts learning efficiency. ReLU's linear gradient propagation enables deeper, faster learning, while Sigmoid's saturating nature can slow convergence, especially in deeper networks. For this shallow network (2 layers), both succeed, but ReLU's advantage would be more pronounced in deeper architectures.

### Conclusion

For most modern neural networks, ReLU is preferred due to its computational efficiency and resistance to vanishing gradients, though Sigmoid remains useful for binary output layers where probability interpretation is needed.
