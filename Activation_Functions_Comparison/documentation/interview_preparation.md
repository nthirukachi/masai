# Interview Preparation: Activation Functions Comparison

## Quick Reference Card

### All Formulas
```
Sigmoid: sigma(z) = 1/(1+e^(-z))
Tanh:    tanh(z) = (e^z-e^(-z))/(e^z+e^(-z))
ReLU:    relu(z) = max(0, z)
```

### All Derivatives
```
Sigmoid': sigma(z) * (1 - sigma(z)), max = 0.25
Tanh':    1 - tanh^2(z), max = 1.0
ReLU':    1 if z > 0, else 0
```

---

## Top Interview Questions

### Q1: Compare sigmoid, tanh, and ReLU
**Answer**: 
- Sigmoid outputs (0,1) for probabilities but has max gradient 0.25
- Tanh outputs (-1,1), zero-centered, max gradient 1.0
- ReLU outputs [0,inf), gradient always 1 for positive, enabling deep networks

### Q2: Why did ReLU revolutionize deep learning?
**Answer**: ReLU has constant gradient = 1 for positive inputs. Unlike sigmoid (gradient shrinks to 0.0066 at z=5), ReLU maintains full gradient regardless of input magnitude, enabling training of 100+ layer networks.

### Q3: What are saturation regions?
**Answer**: 
- Sigmoid: |z| > 4, output near 0 or 1, gradient near 0
- Tanh: |z| > 3, output near -1 or 1, gradient near 0
- ReLU: No saturation for positive inputs

### Q4: What is the dead neuron problem?
**Answer**: In ReLU, if input is always negative, gradient is always 0, so the neuron never learns. Solutions: LeakyReLU, He initialization.

---

## Comparison Tables

### Output Properties
| Property | Sigmoid | Tanh | ReLU |
|----------|---------|------|------|
| Range | (0,1) | (-1,1) | [0,inf) |
| Zero-centered | No | Yes | No |
| Bounded | Yes | Yes | No |

### Gradient Properties
| Property | Sigmoid | Tanh | ReLU |
|----------|---------|------|------|
| Max gradient | 0.25 | 1.0 | 1.0 |
| At point | z=0 | z=0 | All z>0 |
| Vanishing | Yes | Yes | No |

### Use Cases
| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Binary output | Sigmoid | Probability interpretation |
| Hidden layers | ReLU | No vanishing gradient |
| RNNs | Tanh | Zero-centered, bounded |

---

## One-Page Quick Revision

### The Evolution
1. **Sigmoid** (1990s): First, but max gradient only 0.25
2. **Tanh** (2000s): Zero-centered, max gradient 1.0, still decays
3. **ReLU** (2012+): Gradient = 1 always, enabled deep learning

### Key Numbers
- Sigmoid max gradient: 0.25
- Tanh max gradient: 1.0
- ReLU gradient (positive): 1.0 always!

### Decision Rule
```
Output layer + probability? -> Sigmoid
Hidden layer? -> ReLU
RNN/LSTM? -> Tanh
Dead neurons issue? -> LeakyReLU
```

### Must-Remember
1. ReLU enabled deep learning (no vanishing gradient)
2. Sigmoid/Tanh suffer from vanishing gradient
3. ReLU has dead neurons (trade-off)
4. Use sigmoid for binary output only
