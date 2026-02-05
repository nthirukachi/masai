# 🎤 Interview Questions & Answers

## 1. What is a Learning Rate (LR)?
- **👶 Simple Answer:** The speed at which the model learns. If it's too fast, it makes mistakes. If it's too slow, it takes forever.
- **👨‍💻 Technical Answer:** A hyperparameter that scales the gradient during weight updates. It determines the step size in the loss landscape.
- **⚡ Common Mistake:** Thinking LR changes the *direction* of learning. (No, the Gradient determines direction; LR determines *distance*).

## 2. Why do we need a Validation Set?
- **👶 Simple Answer:** To test the student on questions they haven't seen before, to make sure they aren't just memorizing.
- **👨‍💻 Technical Answer:** To assess generalization performance and detect overfitting during training. It acts as a proxy for test data.

## 3. What happens if the Learning Rate is too high?
- **👶 Simple Answer:** The model jumps over the answer and never settles down.
- **👨‍💻 Technical Answer:** The optimizer overshoots the global minima, leading to divergence or oscillation around the minimum (high loss variance).

## 4. What happens if the Learning Rate is too low?
- **👶 Simple Answer:** It learns perfectly but takes 100 years.
- **👨‍💻 Technical Answer:** Convergence becomes extremely slow, and the model is more likely to get stuck in local minima or saddle points.

## 5. What is an Epoch?
- **👶 Simple Answer:** One full round of reading the entire textbook.
- **👨‍💻 Technical Answer:** One complete forward and backward pass of ALL training samples through the neural network.

## 6. Why use Adam optimizer instead of regular Gradient Descent?
- **👶 Simple Answer:** Adam is smarter—it speeds up when it's safe and slows down when it's tricky.
- **👨‍💻 Technical Answer:** Adam uses adaptive learning rates for each parameter (combining Momentum and RMSProp), allowing for faster convergence on complex landscapes.

## 7. What does the `zeros_grad()` function do?
- **👶 Simple Answer:** Clears the memory of the last step so we don't get confused.
- **👨‍💻 Technical Answer:** Resets the gradients of all model parameters to zero. Otherwise, PyTorch *accumulates* gradients (adds them up) by default.

## 8. Why do we use `ReLU` activation?
- **👶 Simple Answer:** It turns off negative signals (like noise) and keeps positive ones.
- **👨‍💻 Technical Answer:** It introduces non-linearity (f(x) = max(0, x)) effectively and efficiently, avoiding the vanishing gradient problem common with Sigmoid/Tanh.

## 9. How do we choose the best Learning Rate?
- **👶 Simple Answer:** Trial and error (Running experiments like we did!).
- **👨‍💻 Technical Answer:** Grid search, Random search, or using a Learning Rate Finder (increasing LR exponentially until loss diverges).

## 10. What does `loss.backward()` do?
- **👶 Simple Answer:** It looks at the mistake and calculates who is responsible.
- **👨‍💻 Technical Answer:** Computes the gradient of the loss with respect to all learnable parameters using the Chain Rule (Backpropagation).

## 11. What is Overfitting?
- **👶 Simple Answer:** Memorizing the answers instead of learning the logic.
- **👨‍💻 Technical Answer:** When the model learns noise in the training data, leading to low training loss but high validation loss.

## 12. Why set a Random Seed?
- **👶 Simple Answer:** To make sure our "random" shuffle is the same every time we play.
- **👨‍💻 Technical Answer:** To ensure deterministic behavior in initialization and data shuffling, making experiments reproducible.

## 13. What is Batch Size?
- **👶 Simple Answer:** How many questions the student answers before checking the answer key.
- **👨‍💻 Technical Answer:** The number of training samples used in one forward/backward pass. It affects gradient stability and memory usage.

## 14. Why normalize images to [0, 1]?
- **👶 Simple Answer:** To make the math easier for the computer.
- **👨‍💻 Technical Answer:** To keep gradients within a manageable range and ensure faster convergence (keeps the loss surface smoother).

## 15. What format is the Fashion-MNIST image?
- **👶 Simple Answer:** A collection of gray pictures.
- **👨‍💻 Technical Answer:** Can be (28, 28) grayscale images, represented as Tensors of shape [1, 28, 28].
