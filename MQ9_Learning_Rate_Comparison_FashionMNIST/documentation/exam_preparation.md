# 📝 Exam Preparation Questions

## Section A: Multiple Choice Questions (MCQ)

**Q1. What is the primary purpose of the Learning Rate?**
A) To determine the direction of the gradient.
B) To initialize the weights of the model.
C) To control the step size of weight updates.
D) To count the number of epochs.
**Answer:** C) To control the step size of weight updates.

**Q2. Which optimizer is generally considered "adaptive"?**
A) SGD (Stochastic Gradient Descent)
B) Adam
C) Constant Descent
D) None of the above
**Answer:** B) Adam

**Q3. If the validation loss increases while training loss decreases, what is happening?**
A) The model is learning perfectly.
B) The model is underfitting.
C) The model is fitting the noise (Overfitting).
D) The learning rate is too low.
**Answer:** C) The model is fitting the noise (Overfitting).

**Q4. What is `torch.nn.Flatten()` used for?**
A) To remove layers from the model.
B) To convert 2D image arrays into 1D vectors.
C) To normalize pixel values.
D) To calculate the loss.
**Answer:** B) To convert 2D image arrays into 1D vectors.

**Q5. Why do we clear gradients with `optimizer.zero_grad()`?**
A) To save memory.
B) Because PyTorch accumulates gradients by default.
C) To reset the model weights.
D) It is optional.
**Answer:** B) Because PyTorch accumulates gradients by default.

---

## Section B: Multiple Select Questions (MSQ)

**Q6. Which of the following are true about Learning Rate? (Select all that apply)**
A) A very high LR can cause the model to diverge.
B) A very low LR guarantees the fastest training.
C) LR is a hyperparameter.
D) LR should effectively be equal to the batch size.
**Answer:** A, C. (Low LR is slow, not fast. LR involves batch size but is not equal to it).

**Q7. Select the components needed for a PyTorch training loop.**
A) Optimizer
B) Loss Function (Criterion)
C) Model
D) Internet Connection
**Answer:** A, B, C.

---

## Section C: Numerical Problems

**Q8. Calculate the total number of updates per epoch.**
**Given:** Dataset size = 60,000 images. Batch Size = 100.
**Formula:** Updates = Total Images / Batch Size
**Calculation:** 60,000 / 100 = 600.
**Answer:** 600 updates per epoch.

**Q9. If a model has 10 input features and 5 neurons in the first hidden layer (fully connected), how many weights are in this layer (ignoring bias)?**
**Given:** Inputs = 10, Neurons = 5.
**Formula:** Weights = Inputs × Neurons
**Calculation:** 10 × 5 = 50.
**Answer:** 50 weights.

---

## Section D: Fill in the Blanks

**Q10. The function used to convert an image (0-255) to a float tensor (0-1) in PyTorch is called `__________`.**
**Answer:** `transforms.ToTensor()`

**Q11. `__________` is an activation function that returns 0 for negative inputs and x for positive inputs.**
**Answer:** ReLU (Rectified Linear Unit)

**Q12. To ensure our random numbers are the same every time, we set a `__________`.**
**Answer:** Seed
