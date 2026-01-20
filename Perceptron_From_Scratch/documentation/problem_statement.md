# Problem Statement: Predicting Student Success with a Perceptron

## 1. Problem Statement

### 1.1 Only Definition
We need to predict whether a student will **Pass** or **Fail** an exam based on two input features:
1.  **Study Hours**: How many hours they studied.
2.  **Attendance**: Their class attendance percentage.

We want to build a simple machine learning model (a Perceptron) that learns a "rule" or "boundary" to separate students who pass from those who fail.

### 1.2 Why this problem exists?
In schools and universities, it is helpful to identify students who are at risk of failing *early*, so teachers can help them. Human teachers have intuition ("if you study little and miss class, you fail"), but we want a computer to learn this pattern mathematically from data.

### 1.3 Real-world Relevance
-   **Education**: Early warning systems for student dropout.
-   **Finance**: Approving or rejecting a loan (Income vs. Debt).
-   **Health**: Diagnosing a disease (High vs. Low blood pressure).
-   **Marketing**: Predicting if a customer will buy (Time on site vs. Age).
This simple "Yes/No" classification is the foundation of almost all AI.

---

## 2. Steps to Solve the Problem

We will follow a standard Machine Learning approach:

1.  **Generate Data**: Create a synthetic dataset of 100 students with study hours and attendance. We will "label" them as Pass (1) or Fail (0) based on a simple rule (for the computer to rediscover).
2.  **Initialize Perceptron**: Create a "brain" (Perceptron) with random thoughts (random weights).
3.  **Train (Learn)**:
    *   Show the Perceptron one student at a time.
    *   Ask it to predict: "Pass or Fail?"
    *   If it's **wrong**, adjust its "thoughts" (update weights) so it performs better next time.
    *   Repeat this many times (epochs).
4.  **Visualize**: Draw a line (decision boundary) to see how well it separates the Passing students from the Failing ones.
5.  **Test**: Ask the model to predict for new students (e.g., Student A, B, C).

---

## 3. Expected Output

### 3.1 What output is expected?
1.  **Learned Weights**: Numbers that represent how important "Study Hours" and "Attendance" are.
2.  **Accuracy Score**: A percentage (e.g., 90%) telling us how often the model is correct.
3.  **Decision Boundary Plot**: A graph showing red dots (Fail) and blue dots (Pass), separated by a line drawn by our Perceptron.
4.  **Convergence Plot**: A graph showing that the number of mistakes decreases as the model learns (the line goes down).

### 3.2 Sample Output Explanation
If the model outputs:
*   Weight for Study Hours = 2.0
*   Weight for Attendance = 0.5
*   Bias = -80

This means the "rule" it learned is roughly:
$$ (2.0 \times \text{Study}) + (0.5 \times \text{Attendance}) - 80 > 0 $$
If this sum is positive, the student passes.

---

## 4. Exam Focus Points

*   **Q:** What sort of problem is this?
    *   **A:** Binary Classification (Two classes: Pass/Fail).
*   **Q:** What algorithm are we using?
    *   **A:** The Perceptron (the simplest neural network).
*   **Q:** Can a Perceptron solve ANY problem?
    *   **A:** No, only **Linearly Separable** problems (where a straight line can split the groups).
