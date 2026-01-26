# 🎓 Exam Preparation

## Section A: Multiple Choice Questions (MCQ)

**Q1. Which feature is MOST negatively correlated with Traffic Congestion?**
A) Vehicle Density
B) Queue Length
C) Average Vehicle Speed
D) Edge Density
> **Correct Answer:** C
> **Reason:** As traffic congestion increases, speed decreases. (Inverse relationship).

**Q2. Why do we split data into Train and Test sets?**
A) To make the code run faster
B) To check if the model can generalize to new data
C) To fix missing values
D) To increase accuracy on training data
> **Correct Answer:** B
> **Reason:** We must simulate "Future Data" (Test) to know if the model really learned or just memorized.

**Q3. What is a "False Negative" in traffic detection?**
A) Predicting Jam when it is Clear
B) Predicting Clear when it is Jam
C) Predicting Jam when it is Jam
D) Predicting Clear when it is Clear
> **Correct Answer:** B
> **Reason:** The model says "Negative" (No Jam), but it is False (Wrong). This is dangerous.

---

## Section B: Multiple Select Questions (MSQ)

**Q4. Which of the following are Supervised Learning algorithms? (Select 2)**
A) K-Means Clustering
B) Random Forest Classifier
C) Logistic Regression
D) PCA
> **Correct Answers:** B, C
> **Reason:** Both require labeled data (Target Y) to learn.

**Q5. How can we improve the model if accuracy is low? (Select 2)**
A) Collect more data (more rows)
B) Remove the 'Target' column
C) Tune Hyperparameters (e.g., more trees)
D) Use a random number generator for predictions
> **Correct Answers:** A, C
> **Reason:** More data helps learn patterns. Tuning the settings (hyperparameters) optimizes the brain.

---

## Section C: Numerical Problems

**Q6. Calculate Traffic Density.**
- **Given:** A road segment is 1 km long. There are 50 cars on it.
- **Formula:** Density = Vehicles / Length
- **Calculation:** 50 / 1
> **Answer:** 50 vehicles/km

**Q7. Calculate Accuracy.**
- Total Predictions: 100
- Correct Predictions: 85
- **Formula:** Accuracy = Correct / Total
- **Calculation:** 85 / 100
> **Answer:** 0.85 or 85%

---

## Section D: Fill in the Blanks

1.  **__________** is the process of putting all features on the same scale (e.g., 0 to 1). (Answer: Scaling / Normalization)
2.  A **__________** Matrix shows True Positives, False Positives, etc. (Answer: Confusion)
3.  The input variables (Speed, Density) are called **__________** and the output (Jam/No Jam) is called the **__________**. (Answer: Features, Target)
