# 📝 Exam Preparation

## Section A: Multiple Choice Questions (MCQ)

**Q1. What is the primary purpose of fitting StandardScaler ONLY on training data?**
A) To save memory
B) To make the code run faster
C) To prevent data leakage
D) It doesn't matter, we can fit on everything
> **Correct Answer: C**  
> **Explanation:** If we fit on test data, our model "peeks" at the distribution of the unknown data, which is cheating.

**Q2. How many new features does K-Means augmentation add if K=4?**
A) 4
B) 8 (4 one-hot + 4 distances)
C) 13
D) 1
> **Correct Answer: B**  
> **Explanation:** We add k binary columns (membership) AND k float columns (distances).

**Q3. Which metric is best for imbalanced binary classification?**
A) Accuracy
B) F1-Score
C) Mean Squared Error
D) R-Squared
> **Correct Answer: B**  
> **Explanation:** Accuracy can be misleading if one class dominates. F1 considers both Precision and Recall.

**Q4. Perceptron is a \_\_\_\_\_\_\_\_ classifier.**
A) Non-linear
B) Linear
C) Tree-based
D) Instance-based
> **Correct Answer: B**  
> **Explanation:** Perceptron draws a straight line (or hyperplane) decision boundary.

---

## Section B: Multiple Select Questions (MSQ)

**Q5. Which of the following are unsupervised learning algorithms? (Select all that apply)**
- [x] K-Means
- [ ] Perceptron
- [x] PCA (Principal Component Analysis)
- [ ] Logistic Regression
> **Explanation:** K-Means and PCA do not use labels (y). Perceptron and Logistic Regression need labels.

**Q6. When is Feature Augmentation useful? (Select all that apply)**
- [x] When the original features are insufficient
- [x] When we have domain knowledge about subgroups
- [ ] When the dataset is massive (billions of rows)
- [ ] When the model is overfitting
> **Explanation:** Augmentation adds information. It usually *increases* risk of overfitting and computational cost (bad for massive data), but helps weak features.

---

## Section C: Numerical Problems

**Q7. Calculate the Euclidean Distance.**
Point A: (1, 2)  
Centroid C: (4, 6)  
Calculate distance.

**Solution:**
$$ d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} $$
$$ d = \sqrt{(4 - 1)^2 + (6 - 2)^2} $$
$$ d = \sqrt{3^2 + 4^2} $$
$$ d = \sqrt{9 + 16} $$
$$ d = \sqrt{25} = 5 $$

**Final Answer:** 5

**Q8. 5-Fold Cross Validation Sizes.**
Total Samples: 200
How many samples in Training Set and Test Set for one fold?

**Solution:**
Test Set = $1/5 = 20\%$
$200 \times 0.20 = 40$ samples

Training Set = $4/5 = 80\%$
$200 \times 0.80 = 160$ samples

**Final Answer:** Train=160, Test=40

---

## Section D: Fill in the Blanks

**Q9.** The input to K-Means should always be \_\_\_\_\_\_\_\_ to ensure features contribute equally to distances.
> **Answer:** Scaled / Standardized / Normalized

**Q10.** A p-value less than \_\_\_\_\_\_\_\_ typically indicates statistical significance.
> **Answer:** 0.05

**Q11.** \_\_\_\_\_\_\_\_ is the process of generating new features from existing data to improve model performance.
> **Answer:** Feature Engineering
