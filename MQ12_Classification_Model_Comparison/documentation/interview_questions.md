# 🎤 Interview Questions - Model Comparison

### 1. What is the difference between a Decision Tree and Random Forest?
- **Simple Answer**: A Decision Tree is like one person making a decision. Random Forest is 100 people voting on the decision.
- **Technical Answer**: A Decision Tree is a single estimator. Random Forest is an ensemble method that uses "Bagging" (Bootstrap Aggregating) to reduce variance and overfitting.

### 2. Why do we need to scale data for KNN but not for Decision Trees?
- **Simple Answer**: KNN calculates distances. If one feature has huge numbers, it will dominate the distance. Trees only care about "is it bigger or smaller?", so the actual size doesn't matter.
- **Technical Answer**: KNN is distance-dependent (Euclidean distance). Features with larger magnitudes will skew the distance calculation. Trees use non-parametric partitions based on feature thresholds.

### 3. When would you prefer Logistic Regression over a Random Forest?
- **Simple Answer**: When you need something very fast and simple, or when your data looks like it follows a straight line.
- **Technical Answer**: When the feature-target relationship is linear and high interpretability/predictive speed is required.

### 4. What does "K" representing in KNN?
- **Simple Answer**: The number of neighbors we look at to make a decision.
- **Technical Answer**: A hyperparameter representing the number of nearest neighbors in the feature space used for classification.

### 5. What is an "Ensemble" method?
- **Simple Answer**: Using many models together to get one better answer.
- **Technical Answer**: A technique that combines predictions from multiple base models to improve generalizability and robustness.
