Build a fraud detection system for credit card transactions, experimenting with different activation functions and network architectures to handle highly imbalanced data.

Dataset: Credit Card Fraud Detection from Kaggle:
- Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Alternative: Use synthetic imbalanced data if dataset unavailable:

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=50000, n_features=30, n_informative=20,
                           n_redundant=10, n_classes=2, weights=[0.98, 0.02],
                           flip_y=0.01, random_state=42)

Context: Credit card fraud detection is challenging because:
- Data is highly imbalanced (~0.17% fraud cases in real data)
- False positives (blocking legitimate transactions) are costly
- False negatives (missing fraud) are very costly
- Real-time prediction requires fast inference

Tasks:
1. Data preprocessing:
   - Load and explore the dataset (check class distribution, feature statistics)
   - Split into train (60%), validation (20%), test (20%) sets
   - Apply StandardScaler normalization
   - Handle class imbalance using SMOTE or class weights

2. Build and compare multiple neural network architectures:
   - Model 1 Shallow-Wide Network: Input -> 64 neurons -> 32 neurons -> 1 output (Hidden activation: ReLU)
   - Model 2 Deep-Narrow Network: Input -> 32 -> 32 -> 32 -> 32 -> 1 output (Hidden activation: ReLU)
   - Model 3 Hybrid Activation Network: Input -> 64 (ReLU) -> 32 (ReLU) -> 16 (tanh) -> 1 output
   - Model 4 Your Custom Design: Design your own architecture based on the problem characteristics and justify your choices

3. Training configuration:
   - Use binary cross-entropy loss with class weights
   - Optimizer: Adam with learning_rate=0.001
   - Train for 50 epochs with early stopping (patience=5)
   - Track: Precision, Recall, F1-score, AUC-ROC for each model

4. Comprehensive evaluation:
   - Generate classification reports for all models on test set
   - Plot ROC curves for all models on the same graph
   - Plot Precision-Recall curves (important for imbalanced data)
   - Create confusion matrices for each model
   - Measure inference time per 1000 predictions

5. Activation function ablation study:
   - Take your best-performing architecture
   - Train 4 versions with different hidden activations: sigmoid, tanh, relu, leaky_relu
   - Compare training convergence, final metrics, and inference speed
   - Identify which activation works best for this specific problem

6. Production-ready analysis (500-600 words):
   - Compare all models: Which architecture performed best? Why?
   - Analyze the precision-recall tradeoff: What threshold would you recommend?
   - Discuss the impact of activation functions on fraud detection performance
   - Address the class imbalance: How did your approach handle it?
   - Consider deployment: Which model would you deploy? Consider accuracy, speed, and interpretability
   - Provide actionable recommendations for a real fraud detection system

Expected Deliverables:
- Complete Python code with data preprocessing, model building, training, and evaluation
- Visualizations: ROC curves, PR curves, confusion matrices, training curves (8-10 plots total)
- Comprehensive comparison table showing all metrics for all models
- Activation function ablation study results
- Production-ready analysis report (500-600 words)
- Final model selection and deployment recommendation
