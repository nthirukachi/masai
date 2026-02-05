@[/full-teaching-project] 
You are preparing a training pipeline for a medical-risk classifier and need an automatic way to stop training before it overfits.

Dataset: sklearn.datasets.load_breast_cancer (standardize features with StandardScaler, split 70/15/15 stratified). Instructions:

    Implement a feed-forward network in Keras or PyTorch with two hidden layers of 64 and 32 units (ReLU).
    Train with Adam (lr=1e-3, weight_decay=1e-4 or kernel_regularizer=tf.keras.regularizers.l2(1e-4)).
    Add early stopping on validation loss with patience 4 and restore the best weights.
    Track validation AUC each epoch.

Deliverables:

    Training script/notebook showing early stopping activation (print the stopping epoch).
    Table or text summary of training vs. validation metrics at the time training stopped.
    Brief reflection (4-5 sentences) on how weight decay and early stopping affected the outcome.

Success Criteria:

    Validation AUC, accuracy, and stop epoch clearly documented.
    Reflection ties each control (weight decay, early stopping) to observed metrics.
    Code resets random seeds and uses the same scaler for train/validation/test.

Solution Guidance: Expect training to stop before 40 epochs with validation AUC >= 0.95; explain how weight decay kept weights small and how early stopping protected against late-epoch overfitting.
