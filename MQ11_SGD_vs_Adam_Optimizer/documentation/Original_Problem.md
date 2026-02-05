A regression model's loss plateaus, and you want to confirm whether switching optimizers improves convergence.

Dataset: sklearn.datasets.make_regression with n_samples=2000, n_features=40, noise=15. Split 70/15/15 and standardize features. Instructions:

    Build a three-layer MLP (hidden sizes 128 and 64 with ReLU) using PyTorch or Keras.
    Train Run A with SGD (lr=5e-3, momentum=0.9) and Run B with Adam (lr=1e-3) for 40 epochs, batch size 64.
    Track training loss, validation loss, and validation RMSE.

Deliverables:

    Training curves comparing both optimizers on the same axes.
    Table summarizing best validation loss/RMSE for each optimizer.
    Narrative (3-4 sentences) recommending which optimizer to keep and whether to adjust learning rates further.

Success Criteria:

    Evidence that both runs used identical initialization and data splits.
    Discussion covers convergence speed, stability, and final validation error.
    Code includes reproducible seeding and saves the best model per optimizer.

Solution Guidance: Expect Adam to converge faster initially, while tuned SGD may catch up later; justify your recommendation with the validation RMSE trends.
