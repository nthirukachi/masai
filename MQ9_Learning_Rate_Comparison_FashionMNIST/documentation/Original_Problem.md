@[/full-teaching-project] 
Your baseline Fashion-MNIST classifier converges slowly and sometimes overshoots, so you want to compare two learning rates to find a stable training speed.

Dataset: torchvision.datasets.FashionMNIST or tf.keras.datasets.fashion_mnist (train split of 60,000 images, pixel values scaled to [0, 1]). Instructions:

    Use the same MLP architecture (two hidden layers with ReLU) in both runs.
    Train Run A with Adam (lr=1e-3) and Run B with Adam (lr=5e-4) for 15 epochs, batch size 128.
    Log training and validation loss/accuracy at each epoch and keep the best validation checkpoint for each run.

Deliverables:

    Code notebook or script that trains both runs reproducibly (set the random seed).
    A line plot comparing training and validation loss for the two learning rates.
    A short paragraph explaining which learning rate you would ship and why.

Success Criteria:

    Validation accuracy and final loss values are reported for both runs.
    Explanation references evidence from the curves (e.g., stability, generalization gap).
    Code uses the same preprocessing and architecture so the comparison isolates the learning rate.

Solution Guidance: Expect the lower learning rate to produce smoother curves and comparable or slightly better validation accuracy; highlight the speed vs. stability trade-off.
