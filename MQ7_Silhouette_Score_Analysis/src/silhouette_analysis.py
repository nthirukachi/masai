import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler
import os

# ----------------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ----------------------------------------------------------------------------------

# 🔹 Set up the output directory
# 2.1 What the line does: Defines the path where we will save our results (plots, tables).
# 2.2 Why it is used: Keeps our project organized. We don't want files scattered everywhere.
# 2.3 When to use it: At the start of any script that produces files.
# 2.4 Where to use it: Real-world projects, data pipelines.
# 2.5 How to use it: `os.makedirs('path/to/folder', exist_ok=True)`
# 2.6 How it works internally: Checks if the folder exists. If not, it asks the OS to create it.
# 2.7 Output: A folder named 'outputs' is created if it doesn't exist.
output_dir = r"c:\masai\MQ7_Silhouette_Score_Analysis\outputs"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------------------
# 2. DATA LOADING / GENERATION
# ----------------------------------------------------------------------------------

def load_or_create_data():
    """
    Creates a simulated Mall Customer dataset for teaching purposes.
    
    ⚙️ Function Arguments Explanation:
    None specifically here, but generally:
    - We simulate data to make sure the code runs for everyone, even without downloading files.
    """
    
    # 🔹 Set random seed for reproducibility
    # 2.1 What the line does: Ensures that random numbers are the same every time we run the code.
    # 2.2 Why it is used: To make our results consistent and debuggable.
    # 2.3 When to use it: Whenever using random processes (like generating data or K-Means initialization).
    # 2.4 Where to use it: Scientific research, tutorials, testing.
    # 2.5 How to use it: `np.random.seed(42)`
    # 2.6 How it works internally: Initializes the random number generator's state.
    # 2.7 Output: The sequence of random numbers will be identical on every run.
    np.random.seed(42)
    
    n_samples = 200
    
    # 🔹 Generate 'CustomerID'
    # 2.1 What the line does: Creates a list of IDs from 1 to 200.
    # 2.2 Why it is used: To uniquely identify each customer.
    # 2.3 When to use it: When datasets don't have a unique key.
    # 2.4 Where to use it: Databases, CRM systems.
    # 2.5 How to use it: `np.arange(1, n + 1)`
    # 2.6 How it works internally: Creates a numpy array [1, 2, ..., 200].
    # 2.7 Output: Array of integers.
    customer_ids = np.arange(1, n_samples + 1)
    
    # 🔹 Generate 'Age'
    # 2.1 What the line does: Generates random ages between 18 and 70.
    # 2.2 Why it is used: To simulate the age demographic of mall shoppers.
    # 2.3 When to use it: Simulating human populations.
    # 2.4 Where to use it: Mock data generation.
    # 2.5 How to use it: `np.random.randint(low, high, size)`
    # 2.6 How it works internally: Picks k random integers from the uniform distribution.
    # 2.7 Output: Array of 200 ages.
    age = np.random.randint(18, 70, n_samples)
    
    # 🔹 Generate 'Annual Income (k$)'
    # 2.1 What the line does: Generates income data with some clusters logic (low, medium, high).
    # 2.2 Why it is used: To create natural groups in the data for K-Means to find.
    # 2.3 When to use it: Creating synthetic datasets with structure.
    # 2.4 Where to use it: Testing clustering algorithms.
    # 2.5 How to use it: `np.concatenate([group1, group2])`
    # 2.6 How it works internally: Joins multiple arrays into one.
    # 2.7 Output: Array of incomes.
    income = np.concatenate([
        np.random.normal(30, 10, 60),  # Low income
        np.random.normal(70, 15, 80),  # Medium income
        np.random.normal(110, 10, 60)  # High income
    ])
    
    # 🔹 Generate 'Spending Score (1-100)'
    # 2.1 What the line does: Creates spending scores corresponding to income groups to form clusters.
    # 2.2 Why it is used: We want clusters like "Rich & Spender", "Rich & Saver", etc.
    # 2.3 When to use it: Simulating behavior patterns.
    # 2.4 Where to use it: Market segmentation simulation.
    # 2.5 How to use it: Concatenating arrays with different distributions.
    # 2.6 How it works internally: Generates numbers and then we clip them to 1-100 range.
    # 2.7 Output: Array of scores.
    spending = np.concatenate([
        np.random.normal(80, 10, 60), # Low income, High spend (students?)
        np.random.normal(50, 15, 80), # Medium income, Medium spend
        np.random.normal(20, 10, 60)  # High income, Low spend (savers)
    ])
    
    # 🔹 Create DataFrame
    # 2.1 What the line does: Combines our arrays into a structured table.
    # 2.2 Why it is used: Pandas DataFrames are the standard for data manipulation in Python.
    # 2.3 When to use it: Always when working with tabular data.
    # 2.4 Where to use it: Data science, analytics, finance.
    # 2.5 How to use it: `pd.DataFrame(dictionary)`
    # 2.6 How it works internally: Aligns data arrays by index and creates a structured object with metadata.
    # 2.7 Output: A table with columns CustomerID, Age, Annual Income, Spending Score.
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': age,
        'Annual Income (k$)': np.abs(income).astype(int),
        'Spending Score (1-100)': np.clip(spending, 1, 100).astype(int)
    })
    
    return df

print("LOADING DATA...")
df = load_or_create_data()
print("Data Loaded Successfully!")
print(df.head())

# ----------------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------------------------

# 🔹 Create 'Spending_to_Income_Ratio'
# 2.1 What the line does: Calculates how much a person spends relative to their income.
# 2.2 Why it is used: Ratios can often separate groups better than raw numbers (e.g., "Living beyond means").
# 2.3 When to use it: When the relationship between two variables matters more than the variables themselves.
# 2.4 Where to use it: Financial risk scoring, credit card fraud detection.
# 2.5 How to use it: `df['new_col'] = df['col_a'] / df['col_b']`
# 2.6 How it works internally: Vectorized division of two columns.
# 2.7 Output: A new column with float values.
df['Spending_to_Income_Ratio'] = df['Spending Score (1-100)'] / (df['Annual Income (k$)'] + 1) # +1 to avoid division by zero

# 🔹 Select Features for Clustering
# 2.1 What the line does: Picks the specific columns used for K-Means.
# 2.2 Why it is used: We shouldn't use ID (it's random) or irrelevant columns.
# 2.3 When to use it: Before feeding data into a model.
# 2.4 Where to use it: Everywhere in ML pipelines.
# 2.5 How to use it: `df[['col1', 'col2']]`
# 2.6 How it works internally: Selects a subset of the DataFrame.
# 2.7 Output: A smaller DataFrame with only 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'.
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# ----------------------------------------------------------------------------------
# 4. DATA SCALING
# ----------------------------------------------------------------------------------

# 🔹 Initialize MinMaxScaler
# 2.1 What the line does: Creates a scaler object that will squash numbers between 0 and 1.
# 2.2 Why it is used: K-Means uses distance (Euclidean). If Income is 100,000 and Age is 50, Income will dominate the distance. Scaling makes them equal importance.
# 2.3 When to use it: For algorithms based on distance (K-Means, KNN, SVM).
# 2.4 Where to use it: Almost every ML project involving numerical data.
# 2.5 How to use it: `scaler = MinMaxScaler()`
# 2.6 How it works internally: Prepares to calculate (x - min) / (max - min).
# 2.7 Output: An un-fitted scaler object.
scaler = MinMaxScaler()

# 🔹 Fit and Transform Data
# 2.1 What the line does: Learns the min/max specific to our data and then scales the data.
# 2.2 Why it is used: To actually convert the data values.
# 2.3 When to use it: During data preprocessing step.
# 2.4 Where to use it: Preprocessing pipelines.
# 2.5 How to use it: `scaler.fit_transform(X)`
# 2.6 How it works internally: Finds min/max for each column, then applies formula.
# 2.7 Output: A numpy array where all values are between 0 and 1.
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------------------
# 5. K-MEANS & SILHOUETTE ANALYSIS
# ----------------------------------------------------------------------------------

K_values = [2, 3, 4, 5]
silhouette_scores = []
inertia_values = []

# Loop through each K value
for k in K_values:
    # 🔹 Initialize K-Means
    # 2.1 What the line does: Creates a K-Means model with 'k' clusters.
    # 2.2 Why it is used: To tell the computer "find k groups".
    # 2.3 When to use it: Inside a loop for hyperparameter tuning (finding best K).
    # 2.4 Where to use it: Customer segmentation, image compression.
    # 2.5 How to use it: `KMeans(n_clusters=k, random_state=42)`
    # 2.6 How it works internally: Prepares the algorithm (Lloyd's algorithm).
    # 2.7 Output: A KMeans model object.
    
    # ⚙️ Function Arguments Explanation (KMeans):
    # - n_clusters: The number of groups we want.
    # - random_state: Makes the random initialization of centroids consistent.
    # - n_init: 'auto' means it will run multiple times (usually 10) to find best start.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    
    # 🔹 Train the Model
    # 2.1 What the line does: Runs the K-Means algorithm on our scaled data.
    # 2.2 Why it is used: To find the cluster centroids and assign labels.
    # 2.3 When to use it: After initialization.
    # 2.4 Where to use it: Model training step.
    # 2.5 How to use it: `model.fit(X)`
    # 2.6 How it works internally: Iteratively moves centroids to minimize distance to points.
    # 2.7 Output: The model learns 'cluster_centers_' and 'labels_'.
    kmeans.fit(X_scaled)
    
    # 🔹 Get Cluster Labels
    # 2.1 What the line does: Gets the group ID (0, 1, 2...) for each customer.
    # 2.2 Why it is used: To calculate silhouette score (needs to know who belongs where).
    # 2.3 When to use it: After fitting.
    # 2.4 Where to use it: Evaluating results.
    # 2.5 How to use it: `model.labels_` or `model.predict(X)`
    # 2.6 How it works internally: Returns the index of the closest centroid for each point.
    # 2.7 Output: Array of size (n_samples,).
    cluster_labels = kmeans.labels_
    
    # 🔹 Calculate Silhouette Score
    # 2.1 What the line does: Computes the average silhouette score for all samples.
    # 2.2 Why it is used: To measure how well the clusters are separated.
    # 2.3 When to use it: Evaluating cluster quality. High score (close to 1) = Good.
    # 2.4 Where to use it: Unsupervised learning evaluation.
    # 2.5 How to use it: `silhouette_score(X, labels)`
    # 2.6 How it works internally: A = mean distance to own cluster, B = mean distance to nearest neighbor cluster. Score = (B - A) / max(A, B).
    # 2.7 Output: A single float value between -1 and 1.
    
    # ⚙️ Function Arguments Explanation (silhouette_score):
    # - X: The data samples (scaled).
    # - labels: The predicted cluster labels.
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)
    
    # 🔹 Store Inertia
    # 2.1 What the line does: Gets the "Sum of Squared Errors" (Inertia).
    # 2.2 Why it is used: Measures how compact the clusters are (lower is better).
    # 2.3 When to use it: For the "Elbow Method".
    # 2.4 Where to use it: Comparing K-Means runs.
    # 2.5 How to use it: `model.inertia_`
    # 2.6 How it works internally: Sums the squared distances of samples to their closest cluster center.
    # 2.7 Output: A large float number.
    inertia_values.append(kmeans.inertia_)
    
    # ------------------------------------------------------------------------------
    # 6. SILHOUETTE PLOT FOR EACH K
    # ------------------------------------------------------------------------------
    
    # Setup the plot
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_scaled) + (k + 1) * 10])

    # 🔹 Calculate Silhouette Values for Each Sample
    # 2.1 What the line does: Computes the score for EVERY single customer, not just the average.
    # 2.2 Why it is used: To draw the detailed silhouette plot.
    # 2.3 When to use it: For visualization of cluster balance.
    # 2.4 Where to use it: Deep dive analysis of clusters.
    # 2.5 How to use it: `silhouette_samples(X, labels)`
    # 2.6 How it works internally: Same math as score, but returns array instead of mean.
    # 2.7 Output: Array of scores for each sample.
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

    y_lower = 10
    
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title(f"Silhouette Plot for K={k}")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=score, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    plt.savefig(os.path.join(output_dir, f'silhouette_plot_k{k}.png'))
    plt.close()

# ----------------------------------------------------------------------------------
# 7. COMPARISON & RESULTS
# ----------------------------------------------------------------------------------

# 🔹 Create Comparison Table
# 2.1 What the line does: Creates a summary table of K, Inertia, and Silhouette Score.
# 2.2 Why it is used: To easily compare the trade-offs.
# 2.3 When to use it: Making the final decision on K.
# 2.4 Where to use it: Reporting results to stakeholders.
# 2.5 How to use it: `pd.DataFrame(...)`
# 2.6 How it works internally: Organizes our lists into columns.
# 2.7 Output: A clean table.
comparison_df = pd.DataFrame({
    'K': K_values,
    'Inertia': inertia_values,
    'Silhouette Score': silhouette_scores
})

print("\n---------------- COMPARISON TABLE ----------------")
print(comparison_df)

# Save the table
comparison_df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)

# 🔹 Plot Comparison (Score vs Inertia)
# 2.1 What the line does: Plots both metrics on dual axes to compare them.
# 2.2 Why it is used: Inertia always goes down as K goes up, but Silhouette peaks.Visualizing both helps find the 'sweet spot'.
# 2.3 When to use it: Selecting model hyperparameters.
# 2.4 Where to use it: Final presentation.
# 2.5 How to use it: `ax1.plot()`, `ax1.twinx()`, `ax2.plot()`
# 2.6 How it works internally: Overlays two line charts with different Y-scales.
# 2.7 Output: A combined chart saved to file.
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (Lower is Better)', color=color)
ax1.plot(K_values, inertia_values, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Silhouette Score (Higher is Better)', color=color)
ax2.plot(K_values, silhouette_scores, color=color, marker='o', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Inertia vs Silhouette Score for Different K')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(os.path.join(output_dir, 'inertia_vs_silhouette.png'))
plt.close()

# ----------------------------------------------------------------------------------
# 8. INTERPRETATION
# ----------------------------------------------------------------------------------

best_k_idx = np.argmax(silhouette_scores)
best_k = K_values[best_k_idx]
best_score = silhouette_scores[best_k_idx]

interpretation = f"""
START INTERPRETATION
--------------------
Based on the analysis, the optimal number of clusters is likely **K={best_k}** with a Silhouette Score of **{best_score:.3f}**.

1. **Inertia vs. Silhouette:** 
   - Inertia decreases as we add more clusters (which is expected).
   - However, the Silhouette Score peaks at K={best_k}, indicating that at this point, the clusters are the most distinct and well-separated.

2. **Marketing Stakeholder Note:**
   - We have identified {best_k} distinct customer segments.
   - These groups likely represent behaviors such as "High Spenders," "Budget Conscious," etc.
   - We should tailor marketing strategies for each of these {best_k} groups specifically.

3. **Visual Confirmation:**
   - Please check 'silhouette_plot_k{best_k}.png'. You will see that the cluster "knives" (the shapes) are roughly equal in thickness and extend past the average line, which is a sign of good clustering.
"""

print(interpretation)

# Save interpretation
with open(os.path.join(output_dir, 'interpretation.txt'), 'w') as f:
    f.write(interpretation)

print("\n[SUCCESS] Execution Complete. All outputs saved to 'outputs/' directory.")
