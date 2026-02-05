import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# 1. Load Dataset
# Try to load from local path, otherwise create sample data
try:
    data_path = '../data/Mall_Customers.csv' # Adjust path if needed
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded dataset from {data_path}")
    else:
        raise FileNotFoundError
except:
    print("Dataset not found. Creating sample Mall Customer data for demonstration.")
    # Creating a synthetic dataset that mimics the structure of Mall Customers
    np.random.seed(42)
    n_samples = 200
    ids = np.arange(1, n_samples + 1)
    genders = np.random.choice(['Male', 'Female'], n_samples)
    ages = np.random.randint(18, 70, n_samples)
    # create income and score with some cluster-like structure
    income = np.concatenate([
        np.random.normal(25, 5, 40), np.random.normal(55, 10, 80), np.random.normal(90, 10, 80)
    ]).astype(int)
    score = np.concatenate([
        np.random.normal(80, 10, 40), np.random.normal(50, 10, 80), np.random.normal(20, 10, 40), np.random.normal(85, 10, 40)
    ]).astype(int)
    
    df = pd.DataFrame({
        'CustomerID': ids,
        'Gender': genders,
        'Age': ages,
        'Annual Income (k$)': income,
        'Spending Score (1-100)': score
    })

# 2. Select Features & Preprocess
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Means Clustering (K=5)
# K=5 is standard for this dataset (Target, Careless, Sensible, Standard, Safe)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Inverse Transform Centroids
# We want to see the centers in ORIGINAL UNITS (e.g. $60k, Score 50)
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Create a DataFrame for the Cluster Profile
cluster_profile = pd.DataFrame(centroids_original, columns=features)
cluster_profile['Cluster_ID'] = range(5)
cluster_profile['Count'] = df['Cluster'].value_counts().sort_index().values
cluster_profile['Percent'] = (cluster_profile['Count'] / len(df)) * 100

# Reorder columns
cluster_profile = cluster_profile[['Cluster_ID', 'Count', 'Percent'] + features]

# Round for readability
cluster_profile = cluster_profile.round(2)

print("\n--- Cluster Profile (Original Scale) ---")
print(cluster_profile)
cluster_profile.to_csv('outputs/cluster_profile.csv', index=False)

# 5. PCA Visualization (2D Projection)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, alpha=0.8)

# Annotate Centroids (transformed first to PCA space)
centroids_pca = pca.transform(centroids_scaled)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', marker='X', label='Centroids')

# Heuristic Naming (Requires human look, but we can programmatically guess based on values)
# For the visual, we'll just annotate the Cluster ID
for i in range(5):
    plt.text(centroids_pca[i, 0], centroids_pca[i, 1]+0.2, f'Cluster {i}', 
             fontsize=12, fontweight='bold', color='black', ha='center')

plt.title('Customer Segments (PCA Projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('outputs/cluster_pca.png')
print("Saved PCA visualization to outputs/cluster_pca.png")

# 6. Stakeholder Briefing Hints
# This part generates summary stats to help write the briefing
print("\n--- Summary for Stakeholder Briefing ---")
for i in range(5):
    row = cluster_profile.iloc[i]
    print(f"Cluster {int(row['Cluster_ID'])}: {int(row['Count'])} customers ({row['Percent']}%)")
    print(f"  - Avg Income: ${row['Annual Income (k$)']}k")
    print(f"  - Avg Score:  {row['Spending Score (1-100)']}")
    
    # Interpretation logic (simple heuristics)
    inc = row['Annual Income (k$)']
    score = row['Spending Score (1-100)']
    label = "Unknown"
    if inc < 40 and score < 40: label = "Low Income, Low Spend (Sensible?)"
    elif inc < 40 and score > 60: label = "Low Income, High Spend (Careless?)"
    elif inc > 70 and score < 40: label = "High Income, Low Spend (Frugal/Target?)"
    elif inc > 70 and score > 60: label = "High Income, High Spend (Ideal Target)"
    else: label = "Average (Standard)"
    print(f"  - Interpretation: {label}\n")
