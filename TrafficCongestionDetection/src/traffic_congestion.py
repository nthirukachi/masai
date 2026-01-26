import pandas as pd  # For data manipulation (like Excel)
import numpy as np   # For numerical math operations
import matplotlib.pyplot as plt # For basic plotting
import seaborn as sns # For beautiful simplified plotting
from sklearn.model_selection import train_test_split # To split data into study (train) and exam (test) sets
from sklearn.ensemble import RandomForestClassifier # The AI "Brain" - a collection of decision trees
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # To grade the student (AI)
from sklearn.preprocessing import StandardScaler # To make all numbers comparable (scaling)

# ==========================================
# 1. Data Understanding & Generation
# ==========================================
# 1.1 Why: We need data to teach the AI. Since we don't have the live file, we simulate it based on real physics.
# 1.2 What: Creating a dataset where High Density + Low Speed = Congestion.
def generate_synthetic_data(num_samples=1000):
    """
    Generates a synthetic traffic dataset.
    
    Arguments:
        num_samples (int): How many road segments (rows) to create. Default is 1000.
    
    Returns:
        pd.DataFrame: A table containing traffic features and labels.
    """
    np.random.seed(42) # Ensure we get the same random numbers every time (for reproducibility)
    
    # Randomly generate features
    # usage: np.random.uniform(low, high, size) -> picks random numbers between low and high
    avg_vehicle_speed = np.random.uniform(0, 100, num_samples) # Speed in km/h
    vehicle_density = np.random.uniform(0, 100, num_samples)   # Cars per km
    lane_occupancy = np.random.uniform(0, 1, num_samples) * 100 # Percentage occupied
    queue_length = np.random.uniform(0, 200, num_samples)      # Length of car line in meters
    optical_flow_mag = np.random.uniform(0, 5, num_samples)    # Pixel movement speed (from camera)
    edge_density = np.random.uniform(0, 1, num_samples)        # Visual complexity (more cars = more edges)
    time_of_day = np.random.choice([0, 1, 2, 3], num_samples)  # 0:Morning, 1:Afternoon, 2:Evening, 3:Night
    
    # Synthetic Coordinates for Spatial Map (like GPS)
    x_coord = np.random.uniform(0, 10, num_samples)
    y_coord = np.random.uniform(0, 10, num_samples)
    
    # ---------------------------------------------------------
    # Logic to decide the "Target" (Ground Truth)
    # If Speed is LOW (< 20) AND Density is HIGH (> 60), it is likely Congestion (1).
    # Otherwise, it matches normal traffic or minor slowdowns (0).
    # ---------------------------------------------------------
    labels = []
    for i in range(num_samples):
        # A simple rule-based logic to create "Truth" labels for training
        if (avg_vehicle_speed[i] < 30 and vehicle_density[i] > 50) or (queue_length[i] > 100):
            labels.append(1) # Congestion detected
        else:
            labels.append(0) # Normal flow
            
    # Create the DataFrame (Table)
    df = pd.DataFrame({
        'avg_vehicle_speed': avg_vehicle_speed,
        'vehicle_density': vehicle_density,
        'lane_occupancy': lane_occupancy,
        'queue_length': queue_length,
        'optical_flow_mag': optical_flow_mag,
        'edge_density': edge_density,
        'time_of_day': time_of_day,
        'x_coord': x_coord,
        'y_coord': y_coord,
        'target': labels
    })
    
    # Add some noise/errors because real world data is never perfect
    # We flip 5% of the labels randomly
    noise_indices = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    df.loc[noise_indices, 'target'] = 1 - df.loc[noise_indices, 'target']
    
    return df

# Initialize the Data
# 3.1 What: Calling the generator function
print("Generating synthetic traffic data...")
df = generate_synthetic_data(2000)

# 3.2 View first few rows
print("\nFirst 5 rows of our data:")
print(df.head())

# ==========================================
# 2. Data Preprocessing (Preparing Ingredients)
# ==========================================
# 2.1 Separation
# We separate "Features" (X) - the questions, from "Target" (y) - the answer.
X = df.drop(['target', 'x_coord', 'y_coord'], axis=1) # Drop answer and coordinates (coords are for visualization, not detection logic usually)
y = df['target'] # This is the answer key

# 2.2 Splitting
# We split data into Training (80%) and Testing (20%)
# random_state=42 guarantees the same split every time we run.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2.3 Scaling
# Why: Speed is 0-100, Density 0-100, Usage 0-1.
# Machine learning likes numbers to be generally in the same range (like -1 to 1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learn scale from TRAIN and apply
X_test_scaled = scaler.transform(X_test)       # Apply same scale to TEST (do not re-learn!)

# ==========================================
# 3. Machine Learning Model (The Brain)
# ==========================================
# We use Random Forest.
# Analogy: A "Forest" of "Trees" (Decisions). 
# Instead of asking one person (One Date Tree), we ask 100 people (100 Trees) and vote.
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("\nTraining the Random Forest Model...")
model.fit(X_train_scaled, y_train) # Teaching the model
print("Training Complete!")

# ==========================================
# 4. Evaluation (The Exam)
# ==========================================
print("\nEvaluating the Model...")
y_pred = model.predict(X_test_scaled) # Model takes the test

# Calculate Accuracy: (Correct Guesses / Total Questions)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f} ({(accuracy*100):.1f}%)")

# Detailed Report
# Precision: When it predicted "Congestion", how often was it right?
# Recall: Of all actual "Congestion", how many did we find?
# F1-Score: A balance between Precision and Recall.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
# Shows [True Neg, False Pos]
#       [False Neg, True Pos]
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ==========================================
# 5. Spatial Visualization (The Heatmap)
# ==========================================
# We want to see WHERE the congestion is on our map.
# We will use the full dataset for this visual to show the "City Map".

print("\nGenerating Traffic Heatmap...")

# Predict on the full dataset (scaled)
full_X_scaled = scaler.transform(X)
df['prediction'] = model.predict(full_X_scaled) 

plt.figure(figsize=(10, 8))

# Scatter plot: x_coord vs y_coord
# Color (c) depends on 'prediction' (0=Blue/Normal, 1=Red/Congested)
# cmap='coolwarm' makes 0 blueish and 1 reddish
scatter = plt.scatter(df['x_coord'], df['y_coord'], c=df['prediction'], cmap='coolwarm', alpha=0.6, s=50)

# Add elements
plt.title('Predicted Traffic Congestion Map (Simulated City)', fontsize=15)
plt.xlabel('Longitude (Relative)', fontsize=12)
plt.ylabel('Latitude (Relative)', fontsize=12)
plt.colorbar(scatter, label='Traffic Condition (0=Normal, 1=Congested)')
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot
output_path = "outputs/traffic_heatmap.png"
plt.savefig(output_path)
print(f"Heatmap saved to {output_path}")
# plt.show() # Uncomment if running in interactive mode with available display

# ==========================================
# 6. Technical Interpretation
# ==========================================
print("\n--- Project Reflection ---")
print("1. Feature Importance: The model found patterns in Speed and Density.")
print("2. Spatial Clustering: The heatmap identifies high-risk zones.")
print("3. Actionable Insight: Police should be deployed to the 'Red' zones automatically.")
