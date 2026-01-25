"""
=============================================================================
AI-Based Forest Fire & Smoke Detection Using Aerial Imagery
=============================================================================

PURPOSE: Detect forest fire and smoke regions from aerial imagery features
         using machine learning for drone-based disaster monitoring.

WHAT THIS SCRIPT DOES:
    1. Loads and explores aerial imagery feature data
    2. Trains a Random Forest classifier to detect fire/smoke
    3. Evaluates model with precision, recall, F1, ROC-AUC
    4. Creates fire-risk heatmaps for spatial analysis
    5. Provides drone deployment recommendations

WHY WE NEED THIS:
    - Early fire detection saves lives, forests, and wildlife
    - Drones can monitor large areas quickly
    - AI helps prioritize where to send firefighters

REAL-WORLD ANALOGY:
    Like a doctor checking symptoms to diagnose illness,
    our AI checks image features to diagnose "fire" or "safe".
"""

# =============================================================================
# SECTION 1: IMPORT LIBRARIES
# =============================================================================
# WHAT: Import tools we need for data processing and machine learning
# WHY: Python needs external libraries for specialized tasks
# ANALOGY: Like getting tools from a toolbox before starting a project

import pandas as pd          # For data tables (like Excel in Python)
import numpy as np           # For mathematical operations on arrays
import matplotlib.pyplot as plt  # For creating graphs and charts
import seaborn as sns        # For beautiful statistical visualizations

# Machine Learning tools from scikit-learn
from sklearn.model_selection import train_test_split  # Split data for training/testing
from sklearn.ensemble import RandomForestClassifier   # Our ML model (decision trees)
from sklearn.metrics import (
    accuracy_score,           # How many predictions were correct overall
    precision_score,          # Of "fire" predictions, how many were right
    recall_score,             # Of actual fires, how many did we catch
    f1_score,                 # Balance of precision and recall
    confusion_matrix,         # Table of correct vs wrong predictions
    classification_report,    # Full performance summary
    roc_curve,                # For plotting ROC curve
    roc_auc_score             # Area Under ROC Curve score
)
from sklearn.preprocessing import StandardScaler  # Normalize feature values

import warnings
warnings.filterwarnings('ignore')  # Hide warning messages for cleaner output

# Set visual style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("🔥 AI-Based Forest Fire & Smoke Detection System")
print("=" * 60)


# =============================================================================
# SECTION 2: TASK 1 - DATA UNDERSTANDING
# =============================================================================
print("\n" + "=" * 60)
print("📁 TASK 1: DATA UNDERSTANDING")
print("=" * 60)

# -----------------------------------------------------------------------------
# 2.1 Load the Dataset
# -----------------------------------------------------------------------------
# WHAT: Read the CSV file containing aerial imagery features
# WHY: We need data to train our AI model
# HOW: pd.read_csv() reads CSV files into a DataFrame (table)

# Dataset URL from Google Sheets (exported as CSV)
DATA_URL = "https://docs.google.com/spreadsheets/d/1aszzbqsZ3G_LmH81EvRL06i5jDwJDl1SyDJXgxsbygM/export?format=csv"

print("\n📥 Loading dataset from Google Sheets...")
df = pd.read_csv(DATA_URL)
print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# -----------------------------------------------------------------------------
# 2.2 Display Basic Information
# -----------------------------------------------------------------------------
print("\n📊 Dataset Preview (First 5 Rows):")
print(df.head().to_string())

print("\n📋 Column Names and Data Types:")
print(df.dtypes.to_string())

print("\n📈 Statistical Summary:")
print(df.describe().round(3).to_string())

# -----------------------------------------------------------------------------
# 2.3 Check for Missing Values
# -----------------------------------------------------------------------------
# WHAT: Find if any data is missing
# WHY: Missing data can cause errors or bad predictions
# ANALOGY: Like checking if any questions on a test are blank

print("\n🔍 Missing Values Check:")
missing = df.isnull().sum()
print(f"   Total missing values: {missing.sum()}")
if missing.sum() == 0:
    print("   ✅ No missing values - dataset is complete!")

# -----------------------------------------------------------------------------
# 2.4 Class Distribution (Fire vs Safe)
# -----------------------------------------------------------------------------
# WHAT: Count how many fire vs safe tiles
# WHY: Imbalanced data can affect model training
# ANALOGY: Like checking if a class has equal boys and girls

print("\n🎯 Target Variable Distribution (fire_label):")
class_counts = df['fire_label'].value_counts()
print(f"   Safe (0): {class_counts[0]} tiles ({class_counts[0]/len(df)*100:.1f}%)")
print(f"   Fire (1): {class_counts[1]} tiles ({class_counts[1]/len(df)*100:.1f}%)")

# -----------------------------------------------------------------------------
# 2.5 Feature Relevance to Fire Detection
# -----------------------------------------------------------------------------
print("\n🔥 Feature Relevance to Fire/Smoke Detection:")
print("-" * 60)

feature_info = {
    'mean_red': 'Fire appears RED/ORANGE - high values indicate fire',
    'mean_green': 'Healthy vegetation is GREEN - low values may indicate burnt areas',
    'mean_blue': 'Reference channel - used in ratio calculations',
    'red_blue_ratio': 'HIGH ratio strongly indicates fire presence',
    'intensity_std': 'Fire flickers causing HIGH variation in brightness',
    'edge_density': 'Smoke BLURS edges - low density suggests smoke',
    'smoke_whiteness': 'Smoke appears WHITE/GRAY - high values indicate smoke',
    'haze_index': 'Smoke creates HAZE - high values suggest smoke',
    'hot_pixel_fraction': 'Fire creates HOT SPOTS - high fraction indicates fire',
    'local_contrast': 'Fire creates HIGH contrast against background'
}

for feature, description in feature_info.items():
    print(f"   • {feature}: {description}")


# =============================================================================
# SECTION 3: DATA VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("📊 DATA VISUALIZATION")
print("=" * 60)

# -----------------------------------------------------------------------------
# 3.1 Feature Distributions by Class
# -----------------------------------------------------------------------------
print("\n📈 Creating feature distribution plots...")

features = [col for col in df.columns if col != 'fire_label']

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for idx, feature in enumerate(features):
    ax = axes[idx]
    # Plot distribution for Safe (0) and Fire (1)
    df[df['fire_label'] == 0][feature].hist(ax=ax, alpha=0.5, label='Safe', bins=20, color='green')
    df[df['fire_label'] == 1][feature].hist(ax=ax, alpha=0.5, label='Fire', bins=20, color='red')
    ax.set_title(feature, fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions: Safe vs Fire', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sample_outputs/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/sample_outputs/feature_distributions.png")

# -----------------------------------------------------------------------------
# 3.2 Correlation Heatmap
# -----------------------------------------------------------------------------
print("\n📊 Creating correlation heatmap...")

plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sample_outputs/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/sample_outputs/correlation_heatmap.png")

# -----------------------------------------------------------------------------
# 3.3 Feature Importance Preview (Correlation with Target)
# -----------------------------------------------------------------------------
print("\n🎯 Feature Correlation with Fire Label:")
correlations = df.corr()['fire_label'].drop('fire_label').sort_values(ascending=False)
for feature, corr in correlations.items():
    indicator = "🔥" if corr > 0.2 else "💨" if corr > 0 else "🌲"
    print(f"   {indicator} {feature}: {corr:.3f}")


# =============================================================================
# SECTION 4: TASK 2 - MACHINE LEARNING MODEL
# =============================================================================
print("\n" + "=" * 60)
print("🤖 TASK 2: MACHINE LEARNING MODEL")
print("=" * 60)

# -----------------------------------------------------------------------------
# 4.1 Prepare Features and Target
# -----------------------------------------------------------------------------
# WHAT: Separate input features (X) from target label (y)
# WHY: ML needs separate inputs and expected outputs
# ANALOGY: X = exam questions, y = answer key

print("\n✂️ Preparing features and target...")
X = df.drop('fire_label', axis=1)  # All columns except target
y = df['fire_label']               # Target column

print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")

# -----------------------------------------------------------------------------
# 4.2 Split Data into Training and Testing Sets
# -----------------------------------------------------------------------------
# WHAT: Divide data into training (80%) and testing (20%)
# WHY: Test on unseen data to check if learning worked
# ANALOGY: Study from textbook, then take a surprise test

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                    # Data to split
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    stratify=y               # Keep class proportions equal
)

print(f"\n📚 Data Split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set: {X_test.shape[0]} samples")

# -----------------------------------------------------------------------------
# 4.3 Feature Scaling
# -----------------------------------------------------------------------------
# WHAT: Normalize features to same scale (mean=0, std=1)
# WHY: Some ML algorithms work better with scaled features
# ANALOGY: Converting different currencies to a common currency

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("   ✅ Features scaled using StandardScaler")

# -----------------------------------------------------------------------------
# 4.4 Train Random Forest Classifier
# -----------------------------------------------------------------------------
# WHAT: Train a model using many decision trees
# WHY: Random Forest is robust and works well for classification
# ANALOGY: Like asking 100 experts and taking majority vote

print("\n🌲 Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
    max_depth=10,            # Maximum depth of each tree
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=2,      # Minimum samples in a leaf
    random_state=42,         # For reproducibility
    n_jobs=-1                # Use all CPU cores
)

rf_model.fit(X_train_scaled, y_train)
print("   ✅ Model training complete!")

# -----------------------------------------------------------------------------
# 4.5 Make Predictions
# -----------------------------------------------------------------------------
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability of fire

print(f"   ✅ Predictions made on {len(y_pred)} test samples")

# -----------------------------------------------------------------------------
# 4.6 Model Evaluation
# -----------------------------------------------------------------------------
print("\n📈 MODEL EVALUATION RESULTS:")
print("-" * 60)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n   🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"   🔍 Precision: {precision:.4f} (Of fire predictions, {precision*100:.1f}% correct)")
print(f"   🚨 Recall:    {recall:.4f} (Caught {recall*100:.1f}% of actual fires)")
print(f"   ⚖️ F1-Score:  {f1:.4f} (Balance of precision and recall)")
print(f"   📊 ROC-AUC:   {roc_auc:.4f} (Ranking ability, 1.0 = perfect)")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Fire (1)']))

# -----------------------------------------------------------------------------
# 4.7 Confusion Matrix
# -----------------------------------------------------------------------------
print("\n🎯 Creating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Safe', 'Predicted Fire'],
            yticklabels=['Actual Safe', 'Actual Fire'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('outputs/sample_outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/sample_outputs/confusion_matrix.png")

# Explain confusion matrix
print("\n   Confusion Matrix Interpretation:")
print(f"   • True Negatives (Safe predicted Safe): {cm[0,0]}")
print(f"   • False Positives (Safe predicted Fire): {cm[0,1]} ⚠️ False alarms")
print(f"   • False Negatives (Fire predicted Safe): {cm[1,0]} ❌ MISSED FIRES!")
print(f"   • True Positives (Fire predicted Fire): {cm[1,1]} ✅ Correctly caught")

# -----------------------------------------------------------------------------
# 4.8 ROC Curve
# -----------------------------------------------------------------------------
print("\n📈 Creating ROC curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Guess (AUC = 0.5)')
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall/Sensitivity)')
plt.title('ROC Curve - Fire Detection Performance', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/sample_outputs/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/sample_outputs/roc_curve.png")

# -----------------------------------------------------------------------------
# 4.9 Feature Importance
# -----------------------------------------------------------------------------
print("\n🏆 Feature Importance (Which features help most):")

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in feature_importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"   {row['Feature']:20s} {bar} {row['Importance']:.3f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='Reds_r')
plt.title('Feature Importance for Fire Detection', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/sample_outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   ✅ Saved: outputs/sample_outputs/feature_importance.png")


# =============================================================================
# SECTION 5: TASK 3 - SPATIAL RISK ANALYSIS & VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("🗺️ TASK 3: SPATIAL RISK ANALYSIS & VISUALIZATION")
print("=" * 60)

# -----------------------------------------------------------------------------
# 5.1 Calculate Fire Risk Probabilities for All Tiles
# -----------------------------------------------------------------------------
print("\n🔥 Calculating fire risk probabilities for all tiles...")

# Get predictions for entire dataset
X_all_scaled = scaler.transform(X)
all_predictions = rf_model.predict(X_all_scaled)
all_probabilities = rf_model.predict_proba(X_all_scaled)[:, 1]

df['fire_risk_probability'] = all_probabilities
df['predicted_label'] = all_predictions

print(f"   ✅ Risk probabilities calculated for {len(df)} tiles")

# -----------------------------------------------------------------------------
# 5.2 Risk Level Classification
# -----------------------------------------------------------------------------
# WHAT: Categorize risk into levels (Low, Medium, High, Critical)
# WHY: Easier to prioritize response

def classify_risk(prob):
    if prob < 0.25:
        return 'Low'
    elif prob < 0.50:
        return 'Medium'
    elif prob < 0.75:
        return 'High'
    else:
        return 'Critical'

df['risk_level'] = df['fire_risk_probability'].apply(classify_risk)

print("\n📊 Risk Level Distribution:")
risk_counts = df['risk_level'].value_counts()
for level in ['Low', 'Medium', 'High', 'Critical']:
    if level in risk_counts:
        count = risk_counts[level]
        pct = count / len(df) * 100
        emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Critical': '🔴'}[level]
        print(f"   {emoji} {level}: {count} tiles ({pct:.1f}%)")

# -----------------------------------------------------------------------------
# 5.3 Create Fire Risk Heatmap
# -----------------------------------------------------------------------------
print("\n🗺️ Creating fire risk heatmap...")

# Create a grid-based visualization (simulating spatial tiles)
n_tiles = len(df)
grid_size = int(np.ceil(np.sqrt(n_tiles)))

# Reshape probabilities into a grid
risk_grid = np.zeros((grid_size, grid_size))
for i, prob in enumerate(all_probabilities):
    row = i // grid_size
    col = i % grid_size
    if row < grid_size and col < grid_size:
        risk_grid[row, col] = prob

plt.figure(figsize=(12, 10))
im = plt.imshow(risk_grid, cmap='YlOrRd', interpolation='nearest', 
                vmin=0, vmax=1, aspect='equal')
plt.colorbar(im, label='Fire Risk Probability', shrink=0.8)
plt.title('🔥 Fire Risk Heatmap\n(Aerial Tile Analysis)', fontsize=14, fontweight='bold')
plt.xlabel('Tile Column (East →)')
plt.ylabel('Tile Row (North →)')

# Add grid lines
plt.grid(True, color='white', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/sample_outputs/fire_risk_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/sample_outputs/fire_risk_heatmap.png")

# -----------------------------------------------------------------------------
# 5.4 Risk Distribution Plot
# -----------------------------------------------------------------------------
print("\n📊 Creating risk distribution plot...")

plt.figure(figsize=(10, 6))
plt.hist(all_probabilities, bins=30, edgecolor='black', alpha=0.7, color='orangered')
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
plt.axvline(x=0.75, color='darkred', linestyle=':', linewidth=2, label='Critical Threshold (0.75)')
plt.xlabel('Fire Risk Probability')
plt.ylabel('Number of Tiles')
plt.title('Distribution of Fire Risk Probabilities', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/sample_outputs/risk_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/sample_outputs/risk_distribution.png")


# =============================================================================
# SECTION 6: TASK 4 - DRONE & DISASTER RESPONSE INTERPRETATION
# =============================================================================
print("\n" + "=" * 60)
print("🚁 TASK 4: DRONE & DISASTER RESPONSE INTERPRETATION")
print("=" * 60)

# -----------------------------------------------------------------------------
# 6.1 Identify High-Priority Tiles
# -----------------------------------------------------------------------------
print("\n🚨 HIGH-PRIORITY TILES FOR DRONE DEPLOYMENT:")
print("-" * 60)

high_risk_tiles = df[df['fire_risk_probability'] >= 0.75].copy()
high_risk_tiles = high_risk_tiles.sort_values('fire_risk_probability', ascending=False)

print(f"\n   🔴 Critical Risk Tiles (≥75% probability): {len(high_risk_tiles)}")

if len(high_risk_tiles) > 0:
    print("\n   Top 10 Priority Tiles for Immediate Inspection:")
    print("   " + "-" * 50)
    for i, (idx, row) in enumerate(high_risk_tiles.head(10).iterrows()):
        print(f"   #{i+1} Tile {idx}: Risk = {row['fire_risk_probability']*100:.1f}%")
        print(f"       Hot Pixels: {row['hot_pixel_fraction']:.3f}, Red/Blue: {row['red_blue_ratio']:.2f}")

# -----------------------------------------------------------------------------
# 6.2 Drone Deployment Strategy
# -----------------------------------------------------------------------------
print("\n\n🚁 DRONE DEPLOYMENT STRATEGY:")
print("-" * 60)

critical_count = len(df[df['risk_level'] == 'Critical'])
high_count = len(df[df['risk_level'] == 'High'])
medium_count = len(df[df['risk_level'] == 'Medium'])

print("\n   📋 RECOMMENDED ACTIONS:")
print()
print(f"   🔴 PHASE 1 - IMMEDIATE (Critical Risk: {critical_count} tiles)")
print("      • Deploy all available drones to critical areas")
print("      • Alert ground firefighting teams")
print("      • Notify emergency services")
print()
print(f"   🟠 PHASE 2 - URGENT (High Risk: {high_count} tiles)")
print("      • Schedule drone patrol within 30 minutes")
print("      • Position firefighting resources nearby")
print("      • Monitor for fire spread")
print()
print(f"   🟡 PHASE 3 - MONITORING (Medium Risk: {medium_count} tiles)")
print("      • Regular patrol every 2 hours")
print("      • Set up automated camera monitoring")
print("      • Update risk assessment hourly")

# -----------------------------------------------------------------------------
# 6.3 Resource Allocation Recommendations
# -----------------------------------------------------------------------------
print("\n\n📊 RESOURCE ALLOCATION RECOMMENDATIONS:")
print("-" * 60)

total_tiles = len(df)
print(f"""
   Based on {total_tiles} analyzed tiles:
   
   🚁 Drones Required:
      • Minimum: {max(1, critical_count // 10)} drones for critical areas
      • Recommended: {max(2, (critical_count + high_count) // 10)} drones for full coverage
   
   👨‍🚒 Firefighter Teams:
      • Standby: {max(1, critical_count // 5)} teams near critical zones
      • On-call: {max(1, high_count // 10)} teams for high-risk areas
   
   🚒 Equipment Positioning:
      • Fire trucks should be positioned near tile clusters
      • Water sources should be identified near high-risk zones
""")


# =============================================================================
# SECTION 7: TASK 5 - REFLECTION
# =============================================================================
print("\n" + "=" * 60)
print("📝 TASK 5: REFLECTION")
print("=" * 60)

print("""
🔍 DATASET LIMITATIONS:

1. ⚠️ TEMPORAL INFORMATION MISSING
   • No timestamp data - can't track fire progression
   • Real fires spread over time
   • Improvement: Add time-series features

2. ⚠️ SPATIAL CONTEXT LIMITED
   • Tiles analyzed independently
   • Fire in one tile affects neighbors
   • Improvement: Include neighboring tile features

3. ⚠️ WEATHER DATA ABSENT
   • Wind speed affects fire spread
   • Humidity affects fire risk
   • Improvement: Integrate weather API data

4. ⚠️ SAMPLE SIZE
   • ~1000 tiles may not represent all conditions
   • Seasonal variations not captured
   • Improvement: Collect year-round data

5. ⚠️ GROUND TRUTH VERIFICATION
   • Labels may have annotation errors
   • No severity levels (small fire vs large fire)
   • Improvement: Multi-class labeling


💡 POTENTIAL IMPROVEMENTS:

1. 🧠 ADVANCED MODELS
   • Try XGBoost, LightGBM for better performance
   • Use deep learning (CNN) on raw images
   • Ensemble multiple models

2. 🔄 REAL-TIME PROCESSING
   • Stream processing for live drone feeds
   • Edge computing on drones
   • Faster alerts

3. 🗺️ BETTER VISUALIZATION
   • 3D terrain maps with risk overlay
   • Integration with GIS systems
   • Mobile app for field teams

4. 🤖 AUTOMATION
   • Automatic drone dispatch
   • AI-controlled patrol routes
   • Predictive maintenance


📊 MODEL PERFORMANCE SUMMARY:

   Metric          Value    Target   Status
   ─────────────────────────────────────────
   Accuracy        {:.1f}%    >85%     {}
   Precision       {:.1f}%    >75%     {}
   Recall          {:.1f}%    >80%     {}
   ROC-AUC         {:.3f}     >0.85    {}

""".format(
    accuracy*100, "✅" if accuracy > 0.85 else "⚠️",
    precision*100, "✅" if precision > 0.75 else "⚠️",
    recall*100, "✅" if recall > 0.80 else "⚠️",
    roc_auc, "✅" if roc_auc > 0.85 else "⚠️"
))


# =============================================================================
# SECTION 8: SAVE RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("💾 SAVING RESULTS")
print("=" * 60)

# Save predictions to CSV
results_df = df[['fire_label', 'predicted_label', 'fire_risk_probability', 'risk_level']]
results_df.to_csv('outputs/sample_outputs/predictions.csv', index=False)
print("\n   ✅ Saved: outputs/sample_outputs/predictions.csv")

# Save model metrics
metrics_summary = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Value': [accuracy, precision, recall, f1, roc_auc]
}
pd.DataFrame(metrics_summary).to_csv('outputs/sample_outputs/model_metrics.csv', index=False)
print("   ✅ Saved: outputs/sample_outputs/model_metrics.csv")

print("\n" + "=" * 60)
print("🎉 FOREST FIRE DETECTION ANALYSIS COMPLETE!")
print("=" * 60)
print("""
   📁 Generated Files:
      • outputs/sample_outputs/feature_distributions.png
      • outputs/sample_outputs/correlation_heatmap.png
      • outputs/sample_outputs/confusion_matrix.png
      • outputs/sample_outputs/roc_curve.png
      • outputs/sample_outputs/feature_importance.png
      • outputs/sample_outputs/fire_risk_heatmap.png
      • outputs/sample_outputs/risk_distribution.png
      • outputs/sample_outputs/predictions.csv
      • outputs/sample_outputs/model_metrics.csv
""")
