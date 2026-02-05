
import nbformat as nbf
import re
import os

# Define paths
PROJECT_DIR = r"C:\masai\MQ5_KMeans_Perceptron_CrossVal"
SRC_FILE = os.path.join(PROJECT_DIR, "src", "MQ5_KMeans_Perceptron_CrossVal.py")
NOTEBOOK_FILE = os.path.join(PROJECT_DIR, "notebook", "MQ5_KMeans_Perceptron_CrossVal.ipynb")

def create_notebook():
    # Read source file
    with open(SRC_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    nb = nbf.v4.new_notebook()
    cells = []

    # 1. Add Top-Level Problem Statement (Section 4 Rule)
    # in a real scenario, we'd extract this from the top docstring, 
    # but for safety/quality I'll add a structured header here based on the known content.
    header_md = """# 🧩 K-Means Feature Augmentation for Perceptron

## 🧩 Problem Statement
**Goal:** Determine if adding cluster-based features improves the performance of a simple Perceptron classifier on the Wine dataset.
**Why:** Feature engineering is key in ML. We want to see if unsupervised learning (clustering) can help supervised learning (classification).

## 🪜 Steps to Solve
1. **Load Data:** Wine dataset (178 samples, 13 features).
2. **Preprocessing:** Standardize features + Convert to binary classification.
3. **Feature Engineering:**
    - Train K-Means (k=4) on training data.
    - Create **One-Hot** features (which cluster?)
    - Create **Distance** features (how far from each centroid?)
4. **Model Training:**
    - Baseline: Perceptron on original 13 features.
    - Enhanced: Perceptron on 21 features (13 original + 8 new).
5. **Evaluation:** Stratified 5-Fold Cross-Validation.

## 🎯 Expected Output
- A comparison table of Accuracy, F1, and AP.
- A bar chart showing if the Enhanced model is better.
- A conclusion on whether the extra complexity is worth it.
"""
    cells.append(nbf.v4.new_markdown_cell(header_md))

    # 2. Split content into logical blocks
    # We look for "SECTION" markers and function definitions to split
    # For a high-quality teaching notebook, we'll parse line by line
    
    sections = re.split(r'(# =+)', content)
    
    current_md = []
    current_code = []
    
    lines = content.split('\n')
    
    for line in lines:
        stripped = line.strip()
        
        # If it's a comment block (starts with #)
        if stripped.startswith('#'):
            # If we have accumulated code, dump it first
            if current_code:
                # Add the previous markdown block
                if current_md:
                    cells.append(nbf.v4.new_markdown_cell('\n'.join(current_md)))
                    current_md = []
                
                # Add the code block
                cells.append(nbf.v4.new_code_cell('\n'.join(current_code)))
                current_code = []
            
            # Clean up the comment for markdown
            comment_text = line.replace('# ', '', 1).replace('#', '')
            current_md.append(comment_text)
            
        else:
            # It's code (or empty string that isn't a comment)
            if stripped == "" and not current_code:
                continue # Skip leading empty lines
            current_code.append(line)

    # Dump remaining buffers
    if current_md or current_code:
        if current_md:
            cells.append(nbf.v4.new_markdown_cell('\n'.join(current_md)))
        if current_code:
            cells.append(nbf.v4.new_code_cell('\n'.join(current_code)))

    # Set notebook cells
    nb['cells'] = cells
    
    # Save notebook
    os.makedirs(os.path.dirname(NOTEBOOK_FILE), exist_ok=True)
    with open(NOTEBOOK_FILE, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"✅ Notebook created successfully at: {NOTEBOOK_FILE}")

if __name__ == "__main__":
    create_notebook()
