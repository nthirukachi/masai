---
description: Convert Python script to teaching-oriented Jupyter Notebook with STRICT 2.1-2.7 and 3.1-3.7 rules.
---

# Convert Python Script to Teaching Jupyter Notebook

This workflow converts a `.py` file into a beginner-friendly, teaching-oriented Jupyter Notebook.
**CRITICAL:** You must follow "User Rules Section 3, 4, 5, 6, 7" without deviation.

## Prerequisites
- Source Python file path.
- Target output directory (must be initialized via `/create-project-structure`).

## Execution Steps

### 1. 📥 Input Analysis
Identify:
- **Imports**: Every library used needs a concept explanation later.
- **Logical Blocks**: Split code not just by function, but by logical step (e.g., "Data Loading", "Preprocessing", "Model Initialization").

### 2. 📝 Top-Level Markdown (MANDATORY START)
**Rule:** Section 4 of User Rules.
The FIRST cell in the notebook MUST be Markdown containing:
```markdown
### 🧩 Problem Statement
- **Goal**: [What are we solving?]
- **relevance**: [Why does it matter in real life?]

### 🪜 Steps to Solve the Problem
1. [Step 1]
2. [Step 2]
...

### 🎯 Expected Output (OVERALL)
- [Description of final result]
- [Success criteria]
```

### 3. 🧱 Block-by-Block Conversion
For **EVERY** logical block of code (including imports):

#### A. The Explanation Cell (Markdown)
**Rule:** Section 5 (Per-Line Rules).
Before the code cell, create a Markdown cell. For **IMPORTANT LINES**, use this EXACT structure:

```markdown
### 🔹 Line Explanation: `[Code Snippet]`

#### 2.1 What the line does
[Simple description]

#### 2.2 Why it is used
[Reasoning]
**Alternatives:**
| Approach | Pros | Cons |
|----------|------|------|
| This Way | ... | ... |
| Other Way| ... | ... |

#### 2.3 When to use it
[Scenario/Condition]

#### 2.4 Where to use it
[Real-world application]

#### 2.5 How to use it
**Syntax:** `...`
**Example:** `...`

#### 2.6 How it works internally
[Step-by-step internal process / Flowchart]

#### 2.7 Output with sample examples
**Input:** `...`
**Output:** `...`
```

#### B. The Code Cell
Insert the actual Python code.

#### C. Function Argument Breakdown (If applicable)
**Rule:** Section 6 (Parameter Rules).
If the code calls a function (e.g., `train_test_split`, `model.fit`), add a specific Markdown section **IMMEDIATELY AFTER** or inside the explanation cell for **EVERY** argument:

```markdown
### ⚙️ Arguments for `[Function Name]`

#### Argument: `[Param Name]`
- **3.1 What it does**: ...
- **3.2 Why it is used**: ...
- **3.3 When to use it**: ...
- **3.4 Where to use it**: ...
- **3.5 How to use it**: ...
- **3.6 Internal Execution**: ...
- **3.7 Output Impact**: [Default vs Custom value examples]
```

### 4. 🎨 Teaching Explanations (Section 2)
For every explanation, ensure you answer:
- **WHY?** (Context)
- **WHAT?** (Definition)
- **WHEN?** (Scenario)
- **WHERE?** (Industry)
- **HOW?** (Syntax/Internals)

### 5. 📎 Final Polish
- Ensure **NO code line** exists without a preceding explanation.
- Verify **all metadata** (imports, variable names) are linked to correct concepts.
- **Save** as `.ipynb` in the `notebook/` folder.
