---
description: Set up the mandatory folder structure (Section 10) with UV venv support.
---

# Create Project Structure Workflow

This workflow sets up the **Mandatory Folder Structure** (Rule Section 10).

## Execution Steps

### 1. Root Directory
Create `<project_name>/` under `c:\masai\`.

### 2. Sub-directories
Create the following EXACT folder tree:

```
📁 <project_name>/
│
├── 📁 notebook/           # Stores <project_name>.ipynb
│
├── 📁 documentation/      # Stores the 7 Markdown files
│   ├── Original_Problem.md           # [1] Raw user input (exact copy)
│   ├── problem_statement.md          # [2] What & Why (simplified)
│   ├── concepts_explained.md         # [3] Core Theory (12 points)
│   ├── observations_and_conclusion.md # [4] Results & Insights
│   ├── interview_questions.md        # [5] 10-20 Interview Q&A
│   ├── exam_preparation.md           # [6] MCQ/MSQ/Numerical
│   └── interview_preparation.md      # [7] Quick Revision Sheet
│
├── 📁 slides/             # Stores Slides
│   ├── slides.md          # Markdown slides
│   └── slides.pdf         # PDF slides
│
├── 📁 src/                # Stores source code
│   └── <project_name>.py  # Main implementation
│
├── 📁 outputs/            # Stores execution results
│   ├── execution_output.md
│   └── sample_outputs/
│
└── README.md              # Project navigation
```

### 3. UV Virtual Environment Setup
This project uses UV for Python environment management.

```powershell
# Navigate to project directory
cd c:\masai\<project_name>

# Initialize UV in project (if not exists)
uv init

# Install dependencies (example)
uv add numpy pandas matplotlib scikit-learn

# Run Python scripts
uv run python src/<project_name>.py
```

### 4. README Generation
Generate a `README.md` at the root that includes:
- Project overview
- Folder structure explanation
- How to run with UV
- Links to all 7 documentation files
