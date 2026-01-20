---
description: Set up the mandatory folder structure (Section 10).
---

# Create Project Structure Workflow

This workflow sets up the **Mandatory Folder Structure** (Rule Section 10).

## Execution Steps

### 1. Root Directory
Create `<project_name>/`.

### 2. Sub-directories
Create the following EXACT folder tree:

```
📁 <project_name>/
│
├── 📁 notebook/           # Stores <project_name>.ipynb
│
├── 📁 documentation/      # Stores the 4 Markdown files
│   ├── problem_statement.md
│   ├── concepts_explained.md
│   ├── observations_and_conclusion.md
│   └── interview_preparation.md
│
├── 📁 slides/             # Stores Slides
│   ├── notebooklm_style_slides.md
│   └── notebooklm_style_slides.pdf
│
├── 📁 src/                # Stores source code
│   └── original_script.py
│
├── 📁 outputs/            # Stores execution results
│   ├── execution_output.md
│   └── sample_outputs/
│
└── README.md              # Project navigation
```

### 3. README Generation
Generate a `README.md` at the root that links to all these locations and explains how to run the project.
