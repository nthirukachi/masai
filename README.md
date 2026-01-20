# Agent Workflows & Project Rules

This repository contains the agent workflows designed to automate the creation of **Teaching-Oriented Data Science Projects**. These workflows strictly adhere to the "User Rules" (Sections 0-12) defining a specific pedagogical style.

## 🚀 Execution Order (The "Flow")

To create a new project from a Python script, you should follow this exact sequence. You can automate the whole process using the **Master Workflow** or run steps individually.

### 1️⃣ Master Workflow (Recommended)
Run this single command to execute the full pipeline:
*   **`/full-teaching-project`**

---

### 2️⃣ Manual Execution (Step-by-Step)

If running manually, you **MUST** follow this order to ensure dependencies are met:

#### Phase 1: Foundation
*   **`/create-project-structure`**
    *   *What it does:* Creates folders (`notebook`, `docs`, `slides`, `src`) and `README.md`.
    *   *Why:* All subsequent steps need this folder structure to exist.

#### Phase 2: Core Content
*   **`/convert-py-to-notebook`**
    *   *What it does:* Converts `.py` source to a Teaching Notebook (`.ipynb`).
    *   *Rules:* Applies **Section 5** (Line-by-line 2.1-2.7) and **Section 6** (Arguments 3.1-3.7).
    *   *Why:* This is the source of truth for the documentation.

#### Phase 3: Deep Dive Documentation
*   **`/create-documentation`**
    *   *What it does:* Generates the 4 mandatory files (Problem, Concepts, Results, Interview).
    *   *Rules:* Applies **Section 11** (12-point concept breakdown, detailed interview prep).
    *   *Why:* Expands on the notebook content for exam/interview prep.

#### Phase 4: Visual Summary
*   **`/create-slides`**
    *   *What it does:* Creates NotebookLM-style slides (Markdown + PDF).
    *   *Rules:* Applies **Section 12** (14-slide structure, clean visuals).
    *   *Why:* formatting the content for presentation/review.

---

## 🛠️ Helper Workflows
These are used *inside* the main workflows but can be used standalone for quick tasks:

*   **`/explain-code-line`**: Generates a Section 5 (2.1-2.7) explanation for a single snippet.
*   **`/explain-function`**: Generates a Section 6 (3.1-3.7) explanation for a function call.
*   **`/interview-prep`**: Generates a revision sheet for a specific topic.

## 📂 Project Structure Standard
Every project created will look like this:

```
📁 <project_name>/
├── 📁 notebook/           # The Teaching Notebook
├── 📁 documentation/      # The 4 Deep-Dive Markdown files
├── 📁 slides/             # The Presentation (MD + PDF)
├── 📁 src/                # The Original Source Code
├── 📁 outputs/            # Execution Artifacts
└── README.md              # Project Guide
```
