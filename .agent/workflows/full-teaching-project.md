---
description: Master workflow to create a complete teaching project. Follows strict order: Structure -> Notebook -> Documentation -> Slides.
---

# Master Teaching Project Workflow

This workflow orchestrates the creation of a full teaching project. It STRICTLY follows the order required to ensure dependencies are met (e.g., code must exist before documentation, documentation before slides).

// turbo-all

## 1. 🏗️ Project Structure (Foundation)
**Workflow:** `/create-project-structure`
**Rationale:** Must exist first to house all artifacts.
- Creates `c:\nagpython\demouv\<project_name>\`
- Creates subfolders: `notebook/`, `src/`, `documentation/`, `slides/`, `outputs/`
- Creates `README.md` with project overview.

## 2. 🐍 to 📓 Notebook Conversion (Core Content)
**Workflow:** `/convert-py-to-notebook`
**Rationale:** The notebook is the primary teaching artifact. It defines the logic and explanations that will be referenced in documentation and slides.
- Copies source script to `src/original_script.py`
- Converts script to `notebook/<project_name>.ipynb`
- **STRICT RULES:**
  - Applies **Section 5 (Line-by-Line 2.1-2.7)**
  - Applies **Section 6 (Arguments 3.1-3.7)**
  - Applies **Section 4 (Problem/Steps/Output)** at the top.

## 3. 📄 Documentation Generation (Deep Dive)
**Workflow:** `/create-documentation`
**Rationale:** Extracts concepts and observations from the now-defined code/notebook.
- Creates `problem_statement.md` (What & Why)
- Creates `concepts_explained.md` (Theory - **Section 11**)
- Creates `observations_and_conclusion.md` (Results)
- Creates `interview_preparation.md` (Revision)

## 4. 🎴 Slide Deck Creation (Visual Summary)
**Workflow:** `/create-slides`
**Rationale:** Summarizes the entire project (code + docs) into a presentation.
- Usess content from Notebook and Documentation.
- Generates `notebooklm_style_slides.md` (14 slides strict structure).
- Generates `notebooklm_style_slides.pdf`.

## 5. ✅ Final Verification
**Action:** User Manual Check
- Verify all 4 folders are populated.
- Verify `ipynb` runs top-to-bottom.
- Verify PDF slides are readable.
