---
description: Master workflow to create a complete teaching project. Follows strict order as per user requirements.
---

# Master Teaching Project Workflow

This workflow orchestrates the creation of a full teaching project. It STRICTLY follows the execution order required by the user.

// turbo-all

## ⚠️ IMPORTANT RULES

### Rule 1: Explain Like Teaching a 10-Year-Old
- **NO unexplained jargon** - Every technical term MUST be explained simply
- Use **real-life analogies** (games, school, toys, food, sports)
- Example: "Algorithm = Recipe for solving a problem, like a recipe for making a sandwich"

### Rule 2: Use Mermaid Diagrams
- Use Mermaid diagrams wherever possible for visual understanding
- Every concept should have at least one diagram

---

## Execution Order (STRICT - 10 Steps)

### Step 1: 🏗️ Create Project Structure
**Workflow:** `/create-project-structure`
- Creates `c:\masai\<project_name>\`
- Creates subfolders: `notebook/`, `src/`, `documentation/`, `slides/`, `outputs/`
- Creates `README.md` with project overview

### Step 2: 📝 Original_Problem.md
**Location:** `<project_name>/documentation/Original_Problem.md`
**Content:** Copy the EXACT problem statement/usecase provided by the user
- No modifications, no explanations
- Just the raw problem statement as given

### Step 3: 📄 problem_statement.md
**Location:** `<project_name>/documentation/problem_statement.md`
**Content:** 
- Problem explained simply (like for a child)
- Real-life analogy
- Steps to solve (breakdown)
- Expected output with examples
- Mermaid diagram showing flow

### Step 4: 🐍 Create .py File
**Location:** `<project_name>/src/<project_name>.py`
**Content:** Complete Python implementation with:
- Full line-by-line comments (2.1-2.7)
- Argument-by-argument docstrings (3.1-3.7)
- Simple explanations for technical terms
- Working, executable code

### Step 5: 📓 Create .ipynb Notebook
**Workflow:** `/convert-py-to-notebook`
**Location:** `<project_name>/notebook/<project_name>.ipynb`

> [!CAUTION]
> **CRITICAL RULE: EVERY CODE CELL MUST HAVE A MARKDOWN CELL BEFORE IT.**
> - Cell order: [Markdown] → [Code] → [Markdown] → [Code] → ...
> - NO consecutive code cells allowed
> - First cell is ALWAYS markdown (Problem Statement)
> - This rule is NON-NEGOTIABLE

**Content:**
- Convert .py to teaching-oriented notebook
- **MANDATORY:** Create a markdown cell BEFORE each code cell
- Apply Section 4 (Problem/Steps/Output at top)
- Apply Section 5 (Line-by-Line 2.1-2.7) in markdown cells
- Apply Section 6 (Arguments 3.1-3.7) in markdown cells
- Include Mermaid diagrams in markdown cells

### Step 6: 📚 concepts_explained.md
**Location:** `<project_name>/documentation/concepts_explained.md`
**Content:** 
- 12 points per concept (see workflow)
- Jargon glossary with simple explanations
- Mermaid diagrams for each concept
- Real-life analogies

### Step 7: 📊 observations_and_conclusion.md
**Location:** `<project_name>/documentation/observations_and_conclusion.md`
**Content:** 
- Execution output
- Output explanation with diagrams
- Observations in simple language
- Insights and conclusion

### Step 8: 🎤 interview_questions.md (NEW)
**Location:** `<project_name>/documentation/interview_questions.md`
**Content:** 
- **MINIMUM 10-20 Interview Questions**
- Each question with:
  - Simple answer (for 10-year-old)
  - Technical answer (for interviewer)
  - Mermaid diagram
  - Real-life analogy
  - Common mistakes
  - Key points to remember

### Step 9: 📝 exam_preparation.md (NEW)
**Location:** `<project_name>/documentation/exam_preparation.md`
**Content:**
- **Section A: MCQ** (10+ questions)
  - 4 options each
  - Correct answer with explanation
  - Why other options are wrong
- **Section B: MSQ** (5+ questions)
  - Multiple correct answers
  - Explanation for each correct option
- **Section C: Numerical** (5+ questions)
  - Step-by-step solution
  - Final answer with units
- **Section D: Fill in the Blanks** (5+ questions)

### Step 10: 📋 interview_preparation.md
**Location:** `<project_name>/documentation/interview_preparation.md`
**Content:** Quick revision sheet with:
- 30-second summary
- Key terms glossary
- Top 10 points to remember
- Comparison tables
- Cheat sheet
- Mermaid summary diagram

### Step 11: 🎴 slides.md
**Workflow:** `/create-slides`
**Location:** `<project_name>/slides/slides.md`
**Content:** NotebookLM-style 14-slide structure with Mermaid diagrams

### Step 12: 📑 slides.pdf
**Location:** `<project_name>/slides/slides.pdf`
**Action:** Generate PDF from slides.md using Python script with UV

> [!CAUTION]
> **DO NOT USE BROWSER FOR PDF GENERATION**
> - ❌ No Playwright, Selenium, or Chrome headless
> - ✅ Use Python libraries: **FPDF2** or **ReportLab**
> - Run: `uv add fpdf2` then `uv run python generate_slides_pdf.py`

---

## Python Execution with UV

**IMPORTANT:** This project uses UV virtual environment.

### Run Python scripts with UV:
```powershell
# Navigate to project directory
cd c:\masai\<project_name>

# Run with UV
uv run python src/<project_name>.py
```

---

## ✅ Final Verification Checklist
- [ ] `Original_Problem.md` contains exact user input
- [ ] `problem_statement.md` is simple and has Mermaid diagram
- [ ] `.py` file has full comments with simple explanations
- [ ] `.ipynb` runs top-to-bottom with Mermaid diagrams
- [ ] **`.ipynb` has a markdown cell BEFORE every code cell (NO EXCEPTIONS)**
- [ ] **`.ipynb` first cell is markdown (Problem Statement)**
- [ ] **`.ipynb` has NO consecutive code cells**
- [ ] `concepts_explained.md` has 12 points per concept + jargon glossary
- [ ] `observations_and_conclusion.md` is complete
- [ ] `interview_questions.md` has 10-20 Q&A with diagrams
- [ ] `exam_preparation.md` has MCQ/MSQ/Numerical/Fill-in-blanks
- [ ] `interview_preparation.md` is quick revision ready
- [ ] `slides.md` has 14 slides
- [ ] `slides.pdf` is generated and readable
- [ ] All technical terms are explained simply (10-year-old test)

---

## 📋 Notebook Cell Pattern (MANDATORY)

When creating `.ipynb` files, ALWAYS follow this pattern:

```
┌──────────────────────────┐
│   MARKDOWN CELL          │  ← Explanation FIRST
│   (2.1-2.7 / 3.1-3.7)    │
└──────────────────────────┘
          ↓
┌──────────────────────────┐
│   CODE CELL              │  ← Code SECOND
│   (Python code)          │
└──────────────────────────┘
          ↓
       [REPEAT]
```

**Every section = 1 Markdown + 1 Code cell**