---
description: Create NotebookLM-style slides (MD + PDF) following Section 12 rules.
---

# Create Slide Deck Workflow

This workflow generates the **NotebookLM-style** slide deck.
**Location:** `<project_name>/slides/`

## Style Guidelines
- **Visual**: Clean, concise, bullet-based.
- **Content**: Concept-first, end-to-end story.
- **Format**: Markdown (`slides.md`) AND PDF (`slides.pdf`).

## 1. Markdown Slides (`slides.md`)
**Mandatory Structure (14 Slides):**

- **Slide 1: Title & Objective** (Goal + Learning outcomes)
- **Slide 2: Problem Statement** (Problem, Relevance, Challenge)
- **Slide 3: Real-World Use Case** (Industry Examples Table)
- **Slide 4: Input Data** (Source, Features, Size)
- **Slide 5: Concepts Used** (High Level List)
- **Slide 6: Concepts Breakdown** (Simple "What/Why/Analogy" per concept)
- **Slide 7: Step-by-Step Solution Flow** (Mermaid Diagram required)
- **Slide 8: Code Logic Summary** (Import -> Load -> Process -> Train -> Eval)
- **Slide 9: Important Functions** (Table: Function | Purpose | Key Params)
- **Slide 10: Execution Output** (Sample output/metrics)
- **Slide 11: Observations & Insights** (What we noticed vs Business Impact)
- **Slide 12: Advantages & Limitations** (Pros/Cons)
- **Slide 13: Interview Key Takeaways** (Common Q&A)
- **Slide 14: Conclusion** (Result + Next Steps)

## 2. PDF Generation (`slides.pdf`)
**Requirement:** Convert the markdown slides to a clean PDF.
- Each slide = One Page.
- Render Mermaid diagrams as images.
- Use a clean, large font (e.g., Helvetica/Arial).
- **Tooling:** Use Python script with UV: `uv run python generate_pdf.py`
- **Validation:** Ensure diagrams are visible and text is not cut off.

## UV Execution
```powershell
# Navigate to project slides directory
cd c:\masai\<project_name>\slides

# Run PDF generation script with UV
uv run python generate_slides_pdf.py
```
