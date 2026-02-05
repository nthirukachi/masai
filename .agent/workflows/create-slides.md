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
- Render Mermaid diagrams as images (save as PNG first, then embed).
- Use a clean, large font (e.g., Helvetica/Arial).

### ⚠️ IMPORTANT: NO BROWSER FOR PDF GENERATION
- **DO NOT** use browser-based PDF generation (e.g., Playwright, Selenium, Chrome headless)
- **USE** Python libraries: **FPDF2** or **ReportLab**

### Recommended Approach (FPDF2)
**Tooling:** Use Python script with UV: `uv run generate_slides_pdf.py`

```python
# Example PDF generation script template
from fpdf import FPDF
from pathlib import Path

class SlidePDF(FPDF):
    def header(self):
        pass  # No header needed for slides
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Slide {self.page_no()}', align='C')
    
    def add_slide(self, title, content_lines, image_path=None):
        self.add_page()
        # Title
        self.set_font('Helvetica', 'B', 24)
        self.cell(0, 20, title, ln=True, align='C')
        self.ln(10)
        # Content
        self.set_font('Helvetica', '', 14)
        for line in content_lines:
            self.multi_cell(0, 8, line)
        # Image if provided
        if image_path and Path(image_path).exists():
            self.image(image_path, x=30, w=150)
```

### Mermaid Diagram Handling
1. Save Mermaid diagrams as PNG images using Mermaid CLI or Python library
2. Embed the PNG images into the PDF
3. Alternative: Use ASCII art representations for simple flows

**Validation:** Ensure diagrams are visible and text is not cut off.

## UV Execution
```powershell
# Navigate to project slides directory
cd c:\masai\<project_name>\slides

# Install dependencies if needed
uv add fpdf2

# Run PDF generation script with UV
uv run python generate_slides_pdf.py
```

### Alternative: ReportLab
```powershell
uv add reportlab
```

ReportLab provides more control over layout but has a steeper learning curve.