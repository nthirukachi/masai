from fpdf import FPDF
import os

class SlidePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'MQ8: Interpreting Clusters for Stakeholders', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_slides_pdf():
    pdf = SlidePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Read the markdown content
    with open('slides/notebooklm_style_slides.md', 'r') as f:
        lines = f.readlines()
        
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for line in lines:
        line = line.strip().encode('ascii', 'ignore').decode('ascii') # Strict ASCII
        if line.startswith('---'):
            pdf.add_page()
        elif line.startswith('## '):
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, line.replace('## ', ''), 0, 1)
            pdf.set_font("Arial", size=12)
        elif line.startswith('### '):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, line.replace('### ', ''), 0, 1)
            pdf.set_font("Arial", size=12)
        elif line.startswith('- '):
            pdf.cell(10) # Indent
            pdf.cell(0, 8, '-' + ' ' + line.replace('- ', ''), 0, 1)
        elif '<!-- slide -->' in line:
            pass
        elif line == '':
            pdf.ln(2)
        else:
            pdf.multi_cell(0, 8, line)
            
    pdf.output("slides/notebooklm_style_slides.pdf")
    print("PDF created successfully: slides/notebooklm_style_slides.pdf")

if __name__ == "__main__":
    try:
        create_slides_pdf()
    except Exception as e:
        print(f"Failed to create PDF: {e}")
