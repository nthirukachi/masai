"""
PDF Slide Generator for MLP Decision Boundaries Project
========================================================
This script converts the slides.md into a PDF presentation.
"""

from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import re

# Define colors
DARK_BG = HexColor('#1a1a2e')
ACCENT = HexColor('#4361ee')
TEXT_COLOR = HexColor('#333333')
HEADER_COLOR = HexColor('#1a1a2e')

def create_slides_pdf(output_path):
    """Creates a PDF from the slides content."""
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(LETTER),
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=HEADER_COLOR,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=ACCENT,
        spaceAfter=15,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HEADER_COLOR,
        spaceBefore=10,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=12,
        textColor=TEXT_COLOR,
        spaceAfter=6,
        leading=16
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=11,
        textColor=TEXT_COLOR,
        leftIndent=20,
        spaceAfter=4,
        bulletIndent=10,
        leading=14
    )
    
    story = []
    
    # Slide 1: Title
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("🧠 MLP Decision Boundaries", title_style))
    story.append(Paragraph("Comparing Activation Functions on make_moons", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Objective: Understand how ReLU, Sigmoid, and Tanh create different decision boundaries", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    key_questions = [
        "• How do different activations shape boundaries?",
        "• Which activation works best?",
        "• Why does the choice matter?"
    ]
    for q in key_questions:
        story.append(Paragraph(q, bullet_style))
    
    # Page break
    from reportlab.platypus import PageBreak
    story.append(PageBreak())
    
    # Slide 2: Problem Statement
    story.append(Paragraph("🧩 The Challenge", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Scenario:</b> Classify points in the 'two moons' pattern", body_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("The make_moons dataset creates two interleaving half-moon shapes.", body_style))
    story.append(Paragraph("<b>Challenge:</b> A straight line CANNOT separate these!", body_style))
    story.append(Paragraph("<b>Solution:</b> Use neural networks with different activation functions", body_style))
    story.append(PageBreak())
    
    # Slide 3: Real-World Use Cases
    story.append(Paragraph("🌍 Real-World Applications", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    app_data = [
        ['Domain', 'Example'],
        ['Medical', 'Classify tumors as benign/malignant'],
        ['Email', 'Spam vs legitimate email'],
        ['Finance', 'Fraud vs normal transactions'],
        ['Vision', 'Cat vs dog in images']
    ]
    
    app_table = Table(app_data, colWidths=[2*inch, 5*inch])
    app_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(app_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Key Insight:</b> Real data is rarely linearly separable!", body_style))
    story.append(PageBreak())
    
    # Slide 4: Dataset
    story.append(Paragraph("📊 The make_moons Dataset", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("from sklearn.datasets import make_moons", body_style))
    story.append(Paragraph("X, y = make_moons(n_samples=300, noise=0.2, random_state=42)", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    data_params = [
        ['Parameter', 'Value', 'Purpose'],
        ['n_samples', '300', 'Total data points'],
        ['noise', '0.2', 'Adds realism'],
        ['random_state', '42', 'Reproducibility']
    ]
    data_table = Table(data_params, colWidths=[2*inch, 2*inch, 3*inch])
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(data_table)
    story.append(PageBreak())
    
    # Slide 5: Key Concepts
    story.append(Paragraph("🔑 Key Concepts", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>1. Neural Network (MLP)</b>", heading_style))
    story.append(Paragraph("• Learns complex patterns through layers of neurons", bullet_style))
    story.append(Paragraph("• Architecture: Input → Hidden → Output", bullet_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>2. Activation Functions</b>", heading_style))
    story.append(Paragraph("• ReLU, Sigmoid, Tanh", bullet_style))
    story.append(Paragraph("• Transform neuron outputs non-linearly", bullet_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>3. Decision Boundary</b>", heading_style))
    story.append(Paragraph("• Where prediction changes from one class to another", bullet_style))
    story.append(Paragraph("• Visualizes how model 'sees' the data", bullet_style))
    story.append(PageBreak())
    
    # Slide 6: Activation Functions
    story.append(Paragraph("⚡ Activation Functions Breakdown", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    act_data = [
        ['Activation', 'Formula', 'Range', 'Boundary Shape'],
        ['ReLU', 'max(0, x)', '[0, ∞)', 'Angular'],
        ['Logistic', '1/(1+e^-x)', '(0, 1)', 'Smooth'],
        ['Tanh', '(e^x-e^-x)/(e^x+e^-x)', '(-1, 1)', 'Smooth']
    ]
    act_table = Table(act_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 1.5*inch])
    act_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(act_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Analogies:</b>", body_style))
    story.append(Paragraph("• ReLU = One-way valve (positive flows, negative blocked)", bullet_style))
    story.append(Paragraph("• Sigmoid = Dimmer switch (smoothly scales 0 to 1)", bullet_style))
    story.append(Paragraph("• Tanh = Centered dimmer (-1 to 1)", bullet_style))
    story.append(PageBreak())
    
    # Slide 7: Solution Flow
    story.append(Paragraph("🪜 Solution Flow", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    steps = [
        "Step 1: Generate make_moons dataset (300 samples)",
        "Step 2: Create 3 MLPClassifier models (ReLU, Logistic, Tanh)",
        "Step 3: Train all models on the same data",
        "Step 4: Create meshgrid and predict on all points",
        "Step 5: Visualize decision boundaries with contourf",
        "Step 6: Compare accuracies and analyze results"
    ]
    for i, step in enumerate(steps):
        story.append(Paragraph(f"<b>{step}</b>", body_style))
        if i < len(steps) - 1:
            story.append(Paragraph("↓", body_style))
    story.append(PageBreak())
    
    # Slide 8: Code Summary
    story.append(Paragraph("💻 Code Logic Summary", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    code_lines = [
        "<b># 1. Data Generation</b>",
        "X, y = make_moons(n_samples=300, noise=0.2, random_state=42)",
        "",
        "<b># 2. Model Creation</b>",
        "model = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', random_state=42)",
        "",
        "<b># 3. Training</b>",
        "model.fit(X, y)",
        "",
        "<b># 4. Visualization</b>",
        "Z = model.predict(meshgrid_points)",
        "plt.contourf(xx, yy, Z, alpha=0.8)"
    ]
    for line in code_lines:
        story.append(Paragraph(line, body_style))
    story.append(PageBreak())
    
    # Slide 9: Parameters
    story.append(Paragraph("⚙️ Important Parameters", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    param_data = [
        ['Parameter', 'Value', 'Effect'],
        ['hidden_layer_sizes', '(8,)', '1 layer, 8 neurons'],
        ['activation', 'varies', 'Boundary shape'],
        ['solver', 'adam', 'Optimization'],
        ['max_iter', '1000', 'Training cycles'],
        ['random_state', '42', 'Fair comparison']
    ]
    param_table = Table(param_data, colWidths=[2.5*inch, 2*inch, 3*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(param_table)
    story.append(PageBreak())
    
    # Slide 10: Results
    story.append(Paragraph("📈 Results", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    results_data = [
        ['Activation', 'Accuracy', 'Rank'],
        ['ReLU', '88.33%', '🥇 1st'],
        ['Tanh', '86.33%', '🥈 2nd'],
        ['Logistic', '85.67%', '🥉 3rd']
    ]
    results_table = Table(results_data, colWidths=[2.5*inch, 2*inch, 2*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Boundary Shapes:</b>", body_style))
    story.append(Paragraph("• ReLU creates angular, piecewise-linear edges", bullet_style))
    story.append(Paragraph("• Sigmoid/Tanh create smooth, curved boundaries", bullet_style))
    story.append(PageBreak())
    
    # Slide 11: Observations
    story.append(Paragraph("🔍 Observations & Insights", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    observations = [
        "<b>1. ReLU creates ANGULAR boundaries</b> - Due to piecewise-linear nature",
        "<b>2. Sigmoid/Tanh create SMOOTH boundaries</b> - Due to continuous curves",
        "<b>3. All accuracies similar (~85-88%)</b> - Dataset is 'easy' for 8 neurons",
        "<b>4. ReLU wins by small margin</b> - Advantages more visible in deep networks"
    ]
    for obs in observations:
        story.append(Paragraph(obs, body_style))
        story.append(Spacer(1, 0.1*inch))
    story.append(PageBreak())
    
    # Slide 12: Trade-offs
    story.append(Paragraph("⚖️ Advantages & Limitations", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>ReLU</b>", heading_style))
    story.append(Paragraph("✅ No vanishing gradient, fast computation, modern default", bullet_style))
    story.append(Paragraph("❌ Dead neurons possible, not zero-centered", bullet_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Sigmoid/Tanh</b>", heading_style))
    story.append(Paragraph("✅ Bounded output, probability interpretation, smooth gradients", bullet_style))
    story.append(Paragraph("❌ Vanishing gradient, slower computation", bullet_style))
    story.append(PageBreak())
    
    # Slide 13: Interview Takeaways
    story.append(Paragraph("💼 Interview Key Takeaways", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    takeaways = [
        "1. ReLU = max(0, x) → Default for hidden layers",
        "2. Sigmoid = 1/(1+e^-x) → Binary output layers",
        "3. Tanh → RNNs, zero-centered needed",
        "4. Vanishing gradient → Sigmoid/Tanh problem, not ReLU",
        "5. hidden_layer_sizes=(8,) → Tuple notation with comma!",
        "6. random_state → For reproducibility"
    ]
    for t in takeaways:
        story.append(Paragraph(f"<b>{t}</b>", body_style))
        story.append(Spacer(1, 0.1*inch))
    story.append(PageBreak())
    
    # Slide 14: Conclusion
    story.append(Paragraph("🎯 Conclusion", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>What We Learned:</b>", heading_style))
    story.append(Paragraph("• Different activations create different boundary shapes", bullet_style))
    story.append(Paragraph("• ReLU: Angular, Sigmoid/Tanh: Smooth", bullet_style))
    story.append(Paragraph("• For simple data, differences are small", bullet_style))
    story.append(Spacer(1, 0.2*inch))
    
    rec_data = [
        ['Scenario', 'Use'],
        ['Hidden layers', 'ReLU'],
        ['Binary output', 'Sigmoid'],
        ['RNNs', 'Tanh'],
        ['Deep networks', 'ReLU (avoid vanishing gradient)']
    ]
    rec_table = Table(rec_data, colWidths=[3*inch, 4*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<i>\"Activation choice matters more in deep networks than shallow ones.\"</i>", body_style))
    
    # Build PDF
    doc.build(story)
    print(f"[OK] PDF generated: {output_path}")

if __name__ == "__main__":
    output_path = "c:/masai/MLP_Decision_Boundaries/slides/slides.pdf"
    create_slides_pdf(output_path)
