from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


def generate_report(data, filename="gait_report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    # =========================
    # TITLE
    # =========================
    elements.append(Paragraph("GAIT ANALYSIS MEDICAL REPORT", styles['Title']))
    elements.append(Spacer(1, 10))

    # =========================
    # PATIENT INFO
    # =========================
    elements.append(Paragraph(f"File: {data['filename']}", styles['Normal']))
    elements.append(Spacer(1, 10))

    # =========================
    # RISK SECTION 
    # =========================
    risk = data["medical_analysis"]["risk_score"]
    severity = data["medical_analysis"]["severity"]

    if severity == "High":
        color = colors.red
    elif severity == "Moderate":
        color = colors.orange
    else:
        color = colors.green

    risk_table = Table([[f"Risk Score: {risk}%", severity]])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))

    elements.append(risk_table)
    elements.append(Spacer(1, 20))

    # =========================
    # RECOMMENDATION
    # =========================
    elements.append(Paragraph("Doctor Recommendations:", styles['Heading2']))

    for rec in data["medical_analysis"]["recommendation"]:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))

    # =========================
    # BUILD PDF
    # =========================
    doc.build(elements)