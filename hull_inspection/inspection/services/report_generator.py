"""
PDF Report Generator for Maritime Hull Inspection

Generates professional PDF reports that can be downloaded and printed.
"""
import os
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


class PDFReportGenerator:
    """
    Service for generating PDF inspection reports.
    
    Uses reportlab for PDF generation with professional maritime
    inspection formatting.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the PDF generator.
        
        Args:
            output_dir: Directory to save generated PDFs
        """
        from django.conf import settings
        self.output_dir = output_dir or str(settings.MEDIA_ROOT / 'reports')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate(self, inspection) -> str:
        """
        Generate a PDF report for an inspection.
        
        Args:
            inspection: InspectionReport model instance
            
        Returns:
            Path to the generated PDF file
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                TableStyle, PageBreak, HRFlowable
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        except ImportError:
            raise ImportError(
                "reportlab package not installed. "
                "Run: pip install reportlab"
            )
        
        # Generate filename
        filename = f"inspection_report_{inspection.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(self.output_dir, filename)
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5282'),
            spaceBefore=20,
            spaceAfter=10
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY
        )
        
        # Build document content
        story = []
        
        # Header
        story.append(Paragraph("MARITIME HULL INSPECTION REPORT", title_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2c5282')))
        story.append(Spacer(1, 20))
        
        # Report metadata
        metadata_data = [
            ['Report ID:', str(inspection.id)],
            ['Date:', inspection.created_at.strftime('%Y-%m-%d %H:%M UTC')],
            ['Status:', inspection.status.upper()],
            ['Vessel:', inspection.vessel_name or 'Not Specified'],
        ]
        
        metadata_table = Table(metadata_data, colWidths=[1.5*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Overall Severity Badge
        severity = inspection.overall_severity or 'unknown'
        severity_colors = {
            'critical': colors.red,
            'severe': colors.orange,
            'moderate': colors.yellow,
            'mild': colors.lightgreen,
            'good': colors.green,
        }
        
        severity_text = f"Overall Condition: {severity.upper()}"
        story.append(Paragraph(severity_text, heading_style))
        story.append(Spacer(1, 20))
        
        # Original Image
        if inspection.original_image:
            story.append(Paragraph("Original Hull Image", heading_style))
            try:
                img = Image(inspection.original_image.path, width=5*inch, height=3.5*inch)
                story.append(img)
            except:
                story.append(Paragraph("[Image could not be loaded]", body_style))
            story.append(Spacer(1, 20))
        
        # Annotated Image
        if inspection.annotated_image:
            story.append(Paragraph("Annotated Detection Image", heading_style))
            try:
                img = Image(inspection.annotated_image.path, width=5*inch, height=3.5*inch)
                story.append(img)
            except:
                story.append(Paragraph("[Image could not be loaded]", body_style))
            story.append(Spacer(1, 20))
        
        # Detection Summary
        story.append(Paragraph("Detection Summary", heading_style))
        
        if inspection.yolo_detections:
            # Create detection table
            detection_data = [['#', 'Defect Type', 'Confidence', 'Severity']]
            
            for i, det in enumerate(inspection.yolo_detections, 1):
                detection_data.append([
                    str(i),
                    det.get('class_name', 'Unknown'),
                    f"{det.get('confidence', 0):.1%}",
                    det.get('severity', 'N/A').upper()
                ])
            
            det_table = Table(detection_data, colWidths=[0.5*inch, 2*inch, 1*inch, 1*inch])
            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(det_table)
        else:
            story.append(Paragraph("No defects detected.", body_style))
        
        story.append(Spacer(1, 20))
        
        # Full Report Content
        story.append(PageBreak())
        story.append(Paragraph("Detailed Inspection Report", heading_style))
        
        if inspection.report_content:
            # Split into paragraphs
            paragraphs = inspection.report_content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Handle section headers
                    if para.strip().startswith('='):
                        continue
                    if any(x in para.upper() for x in ['GENERAL DESCRIPTION', 'SUMMARY', 
                                                        'RECOMMENDATIONS', 'ASSESSMENT']):
                        story.append(Paragraph(para.strip(), heading_style))
                    else:
                        story.append(Paragraph(para.strip(), body_style))
                    story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No detailed report available.", body_style))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.gray))
        story.append(Spacer(1, 10))
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph(
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            footer_style
        ))
        story.append(Paragraph(
            "HullInspect AI - Maritime Hull Inspection System",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def generate_simple_report(self, inspection) -> str:
        """
        Generate a simpler text-based PDF when reportlab is not available.
        Falls back to HTML if needed.
        """
        try:
            return self.generate(inspection)
        except ImportError:
            # Generate HTML file instead
            filename = f"inspection_report_{inspection.id}.html"
            output_path = os.path.join(self.output_dir, filename)
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hull Inspection Report - {inspection.id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #1a365d; }}
        h2 {{ color: #2c5282; border-bottom: 1px solid #2c5282; }}
        .metadata {{ background: #f7fafc; padding: 20px; margin: 20px 0; }}
        .severity {{ padding: 10px 20px; border-radius: 5px; display: inline-block; }}
        .severity-critical {{ background: #fed7d7; color: #c53030; }}
        .severity-severe {{ background: #feebc8; color: #c05621; }}
        .severity-moderate {{ background: #fefcbf; color: #975a16; }}
        .severity-mild {{ background: #c6f6d5; color: #276749; }}
        .severity-good {{ background: #9ae6b4; color: #22543d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
        th {{ background: #2c5282; color: white; }}
        pre {{ background: #f7fafc; padding: 20px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Maritime Hull Inspection Report</h1>
    
    <div class="metadata">
        <p><strong>Report ID:</strong> {inspection.id}</p>
        <p><strong>Date:</strong> {inspection.created_at.strftime('%Y-%m-%d %H:%M UTC')}</p>
        <p><strong>Status:</strong> {inspection.status.upper()}</p>
        <p><strong>Vessel:</strong> {inspection.vessel_name or 'Not Specified'}</p>
    </div>
    
    <div class="severity severity-{inspection.overall_severity or 'unknown'}">
        <strong>Overall Condition:</strong> {(inspection.overall_severity or 'UNKNOWN').upper()}
    </div>
    
    <h2>Detection Summary</h2>
    <p><strong>Total Defects:</strong> {inspection.detection_count}</p>
    
    <h2>Detailed Report</h2>
    <pre>{inspection.report_content or 'No report available.'}</pre>
    
    <hr>
    <p style="color: #718096; font-size: 12px; text-align: center;">
        HullInspect AI - Maritime Hull Inspection System
    </p>
</body>
</html>
"""
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            return output_path
