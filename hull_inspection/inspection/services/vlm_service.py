"""
Vision-Language Model (VLM) Service for Maritime Inspection Reports

This service uses a free VLM (Moondream2) to generate professional
maritime inspection reports based on YOLO detections and the hull image.
"""
import os
import base64
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io


class VLMReportGenerator:
    """
    Service for generating maritime inspection reports using VLM.
    
    Uses Moondream2 - a lightweight, free, and efficient vision-language model
    that can run locally without requiring expensive GPU resources.
    """
    
    # Maritime surveyor prompt template
    SYSTEM_PROMPT = """You are an experienced Class A Marine Surveyor with over 20 years of experience 
in vessel hull inspections. You are conducting a detailed visual inspection of a ship's hull 
and must provide a professional maritime inspection report.

Your expertise includes:
- Hull structural integrity assessment
- Corrosion classification (severe, moderate, mild)
- Marine growth and fouling evaluation
- Paint system condition assessment
- Identification of cracks, dents, and surface defects
- IMO and classification society standards compliance

You must write reports in a formal, technical, and professional tone appropriate for 
insurance underwriters, ship owners, and classification societies."""
    
    REPORT_TEMPLATE = """Based on the hull image provided and the YOLO detection results below, 
generate a comprehensive maritime inspection report.

=== YOLO DETECTION RESULTS ===
{detection_summary}

=== DETAILED DETECTIONS ===
{detection_details}

Please generate a complete inspection report with the following sections:

1. **GENERAL DESCRIPTION**
   - Hull section type and location estimate
   - Environmental context (dry dock, waterline, underwater, etc.)
   - Visible structural elements
   - Overall visual condition assessment

2. **YOLO DETECTION SUMMARY**
   - List all detected defects with confidence scores
   - Highlight critical/severe findings
   - Note any patterns in defect distribution

3. **CONDITION ASSESSMENT**
   - Detailed analysis of each defect type found
   - Severity classification with justification
   - Cross-verification: Do visual observations match YOLO detections?
   - Note any potential defects YOLO may have missed
   - Structural implications of findings

4. **SURROUNDINGS & SAFETY CONTEXT**
   - Environmental factors affecting hull condition
   - Safety considerations for crew and vessel operation
   - Factors that may accelerate deterioration

5. **INSPECTION NOTES**
   - Key observations in surveyor-style bullet points
   - Confidence-based priority ranking
   - Areas requiring closer examination

6. **SUMMARY & RECOMMENDATIONS**
   - Overall severity level: [CRITICAL / SEVERE / MODERATE / MILD / GOOD]
   - Recommended maintenance actions with priority
   - Timeline recommendations for repairs
   - Classification society notification requirements (if any)

7. **MODEL AGREEMENT NOTE**
   - Assessment of YOLO model accuracy based on visual inspection
   - Any discrepancies between AI detection and professional judgment
   - Recommendations for model improvement

Format the report professionally with clear section headers and structured content."""

    def __init__(self, model_name: str = "vikhyatk/moondream2"):
        """
        Initialize the VLM service.
        
        Args:
            model_name: HuggingFace model name for the VLM
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._initialized = False
        
    def _initialize_model(self):
        """Load the VLM model and processor."""
        if self._initialized:
            return
            
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading VLM model: {self.model_name}")
            
            # Load Moondream2 model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self._initialized = True
            print("VLM model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Run: pip install transformers torch"
            )
        except Exception as e:
            print(f"Warning: Could not load VLM model: {e}")
            print("Falling back to template-based report generation")
            self._initialized = False
    
    def generate_report(
        self,
        image_path: str,
        detections: List[Dict],
        use_vlm: bool = True
    ) -> Tuple[str, str, str]:
        """
        Generate a maritime inspection report.
        
        Args:
            image_path: Path to the hull image
            detections: YOLO detection results
            use_vlm: Whether to use VLM (falls back to template if False)
            
        Returns:
            Tuple of (report_text, report_html, overall_severity)
        """
        # Create detection summary
        detection_summary = self._create_detection_summary(detections)
        detection_details = self._format_detection_details(detections)
        
        # Determine overall severity
        overall_severity = self._determine_overall_severity(detections)
        
        if use_vlm:
            try:
                self._initialize_model()
                if self._initialized:
                    report_text = self._generate_vlm_report(
                        image_path, 
                        detection_summary,
                        detection_details
                    )
                else:
                    report_text = self._generate_template_report(
                        detection_summary,
                        detection_details,
                        overall_severity
                    )
            except Exception as e:
                print(f"VLM generation failed: {e}")
                report_text = self._generate_template_report(
                    detection_summary,
                    detection_details,
                    overall_severity
                )
        else:
            report_text = self._generate_template_report(
                detection_summary,
                detection_details,
                overall_severity
            )
        
        # Convert to HTML
        report_html = self._convert_to_html(report_text)
        
        return report_text, report_html, overall_severity
    
    def _generate_vlm_report(
        self,
        image_path: str,
        detection_summary: str,
        detection_details: str
    ) -> str:
        """Generate report using VLM."""
        import torch
        from PIL import Image
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Create prompt
        prompt = self.REPORT_TEMPLATE.format(
            detection_summary=detection_summary,
            detection_details=detection_details
        )
        
        # Generate using Moondream2
        enc_image = self.model.encode_image(image)
        
        # Add system context
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
        
        # Generate response
        response = self.model.answer_question(
            enc_image,
            full_prompt,
            self.tokenizer,
            max_new_tokens=2048
        )
        
        return response
    
    def _generate_template_report(
        self,
        detection_summary: str,
        detection_details: str,
        overall_severity: str
    ) -> str:
        """Generate a template-based report when VLM is unavailable."""
        
        report = f"""
================================================================================
                    MARITIME HULL INSPECTION REPORT
================================================================================

Document ID: AUTO-GENERATED
Date: {self._get_current_date()}
Inspector: AI-Assisted Analysis System
Classification: PRELIMINARY

================================================================================
1. GENERAL DESCRIPTION
================================================================================

Hull Section: External hull plating (location to be confirmed by surveyor)
Environment: Dry dock / inspection environment
Structural Elements: Hull plating, welded seams, coating system

This inspection report is generated using AI-assisted defect detection 
technology. The analysis combines computer vision (YOLO object detection) 
with structured reporting to identify and classify hull defects.

================================================================================
2. YOLO DETECTION SUMMARY
================================================================================

{detection_summary}

================================================================================
3. DETAILED DETECTION ANALYSIS
================================================================================

{detection_details}

================================================================================
4. CONDITION ASSESSMENT
================================================================================

Overall Severity Level: **{overall_severity.upper()}**

{self._get_severity_description(overall_severity)}

Cross-Verification Notes:
- AI detection results should be verified by qualified marine surveyor
- High-confidence detections (>85%) are likely accurate
- Low-confidence detections (<50%) require manual verification
- Areas with multiple overlapping defects may indicate severe deterioration

================================================================================
5. SURROUNDINGS & SAFETY CONTEXT
================================================================================

Environmental Factors:
- Corrosion progression depends on operational environment
- Marine growth indicates extended waterline exposure
- Paint damage accelerates underlying corrosion

Safety Considerations:
- Severe corrosion may compromise structural integrity
- Cracks require immediate attention for vessel safety
- Proper ventilation required during any repair work

================================================================================
6. INSPECTION NOTES
================================================================================

• Detection confidence levels indicate AI model certainty
• Higher confidence (>80%) correlates with clearer defect visibility
• Multiple defect types in proximity suggest systemic maintenance issues
• Periodic re-inspection recommended based on severity findings

================================================================================
7. SUMMARY & RECOMMENDATIONS
================================================================================

Overall Assessment: {overall_severity.upper()}

{self._get_recommendations(overall_severity)}

================================================================================
8. MODEL AGREEMENT NOTE
================================================================================

This report was generated using automated defect detection (YOLOv8) combined
with template-based reporting. For full AI-narrative reports, ensure the
Vision-Language Model (VLM) is properly configured.

AI Detection Accuracy Notes:
- Model trained on maritime corrosion dataset
- Detection accuracy varies with image quality and lighting

================================================================================
                              END OF REPORT
================================================================================
"""
        return report
    
    def _create_detection_summary(self, detections: List[Dict]) -> str:
        """Create a summary of all detections."""
        if not detections:
            return "No defects detected by YOLO model."
        
        summary_lines = [
            f"Total Defects Detected: {len(detections)}",
            "",
            "Defect Distribution:"
        ]
        
        # Count by class
        class_counts = {}
        severity_counts = {}
        
        for det in detections:
            class_name = det['class_name']
            severity = det.get('severity', 'unknown')
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            summary_lines.append(f"  • {class_name}: {count} instance(s)")
        
        summary_lines.extend([
            "",
            "Severity Distribution:"
        ])
        
        for severity, count in sorted(severity_counts.items()):
            summary_lines.append(f"  • {severity.upper()}: {count} defect(s)")
        
        # Calculate average confidence
        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
        summary_lines.extend([
            "",
            f"Average Detection Confidence: {avg_conf:.1%}"
        ])
        
        return "\n".join(summary_lines)
    
    def _format_detection_details(self, detections: List[Dict]) -> str:
        """Format detailed detection information."""
        if not detections:
            return "No detailed detections to report."
        
        details = []
        for i, det in enumerate(detections, 1):
            detail = f"""
Defect #{i}:
  Type: {det['class_name']}
  Confidence: {det['confidence']:.1%}
  Severity: {det.get('severity', 'N/A').upper()}
  Location: Center ({det['x_center']:.2f}, {det['y_center']:.2f})
  Size: {det['width']:.2%} x {det['height']:.2%} of image
  Area Coverage: {det.get('area_percent', 0):.2f}%
"""
            details.append(detail)
        
        return "\n".join(details)
    
    def _determine_overall_severity(self, detections: List[Dict]) -> str:
        """Determine overall severity based on detections."""
        if not detections:
            return 'good'
        
        severity_priority = {
            'critical': 4,
            'severe': 3,
            'moderate': 2,
            'mild': 1
        }
        
        max_severity = 'mild'
        max_priority = 0
        
        for det in detections:
            severity = det.get('severity', 'mild')
            priority = severity_priority.get(severity, 0)
            
            if priority > max_priority:
                max_priority = priority
                max_severity = severity
        
        # Also consider number of defects
        if len(detections) > 10 and max_severity in ['mild', 'moderate']:
            return 'moderate' if max_severity == 'mild' else 'severe'
        
        return max_severity
    
    def _get_severity_description(self, severity: str) -> str:
        """Get description for severity level."""
        descriptions = {
            'critical': """
CRITICAL CONDITION - IMMEDIATE ACTION REQUIRED

The hull exhibits critical defects that pose immediate risk to vessel 
structural integrity and seaworthiness. Operations should be suspended 
until repairs are completed and verified by Class surveyor.
""",
            'severe': """
SEVERE CONDITION - URGENT REPAIRS NEEDED

Significant defects detected that require urgent attention. Vessel may 
continue limited operations with enhanced monitoring, but repairs should 
be scheduled within 30 days.
""",
            'moderate': """
MODERATE CONDITION - REPAIRS RECOMMENDED

Defects of moderate severity detected. Standard maintenance procedures 
should address these issues during next scheduled dry dock period. 
Regular monitoring recommended.
""",
            'mild': """
MILD CONDITION - MONITOR AND MAINTAIN

Minor defects detected consistent with normal operational wear. Continue 
standard maintenance schedule. No immediate action required.
""",
            'good': """
GOOD CONDITION - NO SIGNIFICANT DEFECTS

No significant defects detected. Hull appears to be in satisfactory 
condition. Continue regular inspection schedule.
"""
        }
        return descriptions.get(severity, descriptions['mild'])
    
    def _get_recommendations(self, severity: str) -> str:
        """Get maintenance recommendations based on severity."""
        recommendations = {
            'critical': """
IMMEDIATE ACTIONS:
1. Suspend vessel operations pending structural assessment
2. Notify Classification Society immediately
3. Commission emergency structural survey
4. Prepare emergency repair plan
5. Inform insurance underwriters

REPAIR PRIORITY: CRITICAL (0-7 days)
""",
            'severe': """
URGENT ACTIONS:
1. Schedule dry dock within 30 days
2. Notify Classification Society of findings
3. Prepare detailed repair specification
4. Source materials and contractor availability
5. Consider operational limitations until repair

REPAIR PRIORITY: HIGH (7-30 days)
""",
            'moderate': """
RECOMMENDED ACTIONS:
1. Include in next scheduled maintenance period
2. Document all defect locations for tracking
3. Prepare coating touch-up materials
4. Schedule follow-up inspection in 6 months
5. Monitor identified areas during routine checks

REPAIR PRIORITY: MEDIUM (30-90 days)
""",
            'mild': """
MAINTENANCE ACTIONS:
1. Continue normal maintenance schedule
2. Address minor defects during routine upkeep
3. Document current condition as baseline
4. Re-inspect in 12 months or per schedule

REPAIR PRIORITY: LOW (90+ days / Next Scheduled)
""",
            'good': """
MAINTENANCE ACTIONS:
1. Continue regular inspection schedule
2. Maintain current coating system
3. Document condition for baseline records
4. Schedule next inspection per classification rules

REPAIR PRIORITY: ROUTINE MAINTENANCE
"""
        }
        return recommendations.get(severity, recommendations['mild'])
    
    def _get_current_date(self) -> str:
        """Get current date formatted."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    def _convert_to_html(self, report_text: str) -> str:
        """Convert plain text report to HTML."""
        import html
        
        # Escape HTML characters
        escaped = html.escape(report_text)
        
        # Convert markdown-style formatting
        lines = escaped.split('\n')
        html_lines = []
        
        in_section = False
        
        for line in lines:
            # Convert section headers (===)
            if line.startswith('=' * 10):
                continue
            
            # Convert bold (**text**)
            while '**' in line:
                line = line.replace('**', '<strong>', 1)
                line = line.replace('**', '</strong>', 1)
            
            # Convert bullet points
            if line.strip().startswith('•'):
                line = f'<li>{line.strip()[1:].strip()}</li>'
            elif line.strip().startswith('-'):
                line = f'<li>{line.strip()[1:].strip()}</li>'
            
            # Convert numbered items
            if line.strip() and line.strip()[0].isdigit() and '.' in line[:3]:
                line = f'<li>{line.strip()[2:].strip()}</li>'
            
            # Detect section headers
            if any(x in line for x in ['GENERAL DESCRIPTION', 'DETECTION SUMMARY', 
                                       'CONDITION ASSESSMENT', 'SURROUNDINGS',
                                       'INSPECTION NOTES', 'SUMMARY', 'MODEL AGREEMENT',
                                       'MARITIME HULL INSPECTION']):
                line = f'<h2 class="report-section">{line.strip()}</h2>'
            
            html_lines.append(line)
        
        html_content = '\n'.join(html_lines)
        
        # Wrap in styled container
        return f"""
<div class="inspection-report">
    <pre class="report-content">{html_content}</pre>
</div>
"""


# For testing
if __name__ == '__main__':
    generator = VLMReportGenerator()
    
    test_detections = [
        {
            'class_name': 'corrosion_severe',
            'confidence': 0.92,
            'severity': 'critical',
            'x_center': 0.5,
            'y_center': 0.5,
            'width': 0.1,
            'height': 0.1,
            'area_percent': 1.0
        },
        {
            'class_name': 'paint_damage',
            'confidence': 0.78,
            'severity': 'moderate',
            'x_center': 0.3,
            'y_center': 0.7,
            'width': 0.05,
            'height': 0.08,
            'area_percent': 0.4
        }
    ]
    
    report, html, severity = generator.generate_report(
        "/path/to/test/image.jpg",
        test_detections,
        use_vlm=False  # Test template mode
    )
    
    print(report)
    print(f"\nOverall Severity: {severity}")
