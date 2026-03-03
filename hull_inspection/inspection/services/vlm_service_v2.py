"""
Vision-Language Model (VLM) Service for Maritime Inspection Reports — v2

Optimized for Mac M1. Supports multiple Ollama backends ranked by quality:

1. **minicpm-v** (RECOMMENDED for M1 8GB)
   - 3B params, excellent vision understanding, fast on M1
   - Install: ollama pull minicpm-v

2. **llava:7b** (Good fallback)
   - 7B params, decent vision, runs on M1 8GB
   - Install: ollama pull llava

3. **llava:13b** (Best quality if you have 16GB RAM)
   - 13B params, best analysis quality
   - Install: ollama pull llava:13b

4. **moondream** (Lightest option for 8GB RAM)
   - 1.8B params, fast but less detailed
   - Install: ollama pull moondream

Why Ollama + local models?
- Free, no API costs
- Runs entirely on your Mac M1
- No internet dependency for inference
- Privacy: images never leave your machine

Usage:
    1. Install Ollama: brew install ollama
    2. Pull model: ollama pull minicpm-v
    3. Start Ollama: ollama serve
"""
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import base64
import requests
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
from datetime import datetime


class VLMReportGenerator:
    """
    VLM service for generating maritime inspection reports.
    
    Optimized for Mac M1 with Ollama backends.
    Uses segmentation detection results (area-based severity).
    """
    
    # Ranked list of models to try (best → lightest)
    PREFERRED_MODELS = ['minicpm-v', 'llava', 'llava:13b', 'moondream', 'bakllava']
    
    # Enhanced maritime surveyor prompt — tuned for corrosion segmentation data
    INSPECTION_PROMPT = """You are an experienced Class A Marine Surveyor with 20+ years of hull inspection experience. You are examining a ship hull image with AI-assisted corrosion detection results.

Analyze this hull image carefully and provide a professional assessment:

1. **Hull Condition Overview**: Describe the overall state of the hull surface visible in the image.
2. **Corrosion Analysis**: Describe the type, pattern, and extent of any corrosion you observe (uniform, pitting, crevice, galvanic, etc.).
3. **Coating Assessment**: Comment on the condition of the paint/anti-fouling coating.
4. **Structural Concern Level**: Based on visible corrosion depth and spread, assess if structural integrity could be affected.
5. **Environmental Factors**: Note any evidence of the operational environment (marine growth, waterline marks, etc.).
6. **Recommended Actions**: Provide specific, actionable maintenance recommendations.

Be precise, professional, and detailed. Reference specific areas of the image where you see issues."""

    SEVERITY_CONTEXT_PROMPT = """
AI Segmentation Detection Results:
- Total corrosion regions detected: {num_regions}
- Total corroded area: {total_area:.1f}% of inspected surface
- Largest single corrosion region: {largest_area:.1f}%
- AI-assessed severity: {severity}

Severity breakdown:
{severity_breakdown}

Use these quantitative measurements to support your visual assessment.
Provide your independent expert opinion — you may agree or disagree with the AI severity rating."""

    def __init__(
        self,
        backend: str = "ollama",
        ollama_model: Optional[str] = None,  # Auto-detect best available
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """
        Initialize the VLM service.
        
        Args:
            backend: 'ollama' (recommended)
            ollama_model: Ollama model name (None = auto-detect best available)
            ollama_url: Ollama API URL
            temperature: Generation temperature (0.3 = focused/professional)
            max_tokens: Maximum generation tokens
        """
        self.backend = backend
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = False
        
        # Auto-detect best model if not specified
        if ollama_model:
            self.ollama_model = ollama_model
        else:
            self.ollama_model = self._find_best_model()
    
    def _find_best_model(self) -> str:
        """Find the best available Ollama vision model."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m.get('name', '').split(':')[0] for m in models]
                
                for preferred in self.PREFERRED_MODELS:
                    base = preferred.split(':')[0]
                    if base in available:
                        print(f"✅ Auto-selected VLM: {preferred}")
                        return preferred
                
                # Check for any model with vision capability
                vision_keywords = ['llava', 'vision', 'minicpm', 'moondream', 'bakllava']
                for model in models:
                    name = model.get('name', '').lower()
                    if any(kw in name for kw in vision_keywords):
                        print(f"✅ Found vision model: {model['name']}")
                        return model['name']
                
                print(f"⚠️ No vision model found. Available: {available}")
                print(f"   Install one with: ollama pull minicpm-v")
        except Exception as e:
            print(f"⚠️ Could not detect Ollama models: {e}")
        
        return 'minicpm-v'  # Default recommendation
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and has the vision model."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                # Check both exact match and base name match
                base_model = self.ollama_model.split(':')[0]
                for name in model_names:
                    if name == self.ollama_model or name.split(':')[0] == base_model:
                        return True
                print(f"Model '{self.ollama_model}' not found. Available: {model_names}")
                print(f"Run: ollama pull {self.ollama_model}")
                return False
            return False
        except requests.exceptions.ConnectionError:
            print("Ollama not running. Start with: ollama serve")
            return False
        except Exception as e:
            print(f"Error checking Ollama: {e}")
            return False
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 for Ollama API."""
        # Resize large images to reduce memory usage on M1
        img = Image.open(image_path)
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def generate_report(
        self,
        image_path: str,
        detections: List[Dict],
        use_vlm: bool = True,
    ) -> Tuple[str, str, str]:
        """
        Generate a maritime inspection report.
        
        Args:
            image_path: Path to the hull image
            detections: Detection results from YOLOSegInferenceService
            use_vlm: Whether to use VLM (falls back to template if False/unavailable)
            
        Returns:
            Tuple of (report_text, report_html, overall_severity)
        """
        # Build detection context
        summary = self._build_detection_summary(detections)
        details = self._format_detection_details(detections)
        overall_severity = self._determine_overall_severity(detections)
        
        # Try VLM generation
        vlm_analysis = None
        if use_vlm and self.backend == 'ollama':
            if self._check_ollama_available():
                vlm_analysis = self._call_ollama(image_path, detections, overall_severity)
        
        # Build final report
        if vlm_analysis:
            report_text = self._build_full_report(
                vlm_analysis, summary, details, overall_severity
            )
        else:
            report_text = self._build_template_report(summary, details, overall_severity)
        
        report_html = self._convert_to_html(report_text)
        
        return report_text, report_html, overall_severity
    
    def _call_ollama(
        self,
        image_path: str,
        detections: List[Dict],
        overall_severity: str,
    ) -> Optional[str]:
        """Call Ollama API with the image and context."""
        
        image_base64 = self._encode_image_base64(image_path)
        
        # Build severity context
        total_area = sum(d['area_percent'] for d in detections)
        largest = max((d['area_percent'] for d in detections), default=0)
        
        sev_counts = {}
        for d in detections:
            s = d['severity']
            if s not in sev_counts:
                sev_counts[s] = {'count': 0, 'area': 0}
            sev_counts[s]['count'] += 1
            sev_counts[s]['area'] += d['area_percent']
        
        breakdown = "\n".join(
            f"  - {s.upper()}: {v['count']} region(s), {v['area']:.1f}% area"
            for s, v in sev_counts.items()
        ) or "  No detections"
        
        context = self.SEVERITY_CONTEXT_PROMPT.format(
            num_regions=len(detections),
            total_area=total_area,
            largest_area=largest,
            severity=overall_severity.upper(),
            severity_breakdown=breakdown,
        )
        
        prompt = f"{self.INSPECTION_PROMPT}\n\n{context}"
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": 4096,
                    }
                },
                timeout=180,
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', '').strip()
                if analysis:
                    print(f"✅ VLM analysis generated ({len(analysis)} chars)")
                    return analysis
                print("⚠️ VLM returned empty response")
            else:
                print(f"⚠️ Ollama API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("⚠️ Ollama request timed out (180s)")
        except Exception as e:
            print(f"⚠️ Ollama error: {e}")
        
        return None
    
    def _build_detection_summary(self, detections: List[Dict]) -> str:
        """Build detection summary text."""
        if not detections:
            return "No corrosion detected by the AI segmentation model."
        
        total_area = sum(d['area_percent'] for d in detections)
        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
        
        lines = [
            f"Total Corrosion Regions: {len(detections)}",
            f"Total Corroded Area: {total_area:.2f}% of inspected surface",
            f"Average Detection Confidence: {avg_conf:.1%}",
            "",
            "Severity Distribution (based on segmented area):",
        ]
        
        sev_counts = {}
        for d in detections:
            s = d['severity']
            if s not in sev_counts:
                sev_counts[s] = {'count': 0, 'area': 0}
            sev_counts[s]['count'] += 1
            sev_counts[s]['area'] += d['area_percent']
        
        for sev in ['critical', 'severe', 'moderate', 'mild']:
            if sev in sev_counts:
                v = sev_counts[sev]
                lines.append(f"  • {sev.upper()}: {v['count']} region(s), {v['area']:.2f}% area")
        
        return "\n".join(lines)
    
    def _format_detection_details(self, detections: List[Dict]) -> str:
        """Format detailed detection information."""
        if not detections:
            return "No corrosion regions to report."
        
        details = []
        for i, det in enumerate(detections, 1):
            detail = f"""
Region #{i}:
  Confidence: {det['confidence']:.1%}
  Severity: {det['severity'].upper()}
  Corroded Area: {det['area_percent']:.2f}% of image ({det.get('mask_pixels', 'N/A')} pixels)
  Location: Center ({det['x_center']:.2f}, {det['y_center']:.2f})
  Bounding Box: {det['width']:.1%} x {det['height']:.1%} of image"""
            details.append(detail)
        
        return "\n".join(details)
    
    def _determine_overall_severity(self, detections: List[Dict]) -> str:
        """Determine overall severity based on total corrosion area."""
        if not detections:
            return 'good'
        
        total_area = sum(d['area_percent'] for d in detections)
        num_regions = len(detections)
        
        # Area-based severity
        if total_area > 15.0:
            return 'critical'
        elif total_area > 8.0:
            return 'severe'
        elif total_area > 3.0:
            return 'moderate'
        elif total_area > 0.5:
            return 'mild'
        
        # Also consider number of regions
        if num_regions > 10:
            return 'moderate'
        
        return 'good' if total_area == 0 else 'mild'
    
    def _build_full_report(
        self,
        vlm_analysis: str,
        detection_summary: str,
        detection_details: str,
        overall_severity: str,
    ) -> str:
        """Build comprehensive report with VLM analysis."""
        
        return f"""
================================================================================
                    MARITIME HULL INSPECTION REPORT
================================================================================

Document ID: AUTO-GENERATED
Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
Inspector: AI-Assisted Analysis System (YOLO11-Seg + {self.ollama_model})
Method: Instance Segmentation (pixel-level corrosion mapping)
Classification: PRELIMINARY

================================================================================
1. EXPERT VISUAL ANALYSIS (VLM: {self.ollama_model})
================================================================================

{vlm_analysis}

================================================================================
2. AI SEGMENTATION RESULTS
================================================================================

{detection_summary}

================================================================================
3. DETAILED REGION ANALYSIS
================================================================================

{detection_details}

================================================================================
4. CONDITION ASSESSMENT
================================================================================

Overall Severity Level: **{overall_severity.upper()}**

{self._get_severity_description(overall_severity)}

Methodology Notes:
- Corrosion area measured via pixel-level instance segmentation (not bounding boxes)
- Area percentages represent actual corroded pixels, not rectangular approximations
- Severity is based on total corroded surface area
- VLM provides qualitative expert-level contextual analysis
- Combined AI assessment should be verified by qualified marine surveyor

================================================================================
5. SAFETY & ENVIRONMENTAL CONTEXT
================================================================================

Environmental Factors:
- Corrosion progression depends on operational environment (saltwater, temperature)
- Marine growth presence indicates extended waterline exposure
- Paint/coating damage accelerates underlying corrosion development
- Pitting corrosion poses higher structural risk than uniform surface corrosion

Safety Considerations:
- Corroded areas exceeding 5% warrant immediate structural assessment
- Pitting corrosion may be deeper than surface appearance suggests
- Crevice corrosion around fittings requires close manual inspection
- Proper ventilation and safety equipment required during any repair work

================================================================================
6. RECOMMENDATIONS
================================================================================

{self._get_recommendations(overall_severity)}

================================================================================
7. ANALYSIS METHODOLOGY
================================================================================

Detection Model: YOLO11-Seg (Instance Segmentation)
- Pixel-accurate corrosion boundary detection
- Precise area measurement from segmentation masks
- Trained on MaVeCoDD maritime corrosion dataset

Vision-Language Model: Ollama ({self.ollama_model})
- Professional maritime inspection narrative
- Contextual analysis beyond what detection models capture
- Runs locally on Mac M1 — no cloud dependency

Cross-Verification:
- Segmentation model provides quantitative measurements
- VLM provides qualitative expert-level analysis
- Combined approach offers comprehensive assessment

================================================================================
                              END OF REPORT
================================================================================
"""
    
    def _build_template_report(
        self,
        detection_summary: str,
        detection_details: str,
        overall_severity: str,
    ) -> str:
        """Template-based report when VLM is unavailable."""
        
        return f"""
================================================================================
                    MARITIME HULL INSPECTION REPORT
================================================================================

Document ID: AUTO-GENERATED
Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
Inspector: AI-Assisted Analysis System (YOLO11-Seg)
Method: Instance Segmentation (pixel-level corrosion mapping)
Classification: PRELIMINARY

Note: VLM visual analysis unavailable. Install Ollama and a vision model
for enhanced reporting: ollama pull minicpm-v && ollama serve

================================================================================
1. AI SEGMENTATION RESULTS
================================================================================

{detection_summary}

================================================================================
2. DETAILED REGION ANALYSIS
================================================================================

{detection_details}

================================================================================
3. CONDITION ASSESSMENT
================================================================================

Overall Severity Level: **{overall_severity.upper()}**

{self._get_severity_description(overall_severity)}

================================================================================
4. RECOMMENDATIONS
================================================================================

{self._get_recommendations(overall_severity)}

================================================================================
5. ANALYSIS NOTES
================================================================================

- Detection uses YOLO11-Seg instance segmentation for precise area measurement
- Corrosion boundaries are pixel-accurate (not bounding-box approximations)
- For enhanced reports with expert narrative, configure Ollama VLM
- Final assessment should always be verified by a qualified marine surveyor

================================================================================
                              END OF REPORT
================================================================================
"""
    
    def _get_severity_description(self, severity: str) -> str:
        """Get description for severity level."""
        descriptions = {
            'critical': (
                "CRITICAL CONDITION — IMMEDIATE ACTION REQUIRED\n\n"
                "The hull exhibits extensive corrosion (>15% affected area) that poses\n"
                "immediate risk to structural integrity and seaworthiness. Operations\n"
                "should be suspended until repairs are completed and verified."
            ),
            'severe': (
                "SEVERE CONDITION — URGENT REPAIRS NEEDED\n\n"
                "Significant corrosion detected (8-15% affected area). Vessel may\n"
                "continue limited operations with enhanced monitoring, but repairs\n"
                "should be scheduled within 30 days."
            ),
            'moderate': (
                "MODERATE CONDITION — REPAIRS RECOMMENDED\n\n"
                "Moderate corrosion detected (3-8% affected area). Standard maintenance\n"
                "procedures should address these during next scheduled dry dock.\n"
                "Regular monitoring recommended."
            ),
            'mild': (
                "MILD CONDITION — MONITOR AND MAINTAIN\n\n"
                "Minor corrosion detected (0.5-3% affected area), consistent with\n"
                "normal operational wear. Continue standard maintenance schedule."
            ),
            'good': (
                "GOOD CONDITION — NO SIGNIFICANT CORROSION\n\n"
                "No significant corrosion detected. Hull appears in satisfactory\n"
                "condition. Continue regular inspection schedule."
            ),
        }
        return descriptions.get(severity, descriptions['mild'])
    
    def _get_recommendations(self, severity: str) -> str:
        """Get maintenance recommendations based on severity."""
        recommendations = {
            'critical': """IMMEDIATE ACTIONS:
1. Suspend vessel operations pending structural assessment
2. Notify Classification Society immediately
3. Commission emergency structural survey with ultrasonic thickness measurements
4. Prepare emergency repair plan — consider steel renewal in affected areas
5. Inform insurance underwriters of findings

REPAIR PRIORITY: CRITICAL (0-7 days)""",
            'severe': """URGENT ACTIONS:
1. Schedule dry dock within 30 days
2. Notify Classification Society of findings
3. Commission detailed corrosion mapping with UT thickness gauging
4. Prepare coating renewal specification for affected areas
5. Consider temporary protective measures (inhibitors, cathodic protection check)

REPAIR PRIORITY: HIGH (7-30 days)""",
            'moderate': """RECOMMENDED ACTIONS:
1. Include in next scheduled maintenance period
2. Document all corrosion locations with measurements for trending
3. Prepare coating touch-up and spot-repair materials
4. Schedule follow-up segmentation scan in 6 months
5. Verify cathodic protection system is functioning correctly

REPAIR PRIORITY: MEDIUM (30-90 days)""",
            'mild': """MAINTENANCE ACTIONS:
1. Continue normal maintenance schedule
2. Address minor corrosion spots during routine upkeep
3. Document current condition as baseline for trend analysis
4. Re-inspect in 12 months or per classification schedule
5. Ensure coating system integrity is maintained

REPAIR PRIORITY: LOW (90+ days / Next Scheduled)""",
            'good': """MAINTENANCE ACTIONS:
1. Continue regular inspection schedule
2. Maintain current coating and cathodic protection systems
3. Document condition for baseline records
4. Schedule next inspection per classification rules

REPAIR PRIORITY: ROUTINE MAINTENANCE""",
        }
        return recommendations.get(severity, recommendations['mild'])
    
    def _convert_to_html(self, report_text: str) -> str:
        """Convert plain text report to HTML."""
        import html as html_module
        
        escaped = html_module.escape(report_text)
        
        lines = escaped.split('\n')
        html_lines = []
        
        for line in lines:
            if line.startswith('=' * 10):
                continue
            
            # Bold markers
            while '**' in line:
                line = line.replace('**', '<strong>', 1)
                line = line.replace('**', '</strong>', 1)
            
            # Bullet points
            if line.strip().startswith('•'):
                line = f'<li>{line.strip()[1:].strip()}</li>'
            elif line.strip().startswith('- '):
                line = f'<li>{line.strip()[2:].strip()}</li>'
            
            # Numbered items
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and '.' in stripped[:3]:
                line = f'<li>{stripped[2:].strip()}</li>'
            
            # Section headers
            section_keywords = [
                'EXPERT VISUAL ANALYSIS', 'AI SEGMENTATION', 'DETAILED REGION',
                'CONDITION ASSESSMENT', 'SAFETY', 'RECOMMENDATIONS',
                'ANALYSIS METHODOLOGY', 'ANALYSIS NOTES',
                'MARITIME HULL INSPECTION',
            ]
            if any(kw in line for kw in section_keywords):
                line = f'<h2 class="report-section">{line.strip()}</h2>'
            
            html_lines.append(line)
        
        content = '\n'.join(html_lines)
        
        return f"""
<div class="inspection-report">
    <pre class="report-content">{content}</pre>
</div>
"""
