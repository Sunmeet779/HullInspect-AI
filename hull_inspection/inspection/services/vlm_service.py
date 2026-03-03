"""
Vision-Language Model (VLM) Service for Maritime Inspection Reports

This service supports multiple backends:
- BLIP (Bootstrapping Language-Image Pre-training) - runs locally
- Ollama with LLaVA - runs locally, free, better quality

Ollama is recommended for students - free, local, no API key needed.
"""
# Disable MPS BEFORE importing torch to avoid device conflicts
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import base64
import requests
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class VLMReportGenerator:
    """
    Service for generating maritime inspection reports using VLM.
    
    Supports multiple backends:
    - 'github': GitHub Models API (GPT-4o-mini) — recommended, free with GitHub account
    - 'ollama': Ollama with LLaVA (local, free)
    - 'blip': BLIP model from HuggingFace (runs locally)
    
    Usage with GitHub Models (recommended):
        1. Get token: https://github.com/settings/tokens
        2. Add to .env: GITHUB_TOKEN=ghp_...
        3. Use: generator = VLMReportGenerator(backend='github')
    
    Usage with Ollama:
        1. Install Ollama: brew install ollama
        2. Pull the model: ollama pull llava
        3. Start Ollama: ollama serve
        4. Use: generator = VLMReportGenerator(backend='ollama')
    """
    
    # Maritime surveyor prompt template
    SYSTEM_PROMPT = """You are an experienced Class A Marine Surveyor with over 20 years of experience 
in vessel hull inspections. You are conducting a detailed visual inspection of a ship's hull 
and must provide a professional maritime inspection report."""
    
    OLLAMA_INSPECTION_PROMPT = """You are an experienced Class A Marine Surveyor inspecting a ship's hull.
Analyze this image and provide:
1. Overall condition description
2. Types of defects visible (corrosion, cracks, paint damage, marine growth, etc.)
3. Severity assessment (critical, severe, moderate, mild, or good)
4. Specific observations about each defect area
5. Recommended actions

Be detailed and professional in your assessment."""

    ANALYSIS_QUESTIONS = [
        "Describe the condition of this ship hull in detail, including any visible damage, corrosion, or defects.",
        "What types of defects or damage can you see on this hull surface?",
        "Rate the overall condition of this hull section: critical, severe, moderate, mild, or good.",
    ]

    def __init__(
        self, 
        model_name: str = "Salesforce/blip-image-captioning-large",
        backend: str = "github",
        ollama_model: str = "llava",
        ollama_url: str = "http://localhost:11434",
        github_token: str = None,
        github_model: str = None,
        github_endpoint: str = None,
    ):
        """
        Initialize the VLM service.
        
        Args:
            model_name: HuggingFace model name for BLIP backend
            backend: 'github' (recommended), 'ollama', or 'blip'
            ollama_model: Ollama model name (default: 'llava')
            ollama_url: Ollama API URL (default: http://localhost:11434)
            github_token: GitHub PAT for Models API (loaded from .env if None)
            github_model: GitHub Models model name (loaded from .env if None)
            github_endpoint: GitHub Models endpoint URL (loaded from .env if None)
        """
        self.model_name = model_name
        self.backend = backend
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.model = None
        self.processor = None
        self._initialized = False
        self.device = "cpu"  # Always use CPU for Mac compatibility
        
        # GitHub Models config (from args or .env)
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN', '')
        self.github_model = github_model or os.environ.get('GITHUB_MODELS_MODEL', 'gpt-4o-mini')
        self.github_endpoint = github_endpoint or os.environ.get('GITHUB_MODELS_ENDPOINT', 'https://models.inference.ai.azure.com')
        
    def _initialize_model(self):
        """Load the BLIP model and processor."""
        if self._initialized:
            return
            
        try:
            import torch
            # Force CPU for all operations
            torch.set_default_tensor_type(torch.FloatTensor)
            
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            print(f"Loading VLM model: {self.model_name}")
            print(f"Using device: {self.device}")
            
            # Load BLIP model - explicitly on CPU
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            ).to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            self._initialized = True
            print(f"VLM model loaded successfully on {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Run: pip install transformers torch"
            )
        except Exception as e:
            print(f"Warning: Could not load VLM model: {e}")
            print("Falling back to template-based report generation")
            self._initialized = False
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                if self.ollama_model.split(':')[0] in model_names:
                    return True
                print(f"Ollama model '{self.ollama_model}' not found. Available: {model_names}")
                print(f"Run: ollama pull {self.ollama_model}")
                return False
            return False
        except requests.exceptions.ConnectionError:
            print("Ollama not running. Start it with: ollama serve")
            return False
        except Exception as e:
            print(f"Error checking Ollama: {e}")
            return False
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 for API calls."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    # ── GitHub Models backend ─────────────────────────────────────────

    def _generate_github_models_report(
        self,
        image_path: str,
        detection_summary: str,
        detection_details: str,
        overall_severity: str
    ) -> str:
        """
        Generate report using GitHub Models API (GPT-4o-mini / GPT-4o).

        Sends the hull image + detection data to the model for analysis.
        Falls back to template if API call fails.
        """
        print(f"   📡 Calling GitHub Models API ({self.github_model})...")

        # Encode image for vision
        image_b64 = self._encode_image_base64(image_path)
        ext = os.path.splitext(image_path)[1].lower().lstrip('.')
        mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'bmp': 'bmp'}.get(ext, 'jpeg')

        # Build the prompt
        system_msg = (
            "You are an experienced Class A Marine Surveyor with over 20 years of "
            "experience in vessel hull inspections. You write professional, detailed "
            "maritime inspection reports. Use formal technical language. Structure your "
            "response clearly with numbered sections."
        )

        user_prompt = f"""Analyze this hull inspection image and the automated detection results below.
Write a professional maritime hull inspection report.

AUTOMATED DETECTION RESULTS:
{detection_summary}

DETAILED DEFECT DATA:
{detection_details}

ASSESSED SEVERITY: {overall_severity.upper()}

Write your report with these sections:
1. VISUAL ANALYSIS - What you observe in the image (surface condition, colour patterns, visible damage)
2. DEFECT ASSESSMENT - Interpret each detected defect, its likely cause, and progression risk
3. STRUCTURAL IMPLICATIONS - What these defects mean for hull integrity
4. RECOMMENDATIONS - Specific repair/maintenance actions with priority timelines

Be specific, professional, and reference the detection data. Keep the total response under 800 words."""

        # Call GitHub Models API (OpenAI-compatible)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.github_token}",
        }

        payload = {
            "model": self.github_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        try:
            response = requests.post(
                f"{self.github_endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                ai_analysis = result["choices"][0]["message"]["content"]
                print(f"   ✅ GitHub Models response received ({len(ai_analysis)} chars)")
            elif response.status_code == 401:
                print("   ❌ GitHub token invalid or expired. Regenerate at github.com/settings/tokens")
                return self._generate_template_report(detection_summary, detection_details, overall_severity)
            elif response.status_code == 429:
                print("   ❌ Rate limit exceeded. Try again later or switch to gpt-4o-mini")
                return self._generate_template_report(detection_summary, detection_details, overall_severity)
            else:
                print(f"   ❌ API error {response.status_code}: {response.text[:200]}")
                return self._generate_template_report(detection_summary, detection_details, overall_severity)

        except requests.exceptions.Timeout:
            print("   ❌ GitHub Models request timed out")
            return self._generate_template_report(detection_summary, detection_details, overall_severity)
        except Exception as e:
            print(f"   ❌ GitHub Models error: {e}")
            return self._generate_template_report(detection_summary, detection_details, overall_severity)

        # Build final report combining AI analysis with structured data
        report = f"""
================================================================================
                    MARITIME HULL INSPECTION REPORT
================================================================================

Document ID: AUTO-GENERATED
Date: {self._get_current_date()}
Inspector: AI-Assisted Analysis System (GitHub Models {self.github_model})
Classification: PRELIMINARY

================================================================================
1. AI VISUAL ANALYSIS ({self.github_model.upper()})
================================================================================

{ai_analysis}

================================================================================
2. AUTOMATED DETECTION SUMMARY
================================================================================

{detection_summary}

================================================================================
3. DETAILED DETECTION DATA
================================================================================

{detection_details}

================================================================================
4. CONDITION ASSESSMENT
================================================================================

Overall Severity Level: **{overall_severity.upper()}**

{self._get_severity_description(overall_severity)}

================================================================================
5. INSPECTION NOTES
================================================================================

• AI Vision Model: {self.github_model} (via GitHub Models API)
• Detection Engine: YOLO / Eigen Tree (automated corrosion segmentation)
• The AI model analysed both the image and detection data
• Final assessment should be verified by qualified marine surveyor

================================================================================
                              END OF REPORT
================================================================================
"""
        return report

    # ── Ollama backend ────────────────────────────────────────────────

    def _generate_ollama_report(
        self,
        image_path: str,
        detection_summary: str,
        detection_details: str,
        overall_severity: str
    ) -> str:
        """Generate report using Ollama with LLaVA or similar vision model."""
        
        # Encode image
        image_base64 = self._encode_image_base64(image_path)
        
        # Build context with YOLO detections
        context = f"""
Based on automated YOLO detection, the following defects were found:
{detection_summary}

Detected severity level: {overall_severity}

Now analyze the image and provide your expert maritime surveyor assessment.
"""
        
        prompt = f"{self.OLLAMA_INSPECTION_PROMPT}\n\nContext from automated detection:\n{context}"
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1500
                    }
                },
                timeout=120  # Vision models can be slow
            )
            
            if response.status_code == 200:
                result = response.json()
                vlm_analysis = result.get('response', 'Analysis not available')
            else:
                print(f"Ollama API error: {response.status_code}")
                vlm_analysis = "VLM analysis unavailable - API error"
                
        except requests.exceptions.Timeout:
            print("Ollama request timed out")
            vlm_analysis = "VLM analysis unavailable - timeout"
        except Exception as e:
            print(f"Ollama error: {e}")
            vlm_analysis = f"VLM analysis unavailable - {e}"
        
        # Build the full report
        report = self._build_ollama_report(
            vlm_analysis,
            detection_summary,
            detection_details,
            overall_severity
        )
        
        return report
    
    def _build_ollama_report(
        self,
        vlm_analysis: str,
        detection_summary: str,
        detection_details: str,
        overall_severity: str
    ) -> str:
        """Build comprehensive report from Ollama VLM analysis."""
        
        report = f"""
================================================================================
                    MARITIME HULL INSPECTION REPORT
================================================================================

Document ID: AUTO-GENERATED
Date: {self._get_current_date()}
Inspector: AI-Assisted Analysis System (Ollama LLaVA + YOLO)
Classification: PRELIMINARY

================================================================================
1. VLM VISUAL ANALYSIS (Ollama {self.ollama_model})
================================================================================

{vlm_analysis}

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
- VLM (Ollama) provides detailed contextual analysis
- YOLO provides precise defect localization with confidence scores
- Combined analysis offers comprehensive assessment
- Final assessment should be verified by qualified marine surveyor

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

• VLM Analysis: Ollama ({self.ollama_model})
• Object Detection: YOLOv8 trained on maritime corrosion data
• Detection confidence levels indicate AI model certainty
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

This report combines two AI analysis methods:
1. Ollama Vision-Language Model ({self.ollama_model}): Provides detailed analysis and context
2. YOLO Object Detection: Identifies and localizes specific defects

The VLM provides rich qualitative analysis while YOLO provides
precise quantitative detection with bounding boxes and confidence scores.

================================================================================
                              END OF REPORT
================================================================================
"""
        return report

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
                # GitHub Models backend (recommended — free, cloud)
                if self.backend == "github":
                    if self.github_token:
                        report_text = self._generate_github_models_report(
                            image_path,
                            detection_summary,
                            detection_details,
                            overall_severity
                        )
                    else:
                        print("GitHub token not set, falling back to template")
                        print("Set GITHUB_TOKEN in .env file")
                        report_text = self._generate_template_report(
                            detection_summary,
                            detection_details,
                            overall_severity
                        )
                # Ollama backend (local)
                elif self.backend == "ollama":
                    if self._check_ollama_available():
                        report_text = self._generate_ollama_report(
                            image_path,
                            detection_summary,
                            detection_details,
                            overall_severity
                        )
                    else:
                        print("Ollama not available, falling back to template")
                        report_text = self._generate_template_report(
                            detection_summary,
                            detection_details,
                            overall_severity
                        )
                # BLIP backend
                elif self.backend == "blip":
                    self._initialize_model()
                    if self._initialized:
                        report_text = self._generate_vlm_report(
                            image_path, 
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
                else:
                    print(f"Unknown backend: {self.backend}, using template")
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
        detection_details: str,
        overall_severity: str
    ) -> str:
        """Generate report using BLIP VLM."""
        import torch
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Generate multiple captions/analyses
        analyses = []
        
        with torch.no_grad():
            for question in self.ANALYSIS_QUESTIONS:
                # Prepare inputs
                prompt = f"Question: {question} Answer:"
                inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
                
                # Generate
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=3,
                    early_stopping=True
                )
                
                # Decode
                answer = self.processor.decode(output[0], skip_special_tokens=True)
                analyses.append(answer)
        
        # Also get a general caption
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        caption_output = self.model.generate(**inputs, max_new_tokens=100)
        general_caption = self.processor.decode(caption_output[0], skip_special_tokens=True)
        
        # Build the report combining VLM analysis with YOLO detections
        report = self._build_combined_report(
            general_caption,
            analyses,
            detection_summary,
            detection_details,
            overall_severity
        )
        
        return report
    
    def _build_combined_report(
        self,
        caption: str,
        analyses: List[str],
        detection_summary: str,
        detection_details: str,
        overall_severity: str
    ) -> str:
        """Build a comprehensive report combining VLM and YOLO results."""
        
        report = f"""
================================================================================
                    MARITIME HULL INSPECTION REPORT
================================================================================

Document ID: AUTO-GENERATED
Date: {self._get_current_date()}
Inspector: AI-Assisted Analysis System (BLIP + YOLO)
Classification: PRELIMINARY

================================================================================
1. VISUAL ANALYSIS (AI Vision Model)
================================================================================

General Image Description:
{caption}

Detailed Hull Condition Analysis:
{analyses[0] if len(analyses) > 0 else 'Analysis not available'}

Defect Types Identified:
{analyses[1] if len(analyses) > 1 else 'Analysis not available'}

Condition Rating Assessment:
{analyses[2] if len(analyses) > 2 else 'Analysis not available'}

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
- AI vision analysis and YOLO detections have been combined
- High-confidence detections (>85%) are likely accurate
- Visual analysis provides contextual understanding
- Final assessment should be verified by qualified marine surveyor

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

• AI Vision Model: BLIP (Salesforce)
• Object Detection: YOLOv8 trained on maritime corrosion data
• Detection confidence levels indicate AI model certainty
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

This report combines two AI analysis methods:
1. BLIP Vision-Language Model: Provides visual description and context
2. YOLO Object Detection: Identifies and localizes specific defects

The combination provides both qualitative analysis (what the image shows)
and quantitative detection (specific defects with locations and confidence).

AI Detection Accuracy Notes:
- Visual analysis may identify conditions not detected by YOLO
- YOLO provides precise localization of trained defect classes
- Human expert review recommended for final assessment

================================================================================
                              END OF REPORT
================================================================================
"""
        return report

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
    # Example usage with Ollama (recommended - free, local)
    # First install: brew install ollama
    # Then pull model: ollama pull llava
    # Start server: ollama serve
    
    # Use Ollama backend (default)
    generator = VLMReportGenerator(backend='ollama')
    
    # Or use BLIP backend
    # generator = VLMReportGenerator(backend='blip')
    
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
        use_vlm=True  # Use VLM (Ollama)
    )
    
    print(report)
    print(f"\nOverall Severity: {severity}")
