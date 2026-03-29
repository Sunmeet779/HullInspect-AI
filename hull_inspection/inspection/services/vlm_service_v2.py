"""
Vision-Language Model (VLM) Service for Maritime Inspection Reports — v2

3-stage pipeline with automatic fallback:

  Stage 1 — GitHub Models API (GPT-4o-mini) [PRIMARY]
    - Best report quality, vision-capable LLM
    - Free with GitHub account (rate-limited)
    - Requires: GITHUB_TOKEN in .env

  Stage 2 — OpenRouter (Gemini Flash 3.1 Lite Preview) [FALLBACK]
    - Unified API gateway, no regional restrictions
    - OpenAI-compatible format, strong vision model
    - Requires: OPENROUTER_API_KEY in .env

  Stage 3 — Template-based report [ALWAYS AVAILABLE]
    - Structured report from detection data only
    - No VLM narrative, but always works

Usage:
    1. Add GITHUB_TOKEN and OPENROUTER_API_KEY to .env
    2. Run: python manage.py runserver
    3. Upload hull image → report is generated automatically
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
from dotenv import load_dotenv

# Load .env for GitHub Models API credentials
load_dotenv()


class VLMReportGenerator:
    """
    VLM service for generating maritime inspection reports.
    
    3-stage pipeline with automatic fallback:
      1. GitHub Models API (GPT-4o-mini) — primary, best quality
      2. OpenRouter (Gemini Flash 3.1 Lite) — fallback, no regional restrictions
      3. Template-based report — always works, no VLM narrative
    
    Uses segmentation detection results (area-based severity).
    """
    
    # Ranked list of models to try (best → lightest)
    PREFERRED_MODELS = ['minicpm-v', 'llava', 'llava:13b', 'moondream', 'bakllava']
    
    # Enhanced maritime surveyor prompt — holistic full-image analysis
    INSPECTION_PROMPT = """You are an experienced Class A Marine Surveyor with 20+ years of hull inspection experience, certified by a major Classification Society (Lloyd's Register / DNV / Bureau Veritas). You are examining a ship hull photograph.

CRITICAL INSTRUCTION — INDEPENDENT FULL-IMAGE ANALYSIS:
You MUST first examine the ENTIRE image independently, using your own expert judgement, BEFORE considering the {detection_method} detection data provided below. The {detection_method} detection model is trained ONLY on corrosion and frequently produces FALSE POSITIVES — it may misidentify water, shadows, reflections, wet surfaces, marine growth, paint colour variations, or mechanical damage as "corrosion." You must critically evaluate every {detection_method}-detected region and state whether you agree or disagree with the detection.

Analyse the complete hull image and provide a professional assessment covering ALL of the following:

1. **Hull Surface & Location Context**
   - Identify the hull zone visible (flat bottom, bilge turn, boot-top, topside, waterline area, rudder/stern area)
   - Describe the overall surface condition and which side of the vessel this likely represents
   - Note the approximate extent of the area captured in the image

2. **Independent Visual Assessment** (YOUR OWN OBSERVATIONS — look at the WHOLE image)
   - Describe EVERYTHING you see in the image: surface condition, colours, textures, damage patterns
   - Identify ALL types of damage or anomalies present — not just corrosion
   - Specifically look for: mechanical damage (dents, gouges, impact marks, abrasion), structural deformation, weld issues, coating failures, fouling, AND corrosion
   - If you see water, wet surfaces, reflections, or non-damage features — say so explicitly
   - State clearly what the PRIMARY issue in this image is (corrosion, mechanical damage, coating failure, or other)

3. **{detection_method} Detection Cross-Verification** (CRITICAL — evaluate each detection)
   - For each {detection_method}-detected region, state whether you AGREE or DISAGREE that it shows corrosion
   - If you DISAGREE: explain what the region actually shows (water, shadow, dent, paint, etc.)
   - If detection confidence is below 60%, flag it as potentially unreliable
   - Note any damage the {detection_method} model MISSED that you can see in the image
   - Provide your own corrected assessment of actual corrosion extent (if different from {detection_method})

4. **Corrosion Analysis** (ONLY if you confirm actual corrosion is present)
   - Classify each corrosion type observed: uniform/general wastage, pitting, crevice, galvanic, weld-line/HAZ corrosion, waterline attack, groove/erosion-corrosion
   - Estimate corrosion depth (surface rust only, moderate wastage, or deep/through-wall risk)
   - Describe the corrosion pattern and distribution
   - Assess whether corrosion appears active (fresh red/orange oxide) or stabilised (dark, layered scale)

5. **Coating & Paint System Assessment**
   - Identify the coating system layers where visible (anti-fouling, anti-corrosive primer, tie-coat, shop primer)
   - Classify the paint failure mode: blistering, chalking, flaking, peeling, delamination, cracking, under-film corrosion
   - Rate coating breakdown using DIN EN ISO 4628 scale if possible (Ri 0-5 for rust per Part 3, Re 0-5 for blistering per Part 2)
   - Note any colour changes suggesting coating chemical breakdown or UV degradation

6. **Mechanical Damage & Other Defects** (report if visible, explicitly state "not observed" if absent)
   - Mechanical damage: dents, gouges, impact marks, abrasion — estimate depth and extent
   - Weld defects: undercut, porosity, misalignment, cracks along weld seams
   - Structural deformation: buckling, distortion, unfair plating
   - Marine growth / fouling: type (barnacles, algae, slime), extent, location
   - Cathodic protection: condition of visible anodes (wasted, missing, painted over)

7. **Plate Thickness & Structural Concern**
   - Based on visible damage, estimate whether plate diminution likely exceeds 20% of original thickness
   - Identify specific areas where ultrasonic thickness (UT) measurements are recommended
   - Assess risk to structural integrity (frames, longitudinals, shell plating)
   - Note if any areas appear close to Class minimum thickness allowance

8. **Progression Risk & Root Cause**
   - Assess whether the deterioration is accelerating, stable, or recently initiated
   - Identify likely root causes (coating breakdown, cathodic protection failure, galvanic couple, mechanical impact, operational environment)
   - Estimate time since last maintenance based on visible coating/surface state

9. **Recommended Actions** (with priority and timeline)
   - Immediate safety actions (if any)
   - Areas requiring UT thickness gauging — specify exact locations
   - Coating repair specification (surface prep standard, primer, intermediate, topcoat)
   - Whether dry-docking is required or if underwater repair is feasible
   - Classification Society notification requirements (if applicable per IACS UR Z10.1)
   - Follow-up inspection schedule

IMPORTANT INSTRUCTIONS:
- LOOK AT THE ENTIRE IMAGE — do NOT focus only on {detection_method}-highlighted regions
- Be specific about WHERE in the image you see each issue (top-left, center, along weld line, etc.)
- Use proper maritime and coating terminology throughout
- If {detection_method} detection confidence is low (<60%), state clearly that the detection is unreliable
- Distinguish between confirmed visual findings and what requires physical inspection
- Do NOT blindly trust the {detection_method} detections — they are preliminary and error-prone
- Do NOT repeat the detection data back verbatim — provide YOUR expert interpretation"""

    SEVERITY_CONTEXT_PROMPT = """
--- PRELIMINARY {{detection_method}} DETECTION DATA (requires your cross-verification) ---
NOTE: The {{detection_method}} model is trained ONLY on corrosion and may produce false positives.
Water, shadows, wet surfaces, paint variations, and mechanical damage are
commonly misidentified as corrosion.

- Total regions flagged by {{detection_method}}: {num_regions}
- Total flagged area: {total_area:.1f}% of inspected surface
- Largest single flagged region: {largest_area:.1f}%
- {{detection_method}}-assessed severity (UNVERIFIED): {severity}

{{detection_method}} severity breakdown (UNVERIFIED):
{severity_breakdown}

YOUR TASK: Look at the COMPLETE image independently. Determine what is actually
present. Agree or disagree with {{detection_method}} detections. Provide YOUR corrected assessment."""

    def __init__(
        self,
        ollama_model: Optional[str] = None,  # Legacy — kept for Ollama fallback
        ollama_url: str = "http://localhost:11434",
        github_token: Optional[str] = None,
        github_model: Optional[str] = None,
        github_endpoint: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = 'google/gemini-3.1-flash-lite-preview',
        openrouter_endpoint: str = 'https://openrouter.ai/api/v1',
        temperature: float = 0.3,
        max_tokens: int = 3000,
    ):
        """
        Initialize the VLM service.
        
        Args:
            ollama_model: Ollama model name (legacy fallback)
            ollama_url: Ollama API URL
            github_token: GitHub PAT for Models API (loaded from .env if None)
            github_model: GitHub Models model name (loaded from .env if None)
            github_endpoint: GitHub Models endpoint URL (loaded from .env if None)
            openrouter_api_key: OpenRouter API key (loaded from .env if None)
            openrouter_model: OpenRouter model ID (default: google/gemini-3.1-flash-lite-preview)
            openrouter_endpoint: OpenRouter API base URL
            temperature: Generation temperature (0.3 = focused/professional)
            max_tokens: Maximum generation tokens
        """
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = False
        
        # GitHub Models config (from args or .env)
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN', '')
        self.github_model = github_model or os.environ.get(
            'GITHUB_MODELS_MODEL', 'gpt-4o-mini'
        )
        self.github_endpoint = github_endpoint or os.environ.get(
            'GITHUB_MODELS_ENDPOINT', 'https://models.inference.ai.azure.com'
        )
        
        # OpenRouter config (from args or .env)
        self.openrouter_api_key = openrouter_api_key or os.environ.get('OPENROUTER_API_KEY', '')
        self.openrouter_model = openrouter_model
        self.openrouter_endpoint = openrouter_endpoint
        
        # Track which backend generated the last report
        self._model_used = 'template'
        self._detection_method = 'yolo_seg'
        self._detection_method_name = self.DETECTION_METHOD_NAMES.get('yolo_seg', 'yolo_seg')
        
        # Auto-detect best Ollama model if not specified
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
    
    def _call_github_models(
        self,
        image_path: str,
        detections: List[Dict],
        overall_severity: str,
    ) -> Optional[str]:
        """
        Call GitHub Models API (GPT-4o-mini) with the image and detection context.
        
        Primary VLM backend — free with GitHub account, excellent vision quality.
        Falls back gracefully on any error.
        """
        print(f"📡 Calling GitHub Models API ({self.github_model})...")
        
        # Encode image for the vision API
        image_base64 = self._encode_image_base64(image_path)
        ext = os.path.splitext(image_path)[1].lower().lstrip('.')
        mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'bmp': 'bmp'}.get(ext, 'jpeg')
        
        # Build severity context from detections
        total_area = sum(d.get('area_percent', 0) for d in detections)
        largest = max((d.get('area_percent', 0) for d in detections), default=0)
        
        sev_counts = {}
        for d in detections:
            s = d.get('severity', 'unknown')
            if s not in sev_counts:
                sev_counts[s] = {'count': 0, 'area': 0}
            sev_counts[s]['count'] += 1
            sev_counts[s]['area'] += d.get('area_percent', 0)
        
        breakdown = "\n".join(
            f"  - {s.upper()}: {v['count']} region(s), {v['area']:.1f}% area"
            for s, v in sev_counts.items()
        ) or "  No detections"
        
        # Build the prompt
        system_msg = (
            "You are an experienced Class A Marine Surveyor certified by a major "
            "Classification Society (Lloyd's Register / DNV / Bureau Veritas). You have "
            "20+ years conducting hull condition surveys, dry-dock inspections, and "
            "damage assessments. You write professional reports that follow IACS "
            "Procedural Requirements and use DIN EN ISO 4628 coating condition standards. "
            "Use formal maritime technical language. Structure your response with "
            "numbered sections. Distinguish clearly between confirmed visual findings "
            "and areas requiring physical verification (UT gauging, close-up, etc.).\n\n"
            "CRITICAL: You must independently analyse the ENTIRE image using your own "
            f"expert judgement. The {self._detection_method_name} detection model provided below is "
            "preliminary and frequently produces FALSE POSITIVES — it can misidentify "
            "water, shadows, wet surfaces, reflections, or mechanical damage as "
            f"'corrosion'. You must cross-verify every {self._detection_method_name} detection and state whether "
            "you agree or disagree. Identify ALL types of damage visible in the image, "
            "not just corrosion."
        )
        
        formatted_prompt = self.INSPECTION_PROMPT.format(detection_method=self._detection_method_name)
        
        user_prompt = f"""{formatted_prompt}

--- PRELIMINARY {self._detection_method_name.upper()} DETECTION DATA (requires your cross-verification) ---
NOTE: The {self._detection_method_name} model below is trained ONLY on corrosion detection
and may produce false positives. Water, shadows, wet surfaces, paint variations, and
mechanical damage are commonly misidentified as corrosion. Use your expert judgement.

- Total regions flagged by {self._detection_method_name}: {len(detections)}
- Total flagged area: {total_area:.1f}% of inspected surface
- Largest single flagged region: {largest:.1f}%
- {self._detection_method_name}-assessed severity (UNVERIFIED): {overall_severity.upper()}

{self._detection_method_name} severity breakdown (UNVERIFIED):
{breakdown}

YOUR TASK: Look at the COMPLETE image. Determine what is actually present —
corrosion, mechanical damage, water, paint issues, or other. Agree or disagree
with each {self._detection_method_name} detection. Provide YOUR corrected assessment.
If detection confidence is below 60%, flag the detection as unreliable.
Keep the total response under 1200 words."""

        # Call GitHub Models API (OpenAI-compatible endpoint)
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
                                "url": f"data:image/{mime};base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
                analysis = result["choices"][0]["message"]["content"].strip()
                if analysis:
                    print(f"✅ GitHub Models response received ({len(analysis)} chars)")
                    return analysis
                print("⚠️ GitHub Models returned empty response")
            elif response.status_code == 401:
                print("❌ GitHub token invalid or expired. Regenerate at github.com/settings/tokens")
            elif response.status_code == 429:
                print("❌ Rate limit exceeded. Falling back to next stage.")
            else:
                print(f"❌ GitHub Models API error {response.status_code}: {response.text[:200]}")
        
        except requests.exceptions.Timeout:
            print("⚠️ GitHub Models request timed out (60s)")
        except Exception as e:
            print(f"⚠️ GitHub Models error: {e}")
        
        return None
    
    def _call_openrouter(
        self,
        image_path: str,
        detections: List[Dict],
        overall_severity: str,
    ) -> Optional[str]:
        """
        Call OpenRouter API with the image and detection context.
        
        Stage 2 fallback — OpenAI-compatible gateway, no regional restrictions.
        Uses Gemini Flash 3.1 Lite Preview via OpenRouter.
        """
        print(f"📡 Calling OpenRouter ({self.openrouter_model})...")
        
        # Encode image to base64
        image_base64 = self._encode_image_base64(image_path)
        if not image_base64:
            print("⚠️ Failed to encode image for OpenRouter")
            return None
        
        # Build severity context from detections
        total_area = sum(d.get('area_percent', 0) for d in detections)
        largest = max((d.get('area_percent', 0) for d in detections), default=0)
        
        sev_counts = {}
        for d in detections:
            s = d.get('severity', 'unknown')
            if s not in sev_counts:
                sev_counts[s] = {'count': 0, 'area': 0}
            sev_counts[s]['count'] += 1
            sev_counts[s]['area'] += d.get('area_percent', 0)
        
        breakdown = "\n".join(
            f"  - {s.upper()}: {v['count']} region(s), {v['area']:.1f}% area"
            for s, v in sev_counts.items()
        ) or "  No detections"
        
        # Build the prompt
        formatted_prompt = self.INSPECTION_PROMPT.format(detection_method=self._detection_method_name)
        
        severity_context = self.SEVERITY_CONTEXT_PROMPT.replace(
            '{{detection_method}}', self._detection_method_name
        ).format(
            num_regions=len(detections),
            total_area=total_area,
            largest_area=largest,
            severity=overall_severity.upper(),
            severity_breakdown=breakdown,
        )
        
        full_user_prompt = f"{formatted_prompt}\n\n{severity_context}"
        
        # System message
        system_msg = (
            "You are an experienced Class A Marine Surveyor certified by a major "
            "Classification Society (Lloyd's Register / DNV / Bureau Veritas). You have "
            "20+ years conducting hull condition surveys, dry-dock inspections, and "
            "damage assessments. You write professional reports that follow IACS "
            "Procedural Requirements and use DIN EN ISO 4628 coating condition standards. "
            "Use formal maritime technical language. Structure your response with "
            "numbered sections. Distinguish clearly between confirmed visual findings "
            "and areas requiring physical verification (UT gauging, close-up, etc.).\n\n"
            "CRITICAL: You must independently analyse the ENTIRE image using your own "
            f"expert judgement. The {self._detection_method_name} detection model provided below is "
            "preliminary and frequently produces FALSE POSITIVES — it can misidentify "
            "water, shadows, wet surfaces, reflections, or mechanical damage as "
            f"'corrosion'. You must cross-verify every {self._detection_method_name} detection and state whether "
            "you agree or disagree. Identify ALL types of damage visible in the image, "
            "not just corrosion."
        )
        
        # OpenAI-compatible payload
        payload = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": full_user_prompt},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8090",
            "X-Title": "Hull Inspection System",
        }
        
        try:
            response = requests.post(
                f"{self.openrouter_endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis = data['choices'][0]['message']['content'].strip()
                if analysis:
                    print(f"✅ OpenRouter response received ({len(analysis)} chars)")
                    return analysis
                print("⚠️ OpenRouter returned empty content")
            else:
                print(f"⚠️ OpenRouter HTTP {response.status_code}: {response.text[:200]}")
        
        except requests.exceptions.Timeout:
            print("⚠️ OpenRouter request timed out (90s)")
        except Exception as e:
            print(f"⚠️ OpenRouter error: {e}")
        
        return None
    
    # Human-readable detection method names
    DETECTION_METHOD_NAMES = {
        'yolo_seg': 'YOLO11-Seg (Instance Segmentation)',
        'eigen_tree': 'Pruned Eigen-Tree (Unsupervised Anomaly Detection)',
    }

    def generate_report(
        self,
        image_path: str,
        detections: List[Dict],
        use_vlm: bool = True,
        detection_method: str = 'yolo_seg',
        vlm_model: str = 'gpt4o_mini',
    ) -> Tuple[str, str, str]:
        """
        Generate a maritime inspection report.
        
        Args:
            image_path: Path to the hull image
            detections: Detection results from YOLO or EigenTree
            use_vlm: Whether to use VLM (falls back to template if False/unavailable)
            detection_method: 'yolo_seg' or 'eigen_tree'
            vlm_model: 'gpt4o_mini' (GitHub Models) or 'openrouter' (OpenRouter)
            
        Returns:
            Tuple of (report_text, report_html, overall_severity)
        """
        # Store the detection method name for use in report sections
        self._detection_method = detection_method
        self._detection_method_name = self.DETECTION_METHOD_NAMES.get(
            detection_method, detection_method
        )
        
        # Build detection context
        summary = self._build_detection_summary(detections)
        details = self._format_detection_details(detections)
        overall_severity = self._determine_overall_severity(detections)
        
        vlm_analysis = None
        
        if use_vlm:
            if vlm_model == 'openrouter':
                # User chose OpenRouter as primary
                if self.openrouter_api_key:
                    vlm_analysis = self._call_openrouter(
                        image_path, detections, overall_severity
                    )
                    if vlm_analysis:
                        self._model_used = f'OpenRouter {self.openrouter_model}'
                
                # Fallback to GPT-4o-mini if OpenRouter fails
                if vlm_analysis is None and self.github_token:
                    vlm_analysis = self._call_github_models(
                        image_path, detections, overall_severity
                    )
                    if vlm_analysis:
                        self._model_used = f'GitHub Models {self.github_model}'
            else:
                # Default: GPT-4o-mini as primary
                if self.github_token:
                    vlm_analysis = self._call_github_models(
                        image_path, detections, overall_severity
                    )
                    if vlm_analysis:
                        self._model_used = f'GitHub Models {self.github_model}'
                
                # Fallback to OpenRouter if GPT-4o-mini fails
                if vlm_analysis is None and self.openrouter_api_key:
                    vlm_analysis = self._call_openrouter(
                        image_path, detections, overall_severity
                    )
                    if vlm_analysis:
                        self._model_used = f'OpenRouter {self.openrouter_model}'
            
            # Last resort: Ollama local vision model
            if vlm_analysis is None:
                if self._check_ollama_available():
                    vlm_analysis = self._call_ollama(
                        image_path, detections, overall_severity
                    )
                    if vlm_analysis:
                        self._model_used = f'Ollama {self.ollama_model}'
        
        # Stage 3: Template fallback — always works
        if vlm_analysis:
            report_text = self._build_full_report(
                vlm_analysis, summary, details, overall_severity
            )
        else:
            self._model_used = 'template'
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
        ).replace('{detection_method}', self._detection_method_name)
        
        formatted_prompt = self.INSPECTION_PROMPT.format(detection_method=self._detection_method_name)
        prompt = f"{formatted_prompt}\n\n{context}"
        
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
        method_name = getattr(self, '_detection_method_name', 'AI Detection Model')
        if not detections:
            return f"No regions flagged by {method_name}."
        
        total_area = sum(d['area_percent'] for d in detections)
        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
        
        lines = [
            f"Detection Method: {method_name}",
            f"Total Flagged Regions: {len(detections)}",
            f"Total Flagged Area: {total_area:.2f}% of inspected surface",
            f"Average Detection Confidence: {avg_conf:.1%}",
            "",
            f"NOTE: {method_name} is trained primarily on corrosion patterns.",
            "Regions flagged below may include false positives (water, shadows, mechanical",
            "damage, wet surfaces). See the Expert Visual Analysis for cross-verification.",
            "",
            "Severity Distribution (based on flagged area — requires verification):",
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
        method_name = getattr(self, '_detection_method_name', 'AI Detection Model')
        if not detections:
            return f"No regions flagged by {method_name}."
        
        details = []
        for i, det in enumerate(detections, 1):
            conf = det['confidence']
            reliability = "LOW — may be false positive" if conf < 0.6 else "acceptable"
            detail = f"""
Region #{i}:
  Confidence: {conf:.1%} ({reliability})
  AI-Assigned Severity: {det['severity'].upper()}
  Flagged Area: {det['area_percent']:.2f}% of image ({det.get('mask_pixels', 'N/A')} pixels)
  Location: Center ({det['x_center']:.2f}, {det['y_center']:.2f})
  Bounding Box: {det['width']:.1%} x {det['height']:.1%} of image
  ⚠ Cross-verify with Expert Visual Analysis above"""
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
Inspector: AI-Assisted Analysis System ({self._detection_method_name} + {self._model_used})
Method: {self._detection_method_name} + Independent VLM Visual Analysis
Classification: PRELIMINARY

================================================================================
1. VLM ANALYSIS ({self._model_used})
================================================================================
(Independent full-image assessment — cross-verifies {self._detection_method_name} detections)

{vlm_analysis}

================================================================================
2. {self._detection_method_name.upper()} RESULTS (Preliminary — Requires Cross-Verification)
================================================================================

{detection_summary}

================================================================================
3. FLAGGED REGION DETAILS
================================================================================

{detection_details}

================================================================================
4. CONDITION ASSESSMENT
================================================================================

AI-Assessed Severity Level: **{overall_severity.upper()}**
(Refer to Section 1 for expert-verified assessment — AI severity may differ)

{self._get_severity_description(overall_severity)}

Methodology Notes:
- Detection model ({self._detection_method_name}) is trained primarily on corrosion patterns
- Flagged areas may include false positives (water, shadows, mechanical damage, wet surfaces)
- Area percentages represent model-flagged pixels, not necessarily confirmed corrosion
- The VLM Analysis (Section 1) provides independent cross-verification
- Combined assessment should be verified by qualified marine surveyor

================================================================================
5. SAFETY & ENVIRONMENTAL CONTEXT
================================================================================

Environmental Factors:
- Corrosion progression depends on operational environment (saltwater, temperature)
- Marine growth presence indicates extended waterline exposure
- Paint/coating damage accelerates underlying corrosion development
- Pitting corrosion poses higher structural risk than uniform surface corrosion
- Mechanical damage (dents, gouges) may compromise coating integrity

Safety Considerations:
- Areas with confirmed corrosion exceeding 5% warrant structural assessment
- Pitting corrosion may be deeper than surface appearance suggests
- Mechanical damage sites should be inspected for coating breach and incipient corrosion
- Proper ventilation and safety equipment required during any repair work

================================================================================
6. RECOMMENDATIONS
================================================================================

{self._get_recommendations(overall_severity)}

================================================================================
7. ANALYSIS METHODOLOGY
================================================================================

Detection Model: {self._detection_method_name}
- {'Trained on MaVeCoDD maritime corrosion dataset' if self._detection_method == 'yolo_seg' else 'Unsupervised anomaly detection using eigenvalue decomposition'}
- {'Pixel-accurate instance segmentation with mask-based area measurement' if self._detection_method == 'yolo_seg' else 'Colour-space anomaly detection with decision-tree pruning'}
- May produce false positives for water, shadows, wet surfaces, reflections

Vision-Language Model: {self._model_used}
- Independent full-image analysis (not limited to AI-flagged regions)
- Cross-verifies AI detections — identifies false positives
- Detects damage types the segmentation model cannot (mechanical, structural, fouling)
- Professional maritime inspection narrative

Cross-Verification Approach:
- VLM independently analyses the complete image before considering AI data
- Each AI-flagged region is verified: confirmed, disputed, or reclassified
- Damage types missed by the segmentation model are identified by VLM
- Final assessment integrates both AI quantitative data and expert visual analysis

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
Inspector: AI-Assisted Analysis System ({self._detection_method_name})
Method: {self._detection_method_name}
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

- Detection uses {self._detection_method_name} for area measurement
- {'Corrosion boundaries are pixel-accurate (not bounding-box approximations)' if self._detection_method == 'yolo_seg' else 'Anomaly regions identified via eigenvalue-based colour space analysis'}
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
        """Convert report text to clean HTML.
        
        The VLM (GPT-4o-mini) returns markdown in the analysis section.
        We split by ===== dividers (which separate our report sections),
        then render each section's content through python-markdown.
        """
        import html as html_module
        import re
        
        try:
            import markdown
            has_markdown = True
        except ImportError:
            has_markdown = False
        
        def render(text):
            """Render a block of text to HTML."""
            text = text.strip()
            if not text:
                return ''
            if has_markdown:
                # Ensure blank line before markdown headings (# ## ### ####)
                # GPT-4o-mini sometimes omits them, which breaks parsing
                text = re.sub(r'(?<!\n\n)(^|\n)(#{1,6} )', r'\1\n\2', text)
                # Ensure blank line before list items that follow a paragraph
                text = re.sub(r'(?<!\n\n)(^|\n)(- \*\*)', r'\1\n\2', text)
                return markdown.markdown(text, extensions=['tables', 'sane_lists'])
            # Fallback: basic conversion
            escaped = html_module.escape(text)
            # Handle **bold**
            escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
            return '<p>' + escaped.replace('\n\n', '</p><p>').replace('\n', '<br>') + '</p>'
        
        # ── Split report by ===== divider lines into raw blocks ──
        # A ===== line is anything with 10+ consecutive '='
        blocks = re.split(r'\n *={10,} *\n', report_text)
        
        # Each pair of ===== lines produces: [preamble, title, content, title, content, ...]
        # But since the report starts with ====, the first block is usually empty.
        # We'll scan blocks and classify them as title-blocks vs content-blocks.
        
        sections = []  # list of (title, content_html)
        
        # Report structure: ===title=== content ===title=== content ...
        # After splitting by ====, short blocks (single line, ALL-CAPS) are titles;
        # longer blocks are content.
        
        pending_title = None
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Is this a section title? (1-2 lines, mostly uppercase, < 120 chars)
            lines = block.split('\n')
            is_title = (
                len(lines) <= 2
                and len(block) < 120
                and sum(1 for c in block if c.isupper()) > len(block.replace(' ', '')) * 0.4
            )
            
            if is_title:
                # Save any previous pending title with no content
                if pending_title is not None:
                    sections.append((pending_title, ''))
                pending_title = block
            else:
                title = pending_title or ''
                sections.append((title, block))
                pending_title = None
        
        if pending_title is not None:
            sections.append((pending_title, ''))
        
        # ── Build HTML from sections ──
        html_parts = []
        
        for title, content in sections:
            title_upper = title.upper()
            content_html = render(content)
            
            if 'MARITIME HULL INSPECTION REPORT' in title_upper:
                html_parts.append(
                    '<div class="text-center border-b-2 border-maritime-200 pb-4 mb-6">'
                    f'<h1 class="text-xl font-bold text-maritime-800 tracking-wide">{html_module.escape(title)}</h1>'
                    f'<div class="mt-3 text-sm text-gray-600 leading-relaxed">{content_html}</div></div>'
                )
            elif 'END OF REPORT' in title_upper:
                html_parts.append(
                    '<div class="text-center border-t-2 border-maritime-200 pt-4 mt-6">'
                    f'<p class="text-xs text-gray-400 uppercase tracking-widest">{html_module.escape(title)}</p></div>'
                )
            elif title:
                html_parts.append(
                    '<div class="report-section mb-6">'
                    f'<h2 class="text-base font-bold text-maritime-700 border-b border-gray-200 pb-2 mb-3 uppercase tracking-wide">{html_module.escape(title)}</h2>'
                    f'<div class="text-sm text-gray-800 leading-relaxed prose prose-sm max-w-none">{content_html}</div>'
                    '</div>'
                )
            elif content_html:
                html_parts.append(
                    f'<div class="text-sm text-gray-700 leading-relaxed mb-4 prose prose-sm max-w-none">{content_html}</div>'
                )
        
        return f'<div class="inspection-report space-y-2">{"".join(html_parts)}</div>'
