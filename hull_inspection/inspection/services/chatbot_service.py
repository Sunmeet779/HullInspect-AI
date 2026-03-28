"""
Chatbot Service for Maritime Hull Inspection
=============================================
RAG-style chatbot that retrieves past inspection data from the DB
and uses a local LLM (Ollama Phi-3 Mini) to answer surveyor questions.

No training required — the LLM is pre-trained and receives inspection
context via prompt engineering.
"""

import json
import re
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class InspectionChatbot:
    """
    Conversational assistant for maritime surveyors.

    Retrieves inspection reports from the database and uses Ollama
    (Phi-3 Mini) to generate informed answers about vessel conditions,
    defect trends, and maintenance recommendations.
    """

    # UUID regex to detect when user asks about a specific report
    _UUID_RE = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )
    # Also match short IDs (first 8 chars)
    _SHORT_ID_RE = re.compile(r"\b[0-9a-f]{8}\b", re.IGNORECASE)

    SYSTEM_PROMPT = """You are HullInspect Assistant — a knowledgeable maritime inspection chatbot.
You help surveyors understand hull inspection results, compare vessel conditions, and provide maintenance guidance.

IMPORTANT — DEFECT CATEGORIES:
The inspection system assesses ALL of the following — NOT only corrosion:
1. Corrosion (pitting, uniform, galvanic, crevice) — rated by severity
2. Mechanical Damage — dents, gouges, scratches, impact marks
3. Coating / Paint Condition — peeling, blistering, chalking, delamination
4. Weld Integrity — cracks, undercuts, porosity, misalignment
5. Marine Fouling — barnacles, algae, biological growth
6. Structural Deformation — buckling, warping, misalignment
7. Water / Environmental Damage — waterline effects, tidal erosion

The detection model (YOLO11-Seg or Pruned Eigen-Tree) is trained primarily on corrosion,
so it may flag non-corrosion defects as "corrosion". Each inspection report includes a VLM
cross-verification section (Section 1) that independently re-evaluates every flagged region.
Always refer to the VLM analysis for the corrected assessment when answering about defect types.

IMPORTANT — REPORT IDs vs VESSEL NAMES:
- Each inspection has a unique Report ID (UUID like 27b32422-8f59-...).
- Some inspections have a vessel name, many do not (shown as "Unknown" or empty).
- Report IDs are NOT vessel IDs. Do NOT call them "Vessel ID".
- When listing or comparing inspections, use "Report ID" or "Inspection" — never "Vessel ID".
- Multiple inspections with no vessel name are separate inspections of different (unknown) vessels,
  NOT multiple inspections of the same vessel.

RULES:
- Answer based ONLY on the inspection data provided in the context below.
- If a FOCUSED REPORT is provided, answer primarily from that full report.
- If no relevant data exists for a question, say so clearly.
- Use professional maritime terminology.
- Be concise but thorough. Use bullet points for lists.
- When comparing inspections, present data in a clear structured format.
- Include specific numbers (detection counts, severity, area percentages) when available.
- When discussing defects, note whether the VLM cross-verification confirmed or overrode the detection label.
- Do NOT assume every flagged region is corrosion — check the VLM analysis for the true finding.
- Quote specific sections of the report when relevant.
- Only report information that is actually present in the data. If a field is missing, say "not available" — do NOT guess or hallucinate values.

{context}
"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "phi3:mini",
        max_context_inspections: int = 15,
    ):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.max_context_inspections = max_context_inspections

    def check_available(self) -> bool:
        """Check if Ollama is running and the model is pulled."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if r.status_code == 200:
                names = [m["name"].split(":")[0] for m in r.json().get("models", [])]
                return self.ollama_model.split(":")[0] in names
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # ID extraction helpers
    # ------------------------------------------------------------------

    def _extract_report_ids(self, text: str) -> List[str]:
        """
        Extract inspection IDs from user text.
        Supports full UUIDs and short (8-char) prefixes.
        Returns a list of full UUID strings matched from the DB.
        """
        from inspection.models import InspectionReport

        ids: List[str] = []

        # Full UUIDs
        for match in self._UUID_RE.finditer(text):
            ids.append(match.group(0))

        # Short IDs — look up in DB
        if not ids:
            for match in self._SHORT_ID_RE.finditer(text):
                short = match.group(0).lower()
                found = InspectionReport.objects.filter(
                    id__startswith=short, status="completed"
                ).values_list("id", flat=True)[:1]
                for fid in found:
                    ids.append(str(fid))

        return ids

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def get_inspection_context(self, focused_ids: Optional[List[str]] = None) -> str:
        """
        Build a text summary of all completed inspections for the LLM context.

        If *focused_ids* is provided, the FULL report content of those inspections
        is included under a "FOCUSED REPORT" header, and only a condensed one-line
        inventory is given for other inspections. This prevents hallucination when
        the user asks about a specific report.
        """
        from inspection.models import InspectionReport

        # When focused, we need fewer other inspections (inventory only)
        limit = self.max_context_inspections if not focused_ids else 50
        inspections = InspectionReport.objects.filter(status="completed").order_by(
            "-created_at"
        )[:limit]

        if not inspections:
            return "No inspections have been completed yet."

        focused_set = set(str(fid).lower() for fid in (focused_ids or []))
        lines: List[str] = []

        # ── Focused report(s) — full content ──────────────────────────
        if focused_set:
            for insp in inspections:
                if str(insp.id).lower() in focused_set:
                    lines.append("=" * 72)
                    lines.append("FOCUSED REPORT  (answer primarily from this)")
                    lines.append("=" * 72)
                    lines.append(f"Inspection ID: {insp.id}")
                    lines.append(f"Vessel: {insp.vessel_name or 'Unknown'}")
                    lines.append(f"Date: {insp.created_at.strftime('%Y-%m-%d %H:%M')}")
                    lines.append(f"Method: {insp.get_detection_method_display()}")
                    lines.append(f"Overall Severity: {(insp.overall_severity or 'N/A').upper()}")
                    lines.append(f"Defects found: {insp.detection_count}")
                    lines.append("")
                    # Full report — include everything
                    if insp.report_content:
                        lines.append(insp.report_content)
                    lines.append("=" * 72)
                    lines.append("")

        # ── Inventory of all inspections ──────────────────────────────
        # List each inspection individually (not grouped by vessel, since
        # most have no vessel name and grouping causes confusion).
        lines.append(f"INSPECTION INVENTORY ({inspections.count()} completed):\n")

        for insp in inspections:
            is_focused = str(insp.id).lower() in focused_set
            vessel_label = insp.vessel_name if insp.vessel_name else "(no vessel name)"

            # Detection area summary
            area_str = ""
            sev_str = ""
            if insp.yolo_detections:
                severity_counts: Dict[str, int] = {}
                total_area = 0.0
                for det in insp.yolo_detections:
                    sev = det.get("severity", "unknown")
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                    total_area += det.get("area_percent", 0)
                sev_str = ", ".join(f"{s}: {c}" for s, c in severity_counts.items())
                area_str = f"  |  Total flagged area: {total_area:.2f}%"

            lines.append(
                f"Report {insp.id}  |  Vessel: {vessel_label}  |  "
                f"{insp.created_at.strftime('%Y-%m-%d %H:%M')}  |  "
                f"Method: {insp.get_detection_method_display()}  |  "
                f"Severity: {(insp.overall_severity or 'N/A').upper()}  |  "
                f"Defects: {insp.detection_count}"
                f"{area_str}"
            )
            if sev_str:
                lines.append(f"  Severity breakdown: {sev_str}")

            if is_focused:
                lines.append("  ↑ Full report included above")
            elif not focused_set and insp.report_content:
                # Include excerpt only in general mode
                snippet = insp.report_content[:800].replace("\n", " ").strip()
                if len(insp.report_content) > 800:
                    snippet += " [truncated]"
                lines.append(f"  Excerpt: {snippet}")

            lines.append("")  # blank line between inspections

        return "\n".join(lines)

    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Send a message and get a response.

        Parameters
        ----------
        user_message : str
            The surveyor's question.
        conversation_history : list[dict] | None
            Previous messages in ``[{"role": "user"|"assistant", "content": "..."}]``
            format for multi-turn context.

        Returns
        -------
        str
            The assistant's response text.
        """
        # Extract report IDs the user is asking about
        focused_ids = self._extract_report_ids(user_message)

        # Also scan recent conversation history for IDs (user may reference
        # an ID in an earlier message and then ask follow-up questions)
        if not focused_ids and conversation_history:
            for msg in reversed(conversation_history[-6:]):
                if msg.get("role") == "user":
                    focused_ids = self._extract_report_ids(msg.get("content", ""))
                    if focused_ids:
                        break

        # Build context — full report for focused IDs, short for rest
        context = self.get_inspection_context(focused_ids=focused_ids or None)
        system_msg = self.SYSTEM_PROMPT.format(context=context)

        # Build messages list
        messages = [{"role": "system", "content": system_msg}]

        # Add conversation history (keep last 10 turns to stay within context)
        if conversation_history:
            messages.extend(conversation_history[-10:])

        messages.append({"role": "user", "content": user_message})

        # Increase context window when a full report is included
        num_ctx = 8192 if focused_ids else 4096

        # Call Ollama
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1500,
                        "num_ctx": num_ctx,
                    },
                },
                timeout=180,
            )

            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "No response generated.")
            else:
                return f"Error: Ollama returned status {r.status_code}. Is it running?"

        except requests.exceptions.ConnectionError:
            return (
                "I can't connect to the Ollama server. "
                "Please start it with: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            return "The response took too long. Please try a shorter question."
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def get_suggested_questions(self) -> List[str]:
        """Return suggested starter questions based on available data."""
        from inspection.models import InspectionReport

        inspections = InspectionReport.objects.filter(status="completed")
        count = inspections.count()

        suggestions = []
        if count == 0:
            suggestions.append("What can you help me with?")
            suggestions.append("How does the inspection system work?")
        else:
            suggestions.append("Give me an overview of all inspections")
            suggestions.append("Which vessel has the worst condition?")

            # Add vessel-specific question if vessels exist
            vessels = (
                inspections.exclude(vessel_name__isnull=True)
                .exclude(vessel_name="")
                .values_list("vessel_name", flat=True)
                .distinct()[:3]
            )
            for v in vessels:
                suggestions.append(f"What is the condition of {v}?")

            if count > 1:
                suggestions.append("Compare all inspected vessels")
                suggestions.append("Which inspections need urgent attention?")

        return suggestions[:5]
