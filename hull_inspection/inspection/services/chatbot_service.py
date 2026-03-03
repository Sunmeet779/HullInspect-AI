"""
Chatbot Service for Maritime Hull Inspection
=============================================
RAG-style chatbot that retrieves past inspection data from the DB
and uses a local LLM (Ollama Phi-3 Mini) to answer surveyor questions.

No training required — the LLM is pre-trained and receives inspection
context via prompt engineering.
"""

import json
import requests
from typing import Dict, List, Optional
from datetime import datetime


class InspectionChatbot:
    """
    Conversational assistant for maritime surveyors.

    Retrieves inspection reports from the database and uses Ollama
    (Phi-3 Mini) to generate informed answers about vessel conditions,
    defect trends, and maintenance recommendations.
    """

    SYSTEM_PROMPT = """You are HullInspect AI Assistant — a knowledgeable maritime inspection chatbot.
You help surveyors understand hull inspection results, compare vessel conditions, and provide maintenance guidance.

RULES:
- Answer based ONLY on the inspection data provided in the context below.
- If no relevant data exists for a question, say so clearly.
- Use professional maritime terminology.
- Be concise but thorough. Use bullet points for lists.
- When comparing vessels, present data in a clear structured format.
- Include specific numbers (detection counts, severity, area percentages) when available.
- If the surveyor asks about a vessel not in the data, say it hasn't been inspected yet.

CONTEXT — All completed inspections in the system:
{context}
"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "phi3:mini",
        max_context_inspections: int = 50,
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

    def get_inspection_context(self) -> str:
        """
        Build a text summary of all completed inspections for the LLM context.
        Imports Django models inside the function so the module can be imported
        without Django being fully configured.
        """
        from inspection.models import InspectionReport

        inspections = InspectionReport.objects.filter(status="completed").order_by(
            "-created_at"
        )[: self.max_context_inspections]

        if not inspections:
            return "No inspections have been completed yet."

        lines: List[str] = []
        lines.append(f"Total completed inspections: {inspections.count()}\n")

        # Group by vessel
        vessel_groups: Dict[str, list] = {}
        for insp in inspections:
            vname = insp.vessel_name or "Unknown Vessel"
            vessel_groups.setdefault(vname, []).append(insp)

        for vessel, insps in vessel_groups.items():
            lines.append(f"═══ Vessel: {vessel} ({len(insps)} inspection(s)) ═══")
            for insp in insps:
                lines.append(f"  Inspection ID: {str(insp.id)[:8]}")
                lines.append(f"  Date: {insp.created_at.strftime('%Y-%m-%d %H:%M')}")
                lines.append(f"  Method: {insp.get_detection_method_display()}")
                lines.append(f"  Severity: {(insp.overall_severity or 'N/A').upper()}")
                lines.append(f"  Defects found: {insp.detection_count}")

                # Summarise detections
                if insp.yolo_detections:
                    severity_counts: Dict[str, int] = {}
                    total_area = 0.0
                    for det in insp.yolo_detections:
                        sev = det.get("severity", "unknown")
                        severity_counts[sev] = severity_counts.get(sev, 0) + 1
                        total_area += det.get("area_percent", 0)
                    sev_str = ", ".join(
                        f"{s}: {c}" for s, c in severity_counts.items()
                    )
                    lines.append(f"  Severity breakdown: {sev_str}")
                    lines.append(f"  Total corroded area: {total_area:.2f}%")

                # Short snippet of the text report
                if insp.report_content:
                    snippet = insp.report_content[:400].replace("\n", " ").strip()
                    lines.append(f"  Report excerpt: {snippet}...")

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
        # Build context from database
        context = self.get_inspection_context()
        system_msg = self.SYSTEM_PROMPT.format(context=context)

        # Build messages list
        messages = [{"role": "system", "content": system_msg}]

        # Add conversation history (keep last 10 turns to stay within context)
        if conversation_history:
            messages.extend(conversation_history[-10:])

        messages.append({"role": "user", "content": user_message})

        # Call Ollama
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 1024,
                    },
                },
                timeout=120,
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
