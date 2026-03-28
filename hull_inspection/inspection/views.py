"""
Views for Hull Inspection System
"""
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import TemplateView, DetailView, ListView
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
import json
import os

from .models import InspectionReport, DefectAnnotation
# Use segmentation-based services (v2) for better corrosion detection
try:
    from .services.yolo_seg_inference import YOLOSegInferenceService as YOLOInferenceService
except ImportError:
    from .services.yolo_inference import YOLOInferenceService
try:
    from .services.vlm_service_v2 import VLMReportGenerator
except ImportError:
    from .services.vlm_service import VLMReportGenerator
from .services.report_generator import PDFReportGenerator
from .services.eigen_tree_service import EigenTreeCorrosionDetector
from .services.chatbot_service import InspectionChatbot


class HomeView(TemplateView):
    """Homepage with system introduction."""
    template_name = 'inspection/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['recent_inspections'] = InspectionReport.objects.filter(
            status='completed'
        )[:5]
        context['total_inspections'] = InspectionReport.objects.filter(
            status='completed'
        ).count()
        return context


class UploadView(TemplateView):
    """Image upload page."""
    template_name = 'inspection/upload.html'


class ResultsView(DetailView):
    """Results page showing detection and report."""
    model = InspectionReport
    template_name = 'inspection/results.html'
    context_object_name = 'inspection'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        inspection = self.object
        
        # Get defect summary
        context['defect_summary'] = inspection.get_defect_summary()
        context['defects'] = inspection.defects.all()
        
        return context


class HistoryView(ListView):
    """View past inspections."""
    model = InspectionReport
    template_name = 'inspection/history.html'
    context_object_name = 'inspections'
    paginate_by = 10
    
    def get_queryset(self):
        return InspectionReport.objects.filter(status='completed')


@csrf_exempt
@require_POST
def api_upload_image(request):
    """
    API endpoint to upload and process hull image.
    Returns inspection ID for status polling.
    """
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        vessel_name = request.POST.get('vessel_name', '')
        detection_method = request.POST.get('detection_method', 'yolo_seg')
        if detection_method not in ('yolo_seg', 'eigen_tree'):
            detection_method = 'yolo_seg'
        
        # Create inspection record
        inspection = InspectionReport.objects.create(
            original_image=image_file,
            vessel_name=vessel_name,
            detection_method=detection_method,
            status='processing'
        )
        
        # Process the image (in production, use Celery for async)
        try:
            process_inspection(inspection)
        except Exception as e:
            inspection.status = 'failed'
            inspection.save()
            return JsonResponse({
                'error': str(e),
                'inspection_id': str(inspection.id)
            }, status=500)
        
        return JsonResponse({
            'success': True,
            'inspection_id': str(inspection.id),
            'redirect_url': f'/results/{inspection.id}/'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def api_check_status(request, pk):
    """Check processing status of an inspection."""
    inspection = get_object_or_404(InspectionReport, pk=pk)
    
    return JsonResponse({
        'status': inspection.status,
        'detection_count': inspection.detection_count,
        'has_report': bool(inspection.report_content),
    })


def api_download_pdf(request, pk):
    """Download inspection report as PDF."""
    inspection = get_object_or_404(InspectionReport, pk=pk)
    
    if not inspection.report_pdf:
        # Generate PDF on demand
        generator = PDFReportGenerator()
        pdf_path = generator.generate(inspection)
        inspection.report_pdf = pdf_path
        inspection.save()
    
    if inspection.report_pdf:
        return FileResponse(
            inspection.report_pdf.open('rb'),
            as_attachment=True,
            filename=f'inspection_report_{inspection.id}.pdf'
        )
    
    return JsonResponse({'error': 'PDF not available'}, status=404)


def process_inspection(inspection: InspectionReport):
    """
    Main processing pipeline for hull inspection.
    1. Run YOLO detection
    2. Generate VLM report
    3. Create annotated image
    4. Update inspection record
    """
    image_path = inspection.original_image.path
    
    # Step 1: Run detection based on selected method
    method = getattr(inspection, 'detection_method', 'yolo_seg')
    if method == 'eigen_tree':
        detector = EigenTreeCorrosionDetector(max_depth=7, prune=True)
        detections, annotated_image_path = detector.detect(image_path)
    else:
        yolo_service = YOLOInferenceService()
        detections, annotated_image_path = yolo_service.detect(image_path)
    
    # Store detections (includes mask area data)
    inspection.yolo_detections = detections
    inspection.detection_count = len(detections)
    
    # Save annotated image (now with segmentation mask overlays)
    if annotated_image_path:
        from django.core.files import File
        with open(annotated_image_path, 'rb') as f:
            inspection.annotated_image.save(
                os.path.basename(annotated_image_path),
                File(f)
            )
    
    # Create DefectAnnotation records
    for det in detections:
        DefectAnnotation.objects.create(
            inspection=inspection,
            x_center=det['x_center'],
            y_center=det['y_center'],
            width=det['width'],
            height=det['height'],
            class_id=det['class_id'],
            class_name=det['class_name'],
            confidence=det['confidence'],
            severity=det.get('severity')
        )
    
    # Step 2: Generate VLM Report (v2 — method-aware)
    vlm_service = VLMReportGenerator()
    report_content, report_html, severity = vlm_service.generate_report(
        image_path=image_path,
        detections=detections,
        detection_method=method,
    )
    
    inspection.report_content = report_content
    inspection.report_html = report_html
    inspection.overall_severity = severity
    
    # Step 3: Mark as completed
    inspection.status = 'completed'
    inspection.save()
    
    return inspection


# ── Chatbot views ─────────────────────────────────────────────────────

class ChatView(TemplateView):
    """Chatbot page."""
    template_name = 'inspection/chat.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        chatbot = InspectionChatbot()
        context['ollama_available'] = chatbot.check_available()
        context['suggestions'] = chatbot.get_suggested_questions()
        context['total_inspections'] = InspectionReport.objects.filter(
            status='completed'
        ).count()
        return context


@csrf_exempt
@require_POST
def api_chat(request):
    """
    AJAX endpoint for chatbot messages.

    Expects JSON: {"message": "...", "history": [...]}
    Returns JSON: {"reply": "...", "status": "ok"|"error"}
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"reply": "Invalid request.", "status": "error"}, status=400)

    user_message = body.get("message", "").strip()
    history = body.get("history", [])

    if not user_message:
        return JsonResponse({"reply": "Please enter a message.", "status": "error"})

    chatbot = InspectionChatbot()

    if not chatbot.check_available():
        return JsonResponse({
            "reply": "The Ollama server is not running. Please start it with `ollama serve` and ensure the phi3:mini model is pulled.",
            "status": "error",
        })

    reply = chatbot.chat(user_message, conversation_history=history)
    return JsonResponse({"reply": reply, "status": "ok"})
