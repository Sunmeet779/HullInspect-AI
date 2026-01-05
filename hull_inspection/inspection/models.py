"""
Models for Hull Inspection System
"""
from django.db import models
from django.utils import timezone
import uuid
import os


def upload_to_hull_images(instance, filename):
    """Generate unique path for uploaded hull images."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads', 'hull_images', filename)


def upload_to_annotated(instance, filename):
    """Generate path for annotated images."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}_annotated.{ext}"
    return os.path.join('uploads', 'annotated', filename)


class InspectionReport(models.Model):
    """Model to store inspection reports and related data."""
    
    # Status choices
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    # Severity choices
    SEVERITY_CHOICES = [
        ('critical', 'Critical'),
        ('severe', 'Severe'),
        ('moderate', 'Moderate'),
        ('mild', 'Mild'),
        ('good', 'Good Condition'),
    ]
    
    # Basic info
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Images
    original_image = models.ImageField(upload_to=upload_to_hull_images)
    annotated_image = models.ImageField(upload_to=upload_to_annotated, null=True, blank=True)
    
    # Processing status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # YOLO Detection Results (stored as JSON)
    yolo_detections = models.JSONField(null=True, blank=True)
    detection_count = models.IntegerField(default=0)
    
    # VLM Generated Report
    report_content = models.TextField(null=True, blank=True)
    report_html = models.TextField(null=True, blank=True)
    
    # Overall assessment
    overall_severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, null=True, blank=True)
    
    # Report file (PDF)
    report_pdf = models.FileField(upload_to='reports/', null=True, blank=True)
    
    # Metadata
    vessel_name = models.CharField(max_length=200, null=True, blank=True)
    inspector_notes = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Inspection Report'
        verbose_name_plural = 'Inspection Reports'
    
    def __str__(self):
        return f"Inspection {self.id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_defect_summary(self):
        """Get summary of detected defects."""
        if not self.yolo_detections:
            return {}
        
        summary = {}
        for detection in self.yolo_detections:
            class_name = detection.get('class_name', 'unknown')
            if class_name not in summary:
                summary[class_name] = 0
            summary[class_name] += 1
        
        return summary


class DefectAnnotation(models.Model):
    """Individual defect annotations from YOLO detection."""
    
    inspection = models.ForeignKey(
        InspectionReport, 
        on_delete=models.CASCADE, 
        related_name='defects'
    )
    
    # Bounding box coordinates (normalized 0-1)
    x_center = models.FloatField()
    y_center = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()
    
    # Detection info
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    
    # Additional metadata
    severity = models.CharField(max_length=50, null=True, blank=True)
    
    class Meta:
        verbose_name = 'Defect Annotation'
        verbose_name_plural = 'Defect Annotations'
    
    def __str__(self):
        return f"{self.class_name} ({self.confidence:.2f}) - {self.inspection.id}"
    
    def get_bbox_pixels(self, img_width, img_height):
        """Convert normalized coordinates to pixel coordinates."""
        x1 = int((self.x_center - self.width/2) * img_width)
        y1 = int((self.y_center - self.height/2) * img_height)
        x2 = int((self.x_center + self.width/2) * img_width)
        y2 = int((self.y_center + self.height/2) * img_height)
        return x1, y1, x2, y2
