"""
Django Admin configuration for Hull Inspection
"""
from django.contrib import admin
from .models import InspectionReport, DefectAnnotation


class DefectAnnotationInline(admin.TabularInline):
    model = DefectAnnotation
    extra = 0
    readonly_fields = ['class_name', 'confidence', 'severity', 'x_center', 'y_center', 'width', 'height']


@admin.register(InspectionReport)
class InspectionReportAdmin(admin.ModelAdmin):
    list_display = ['id', 'created_at', 'status', 'overall_severity', 'detection_count', 'vessel_name']
    list_filter = ['status', 'overall_severity', 'created_at']
    search_fields = ['id', 'vessel_name']
    readonly_fields = ['id', 'created_at', 'updated_at', 'detection_count', 'yolo_detections']
    inlines = [DefectAnnotationInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'vessel_name', 'status', 'created_at', 'updated_at')
        }),
        ('Images', {
            'fields': ('original_image', 'annotated_image')
        }),
        ('Detection Results', {
            'fields': ('detection_count', 'yolo_detections', 'overall_severity')
        }),
        ('Report', {
            'fields': ('report_content', 'report_pdf'),
            'classes': ('collapse',)
        }),
        ('Notes', {
            'fields': ('inspector_notes',),
            'classes': ('collapse',)
        }),
    )


@admin.register(DefectAnnotation)
class DefectAnnotationAdmin(admin.ModelAdmin):
    list_display = ['id', 'inspection', 'class_name', 'confidence', 'severity']
    list_filter = ['class_name', 'severity']
    search_fields = ['inspection__id', 'class_name']
