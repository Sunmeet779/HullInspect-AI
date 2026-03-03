"""
URL patterns for Inspection app
"""
from django.urls import path
from . import views

app_name = 'inspection'

urlpatterns = [
    # Main pages
    path('', views.HomeView.as_view(), name='home'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('results/<uuid:pk>/', views.ResultsView.as_view(), name='results'),
    
    # API endpoints
    path('api/upload/', views.api_upload_image, name='api_upload'),
    path('api/status/<uuid:pk>/', views.api_check_status, name='api_status'),
    path('api/report/<uuid:pk>/pdf/', views.api_download_pdf, name='api_download_pdf'),
    
    # History
    path('history/', views.HistoryView.as_view(), name='history'),
    
    # Chatbot
    path('chat/', views.ChatView.as_view(), name='chat'),
    path('api/chat/', views.api_chat, name='api_chat'),
]
