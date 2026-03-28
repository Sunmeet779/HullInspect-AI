# Hull Inspection Project - Django Settings

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-replace-this-with-a-secure-key-in-production'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '*']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Local apps
    'inspection.apps.InspectionConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'hull_inspection.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'hull_inspection.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files (User uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# =============================================================================
# =============================================================================
# YOLO & ML Configuration
# =============================================================================
# Segmentation model (preferred — YOLO11-Seg)
YOLO_SEG_MODEL_PATH = BASE_DIR / 'ml_models' / 'yolo' / 'weights' / 'best_seg.pt'
YOLO_SEG_RUNS_PATH = BASE_DIR / 'runs' / 'segment'

# Legacy detection model (fallback)
YOLO_MODEL_PATH = BASE_DIR / 'ml_models' / 'yolo' / 'weights' / 'best.pt'
YOLO_RUNS_PATH = BASE_DIR / 'runs' / 'detect'

YOLO_CONFIG_PATH = BASE_DIR / 'ml_models' / 'yolo' / 'config'

# Dataset paths
YOLO_SEG_DATASET_PATH = BASE_DIR / 'datasets' / 'hull_defects_seg'
YOLO_DATASET_PATH = BASE_DIR / 'datasets' / 'hull_defects'

# Corrosion class (single class for segmentation — matches the actual dataset)
SEG_CLASSES = {
    0: 'corrosion',
}

# Legacy 8-class mapping (kept for backward compatibility)
DEFECT_CLASSES = {
    0: 'corrosion_severe',
    1: 'corrosion_moderate', 
    2: 'corrosion_mild',
    3: 'paint_damage',
    4: 'fouling',
    5: 'crack',
    6: 'dent',
    7: 'surface_defect',
}

# =============================================================================
# VLM Configuration
# =============================================================================
# 3-stage pipeline: GitHub Models (primary) → Ollama (fallback) → Template
#
# Stage 1: GitHub Models API (GPT-4o-mini)
# Credentials loaded from .env file automatically by vlm_service_v2
# Required .env keys: GITHUB_TOKEN, GITHUB_MODELS_ENDPOINT, GITHUB_MODELS_MODEL
VLM_GITHUB_MODEL = 'gpt-4o-mini'
VLM_GITHUB_ENDPOINT = 'https://models.inference.ai.azure.com'

# Stage 2: Ollama backend (local fallback for Mac M1)
# Install: brew install ollama
# Models (pick one):
#   ollama pull minicpm-v     (3B, RECOMMENDED, best quality/speed ratio)
#   ollama pull llava          (7B, good quality)
#   ollama pull moondream      (1.8B, lightest)
VLM_OLLAMA_URL = 'http://localhost:11434'
# Model auto-detected from installed Ollama models; set to override:
VLM_OLLAMA_MODEL = None  # None = auto-detect best available

# Report Configuration
REPORT_OUTPUT_DIR = MEDIA_ROOT / 'reports'
