"""
ASGI config for hull_inspection project.
"""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hull_inspection.settings')
application = get_asgi_application()
