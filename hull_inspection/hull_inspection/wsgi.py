"""
WSGI config for hull_inspection project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hull_inspection.settings')
application = get_wsgi_application()
