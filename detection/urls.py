from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import check_360, index, serve_video

urlpatterns = [
    path('', index, name='index'),
    path('api/check360/', check_360, name='check_360'),
    path('media/videos/<str:filename>', serve_video, name='serve_video'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) if settings.DEBUG else []

