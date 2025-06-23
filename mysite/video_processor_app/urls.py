from django.urls import path
from . import views

app_name = 'video_processor_app'

urlpatterns = [
    path('upload/', views.video_upload_view, name='video_upload'),
    path('results/<int:pk>/', views.VideoResultsView.as_view(), name='video_results'),
    path('results/<int:pk>/data/', views.get_video_detections_data, name='get_video_detections_data'),
    # Optional: URL to manually trigger processing if not done automatically
    # path('process/<int:pk>/start/', views.trigger_processing_view, name='trigger_processing'),
]
