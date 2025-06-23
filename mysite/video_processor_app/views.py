from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse # Keep if get_video_detections_data uses it
from django.views import View
from django.urls import reverse # Make sure reverse is imported
from django.contrib import messages

from .forms import VideoUploadForm
from .models import UploadedVideo
from .tasks import process_video_file

def video_upload_view(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_video = form.save(commit=False)
            if request.user.is_authenticated:
                uploaded_video.user = request.user
            uploaded_video.save()
            
            messages.info(request, f"Video '{uploaded_video.original_filename}' uploaded. Processing synchronously...")
            
            try:
                process_video_file(uploaded_video.id)
                uploaded_video.refresh_from_db() # Get updated status
                if uploaded_video.processing_status == 'completed':
                    messages.success(request, f"Processing for '{uploaded_video.original_filename}' complete.")
                else:
                    messages.error(request, f"Processing for '{uploaded_video.original_filename}' finished with status: {uploaded_video.get_processing_status_display()}.")
            except Exception as e:
                messages.error(request, f"An error occurred during processing for '{uploaded_video.original_filename}': {str(e)}")
                if uploaded_video.processing_status != 'failed':
                    uploaded_video.processing_status = 'failed'
                    uploaded_video.save()
            
            # THIS IS THE CORRECTED LINE:
            return redirect(reverse('video_processor_app:video_results', kwargs={'pk': uploaded_video.pk}))
        else:
            messages.error(request, "Upload failed. Please correct the errors below.")
    else:
        form = VideoUploadForm()
    
    return render(request, 'video_processor_app/upload_video.html', {'form': form})

# Make sure your VideoResultsView and get_video_detections_data functions/classes are also in this file.
# (The content of those doesn't need to change for this specific error)

class VideoResultsView(View):
    def get(self, request, pk):
        video = get_object_or_404(UploadedVideo, pk=pk)
        detections = video.detections.all().order_by('timestamp_in_video')
        context = {
            'video': video,
            'detections': detections,
            'is_processing_complete': video.processing_status == 'completed',
            'is_processing_failed': video.processing_status == 'failed',
            'is_processing_pending': video.processing_status == 'pending' or video.processing_status == 'processing',
        }
        return render(request, 'video_processor_app/video_results.html', context)

def get_video_detections_data(request, pk):
    video = get_object_or_404(UploadedVideo, pk=pk)
    detections_qs = video.detections.all().order_by('timestamp_in_video')
    detections_data = []
    for det in detections_qs:
        detections_data.append({
            'timestamp_in_video': f"{int(det.timestamp_in_video // 3600):02d}:{int((det.timestamp_in_video % 3600) // 60):02d}:{int(det.timestamp_in_video % 60):02d}.{int((det.timestamp_in_video * 1000) % 1000):03d}",
            'label': det.label,
            'confidence': f"{det.confidence:.2f}",
            'bounding_box': [
                det.bounding_box_x1, 
                det.bounding_box_y1, 
                det.bounding_box_x2, 
                det.bounding_box_y2
            ]
        })
    return JsonResponse({
        'detections': detections_data, 
        'processing_status': video.processing_status,
        'video_url': video.file.url if video.file else None,
        'original_filename': video.original_filename
    })