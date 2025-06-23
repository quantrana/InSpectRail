from django.conf import settings
from .models import UploadedVideo, VideoDetection
# Only import the functions, not the global variables directly
from cam_app.camera import process_frame_for_detections, initialize_yolo_model 
import cv2
import os
import traceback # For detailed error logging

def process_video_file(uploaded_video_id):
    try:
        uploaded_video = UploadedVideo.objects.get(id=uploaded_video_id)
    except UploadedVideo.DoesNotExist:
        print(f"[tasks.py] Error: UploadedVideo with id {uploaded_video_id} not found.")
        return

    print(f"[tasks.py] Starting processing for video ID: {uploaded_video_id}, File: {uploaded_video.file.name}")

    print("[tasks.py] Calling initialize_yolo_model()...")
    local_yolo_model, local_yolo_class_names = initialize_yolo_model()
    print(f"[tasks.py] Returned from initialize_yolo_model() - Model is None: {local_yolo_model is None}, Class Names: {local_yolo_class_names}")

    model_ready = False
    if local_yolo_model is not None:
        if local_yolo_class_names and isinstance(local_yolo_class_names, list) and len(local_yolo_class_names) > 0:
            model_ready = True
            print("[tasks.py] Model check: Model and class names received successfully.")
        else:
            print(f"[tasks.py] Model check Error: Model object received, but class names are invalid/empty. Names: {local_yolo_class_names}. Video ID: {uploaded_video_id}")
    else:
        print(f"[tasks.py] Model check Error: Model object is None. Video ID: {uploaded_video_id}")

    if not model_ready:
        print(f"[tasks.py] Critical Error: YOLO model not fully initialized. Processing cannot continue for video {uploaded_video_id}.")
        uploaded_video.processing_status = 'failed'
        uploaded_video.save()
        return
    
    print(f"[tasks.py] Model confirmed ready. Using model type: {type(local_yolo_model)}, First 3 class names: {local_yolo_class_names[:3]}...")

    video_path = uploaded_video.file.path
    print(f"[tasks.py] Video file path: {video_path}")
    if not os.path.exists(video_path):
        print(f"[tasks.py] Error: Video file not found at {video_path} for video ID {uploaded_video_id}.")
        uploaded_video.processing_status = 'failed'
        uploaded_video.save()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[tasks.py] Error: Could not open video file {video_path} for video ID {uploaded_video_id}.")
        uploaded_video.processing_status = 'failed'
        uploaded_video.save()
        return

    uploaded_video.processing_status = 'processing'
    uploaded_video.save()
    print(f"[tasks.py] Video status set to 'processing' for video ID: {uploaded_video_id}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[tasks.py] Warning: Video FPS is {fps} for {video_path}. Assuming 25 FPS.")
        fps = 25
    
    PROCESS_EVERY_N_SECONDS = 1.0
    frame_interval = int(fps * PROCESS_EVERY_N_SECONDS)
    if frame_interval <= 0: frame_interval = 1
    print(f"[tasks.py] Processing approx. every {frame_interval} frames.")

    frame_count = 0
    processed_frame_counter = 0
    detections_saved_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break 
            frame_count += 1
            if frame_count % frame_interval != 0: continue 
            processed_frame_counter +=1
            current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_time_sec = current_time_msec / 1000.0
            if frame is None: continue
            
            _, frame_detections_list = process_frame_for_detections(
                frame, local_yolo_model, local_yolo_class_names, video_timestamp_sec=current_time_sec
            )
            for det_data in frame_detections_list:
                if det_data['confidence'] < 0.45:
                    continue  # Skip detections below confidence threshold
                VideoDetection.objects.create(
                    video=uploaded_video,
                    timestamp_in_video=current_time_sec,
                    label=det_data['label'],
                    confidence=det_data['confidence'],
                    bounding_box_x1=det_data['bounding_box'][0],
                    bounding_box_y1=det_data['bounding_box'][1],
                    bounding_box_x2=det_data['bounding_box'][2],
                    bounding_box_y2=det_data['bounding_box'][3],
                )
                detections_saved_count += 1
            if processed_frame_counter % 10 == 0:
                 print(f"[tasks.py] Processed {processed_frame_counter} frames for video {uploaded_video_id} (time: {current_time_sec:.2f}s). Detections: {detections_saved_count}")
        uploaded_video.processing_status = 'completed'
        print(f"[tasks.py] Video processing completed for {uploaded_video_id}. Frames processed: {processed_frame_counter}. Detections: {detections_saved_count}.")
    except Exception as e:
        print(f"[tasks.py] CRITICAL EXCEPTION during video processing loop for {uploaded_video_id}: {e}")
        traceback.print_exc()
        uploaded_video.processing_status = 'failed'
    finally:
        cap.release()
        uploaded_video.save()
        print(f"[tasks.py] Final status for video {uploaded_video_id}: {uploaded_video.processing_status}")
    


# Example of how to call this (depending on your chosen task queue):
#
# If using Celery:
# In your views.py (e.g., after form.save()):
#   from .tasks import process_video_file
#   process_video_file.delay(uploaded_video.id)
#
# If using Django-Q:
# In your views.py:
#   from django_q.tasks import async_task
#   async_task('video_processor_app.tasks.process_video_file', uploaded_video.id)
#
# If running synchronously (NOT RECOMMENDED FOR WEB REQUESTS, but for testing):
#   process_video_file(uploaded_video.id)
