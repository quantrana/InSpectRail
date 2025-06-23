import pickle
from django.conf import settings
from cam_app import views
from django.http import StreamingHttpResponse
import sqlite3
import datetime

# import some common libraries
import numpy as np
import os, json, cv2, random, glob, uuid
import matplotlib.pyplot as plt

from pathlib import Path
import time
import torch
from ultralytics import YOLO

import os

# Global model and class names to be initialized once
YOLO_MODEL_GLOBAL = None
YOLO_CLASS_NAMES_GLOBAL = []

def initialize_yolo_model():
    # Use local variables for initialization, then assign to globals AND return
    # This function will now be the single source of truth for loading.
    global YOLO_MODEL_GLOBAL, YOLO_CLASS_NAMES_GLOBAL

    # If already loaded by this process/thread, return the existing ones.
    if YOLO_MODEL_GLOBAL is not None and YOLO_CLASS_NAMES_GLOBAL:
        print("[initialize_yolo_model] Returning already loaded global model and class names.")
        return YOLO_MODEL_GLOBAL, YOLO_CLASS_NAMES_GLOBAL

    print("Attempting to initialize YOLO model (globals were None/empty)...")
    local_model = None
    local_class_names = []
    model_path = os.path.join(settings.BASE_DIR, 'yolo11', 'best.pt')
    print(f"[initialize_yolo_model] Calculated model path: {model_path}")

    if not os.path.exists(model_path):
        print(f"[initialize_yolo_model] CRITICAL ERROR: Model file does not exist at {model_path}")
        return None, []

    try:
        print(f"[initialize_yolo_model] Loading model from: {model_path}")
        temp_model = YOLO(model_path)
        print("[initialize_yolo_model] YOLO(model_path) call completed.")

        if temp_model:
            print("[initialize_yolo_model] temp_model object created successfully.")
            names_dict = None
            if hasattr(temp_model, 'names') and isinstance(temp_model.names, dict) and temp_model.names:
                names_dict = temp_model.names
            elif hasattr(temp_model, 'model') and hasattr(temp_model.model, 'names') and isinstance(temp_model.model.names, dict) and temp_model.model.names:
                names_dict = temp_model.model.names
            
            if names_dict:
                if all(isinstance(k, int) for k in names_dict.keys()):
                    local_class_names = [names_dict[i] for i in sorted(names_dict.keys())]
                else:
                    local_class_names = list(names_dict.values())
            
            if local_class_names:
                local_model = temp_model
                print(f"[initialize_yolo_model] Successfully loaded model and class names: {local_class_names}")
            else:
                print("[initialize_yolo_model] Warning: Class names list is empty. Using hardcoded fallback.")
                local_model = temp_model 
                local_class_names = ['Cracks', 'flaking', 'joints', 'squats']
        else:
            print("[initialize_yolo_model] Error: YOLO(model_path) returned a Falsy model object.")
    
    except Exception as e:
        print(f"[initialize_yolo_model] CRITICAL EXCEPTION during YOLO model loading: {e}")
        local_model = None
        local_class_names = []

    # Update globals for any part of cam_app that might still use them directly
    YOLO_MODEL_GLOBAL = local_model
    YOLO_CLASS_NAMES_GLOBAL = local_class_names
    
    if local_model and local_class_names:
        print("[initialize_yolo_model] Function end: Returning successfully loaded model and class names.")
    else:
        print("[initialize_yolo_model] Function end: Failed to load. Returning None/empty.")
        
    return local_model, local_class_names



def process_frame_for_detections(frame_bgr, model, class_names, video_timestamp_sec=None):
    """
    Processes a single frame for object detection using a YOLO model.

    Args:
        frame_bgr: The input image frame in BGR format (from OpenCV).
        model: The loaded YOLO model instance.
        class_names: A list of class names corresponding to model output.
        video_timestamp_sec: Optional. Timestamp in seconds if processing a video file.
                             If None, a real-time timestamp will be generated.

    Returns:
        A tuple: (processed_frame_bgr, detections_list)
        processed_frame_bgr: The frame with bounding boxes drawn (BGR format).
        detections_list: A list of dictionaries, where each dictionary contains
                         'label', 'confidence', 'bounding_box', and 'timestamp'.
    """
    
    COLORS = [
    (255, 255, 255),  # white
    (0, 255, 255),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 0),      # green
    (255, 255, 0),    # yellow
    (0, 0, 255),      # red
    (255, 0, 0)       # blue
    ]
    if frame_bgr is None:
        # This case should ideally be handled before calling this function
        print("Error: process_frame_for_detections received a None frame.")
        return None, []

    processed_frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8).copy()
    original_height, original_width = processed_frame_bgr.shape[:2]
    
    # Resize for the model
    img_for_model = cv2.resize(processed_frame_bgr, (640, 640))
    
    # Manual PyTorch tensor conversion
    img_rgb = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_chw = np.ascontiguousarray(img_chw)
    img_norm = img_chw.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(img_norm)
    input_tensor = input_tensor.unsqueeze(0)
    
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(model.parameters()).device # Fallback for some model types
    input_tensor = input_tensor.to(model_device)

    # Perform inference
    results = model(input_tensor)
    
    current_detections = []

    # Process results
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = detection
        if float(confidence) < 0.45:
            continue  # Skip detections with confidence < 0.45
        # Scale coordinates back to original frame size
        x_scale = original_width / 640.0
        y_scale = original_height / 640.0
        x1_orig = int(x1 * x_scale)
        y1_orig = int(y1 * y_scale)
        x2_orig = int(x2 * x_scale)
        y2_orig = int(y2 * y_scale)
        
        label = class_names[int(class_id)]
        text_to_display = f'{label} {confidence:.2f}'
        
        # Draw on the original resolution frame copy
        color = COLORS[int(class_id) % len(COLORS)]

        # Draw semi-transparent box (optional)
        overlay = processed_frame_bgr.copy()
        cv2.rectangle(overlay, (x1_orig, y1_orig), (x2_orig, y2_orig), color, -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, processed_frame_bgr, 1 - alpha, 0, processed_frame_bgr)

        # Draw bounding box
        cv2.rectangle(processed_frame_bgr, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)

        # Label background
        (text_width, text_height), baseline = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(processed_frame_bgr, (x1_orig, y1_orig - text_height - 10), (x1_orig + text_width, y1_orig), color, -1)

        # Draw label text
        cv2.putText(processed_frame_bgr, text_to_display, (x1_orig, y1_orig - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


        if video_timestamp_sec is not None:
            timestamp_str = f"{int(video_timestamp_sec // 3600):02d}:{int((video_timestamp_sec % 3600) // 60):02d}:{int(video_timestamp_sec % 60):02d}"
        else:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        current_detections.append({
            "label": label,
            "confidence": round(float(confidence), 2),
            "bounding_box": [x1_orig, y1_orig, x2_orig, y2_orig],
            "timestamp": timestamp_str 
        })
        
    return processed_frame_bgr, current_detections


class VideoCamera(object):
    def __init__(self):
        self.dashboard = []
        self.video = None
        self._open_camera()
        # Model is loaded on-demand by get_frame_with_detection

    def _open_camera(self):
        if self.video is None or not self.video.isOpened():
            self.video = cv2.VideoCapture(0)
            if not self.video.isOpened():
                print("[VideoCamera] Error: Cannot open camera.")
                self.video = None

    def __del__(self):
        if self.video and self.video.isOpened():
            self.video.release()

    def get_frame_with_detection(self):
        self.dashboard.clear()
        if self.video is None or not self.video.isOpened():
            self._open_camera()
            if self.video is None:
                placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_image, "No Camera", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                ret, jpeg_bytes = cv2.imencode('.jpg', placeholder_image)
                return (jpeg_bytes.tobytes() if ret else b''), placeholder_image, []

        success, frame_bgr = self.video.read()
        if not success or frame_bgr is None:
            placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_image, "No Signal", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, jpeg_bytes = cv2.imencode('.jpg', placeholder_image)
            return (jpeg_bytes.tobytes() if ret else b''), placeholder_image, []

        current_model, current_class_names = initialize_yolo_model()

        if not current_model or not current_class_names:
            print("[VideoCamera] Error: YOLO model not available for live detection. Returning raw frame.")
            ret, output_jpeg_bytes = cv2.imencode('.jpg', frame_bgr)
            return (output_jpeg_bytes.tobytes() if ret else b''), frame_bgr, []
        
        # Ensure process_frame_for_detections is imported or defined in this file
        processed_frame_bgr, frame_detections = process_frame_for_detections(
            frame_bgr, current_model, current_class_names
        )
        
        self.dashboard.extend(frame_detections)
        ret, output_jpeg_bytes = cv2.imencode('.jpg', processed_frame_bgr)
        if not ret:
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "JPEG Encode Error", (50,240), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            ret_err, error_jpeg = cv2.imencode('.jpg', error_img)
            return (error_jpeg.tobytes() if ret_err else b''), frame_bgr, []
        return output_jpeg_bytes.tobytes(), processed_frame_bgr, self.dashboard

    def get_frame_without_detection(self):
        if self.video is None or not self.video.isOpened():
            self._open_camera()
            if self.video is None:
                placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_image, "No Camera", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                ret, jpeg_bytes = cv2.imencode('.jpg', placeholder_image)
                return (jpeg_bytes.tobytes() if ret else b''), placeholder_image
        
        success, image = self.video.read()
        if not success or image is None:
            placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_image, "No Signal / Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            ret, jpeg_bytes = cv2.imencode('.jpg', placeholder_image)
            return (jpeg_bytes.tobytes() if ret else b''), placeholder_image
        
        ret, outputImagetoReturn = cv2.imencode('.jpg', image)
        return outputImagetoReturn.tobytes(), image

def generate_frames(camera, AI):
    try:
        while True:
            if AI:
                frame, img, dashboard = camera.get_frame_with_detection()
            if not AI:
                frame, img = camera.get_frame_without_detection()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(e)

    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()
