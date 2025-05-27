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

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.model = YOLO('./yolo11/best.pt')
        self.class_names = ['Cracks', 'flaking', 'joints', 'squats']
        self.dashboard = []

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self):
        self.dashboard.clear()
        success, image = self.video.read()
        if not success or image is None:
            # Log an error or warning (optional, but good for debugging)
            print("Error: Failed to capture frame or frame is None.")
            # Create a placeholder black image (e.g., 640x480)
            # Adjust dimensions as appropriate if you know the expected camera resolution
            placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_image, "No Signal", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, outputImagetoReturn = cv2.imencode('.jpg', placeholder_image)
            if not ret:
                # Handle encoding failure for placeholder, though unlikely for a simple black image
                # This might return a minimal valid JPEG byte sequence or raise an error
                # For simplicity, we'll assume encoding the placeholder always works
                # Or return (None, None) and let generate_frames handle it by breaking/logging
                return b'', placeholder_image # Return empty bytes and the placeholder
            return outputImagetoReturn.tobytes(), placeholder_image
        image = np.ascontiguousarray(image, dtype=np.uint8).copy()
        original_height, original_width = image.shape[:2]
        image_for_model = cv2.resize(image, (640, 640))
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        outputs = image # Keep this, as drawing happens on the original BGR 'image'
        
        # Manual PyTorch tensor conversion
        # 1. Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_for_model, cv2.COLOR_BGR2RGB) # Use image_for_model

        # 2. Transpose HWC to CHW
        image_chw = np.transpose(image_rgb, (2, 0, 1))
        image_chw = np.ascontiguousarray(image_chw) # Ensure C-contiguous

        # 3. Convert to float32 and normalize
        image_norm = image_chw.astype(np.float32) / 255.0

        # 4. Convert NumPy array to PyTorch tensor
        print(f"Debug (before torch.from_numpy): Type of image_norm: {type(image_norm)}")
        if isinstance(image_norm, np.ndarray):
            print(f"Debug (before torch.from_numpy): image_norm.shape: {image_norm.shape}")
            print(f"Debug (before torch.from_numpy): image_norm.dtype: {image_norm.dtype}")
            print(f"Debug (before torch.from_numpy): image_norm.flags: {image_norm.flags}")
        else:
            print("Debug (before torch.from_numpy): image_norm is NOT a NumPy array.")
        input_tensor = torch.from_numpy(image_norm)

        # 5. Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        # 6. Move tensor to the model's device
        try:
            model_device = self.model.device
        except AttributeError:
            model_device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(model_device)

        # Keep this diagnostic for the upcoming test
        print(f"Debug: Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}, device: {input_tensor.device}")
        
        # Perform inference
        results = self.model(input_tensor)

        # Process results (ensure this loop is present and uses the original 'image' for drawing)
        for detection in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection
            # x1, y1, x2, y2 are for the 640x640 image

            # SCALING:
            # Calculate scaling factors
            x_scale = original_width / 640.0 # Use floating point division
            y_scale = original_height / 640.0 # Use floating point division

            # Scale coordinates
            x1_orig = int(x1 * x_scale)
            y1_orig = int(y1 * y_scale)
            x2_orig = int(x2 * x_scale)
            y2_orig = int(y2 * y_scale)
            
            label = self.class_names[int(class_id)] # Make sure self.class_names is available
            text_to_display = f'{label} {confidence:.2f}'
            
            # Draw on the ORIGINAL 'image' using scaled coordinates
            cv2.rectangle(image, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
            cv2.putText(image, text_to_display, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.dashboard
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.dashboard.append({
                "label": label,
                "confidence": round(float(confidence), 2),
                "bounding_box": [x1_orig, y1_orig, x2_orig, y2_orig],
                "timestamp": timestamp
            })
        
        outputImage = image # This 'image' is the one with drawings
        
        ret, outputImagetoReturn = cv2.imencode('.jpg', outputImage)
        if not ret:
            print("Error: Failed to encode JPEG after YOLO processing.")
            # Fallback or error handling for encoding
            # For now, create a simple black frame with an error message
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "JPEG Encode Error", (50,240), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            ret_err, error_jpeg = cv2.imencode('.jpg', error_img)
            if ret_err: return error_jpeg.tobytes(), error_img
            else: return b'', None # Last resort
        return outputImagetoReturn.tobytes(), outputImage, self.dashboard
    
    def get_frame_without_detection(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        outputs = image
        outputImage = outputs
        ret, outputImagetoReturn = cv2.imencode('.jpg', outputImage)
        return outputImagetoReturn.tobytes(), outputImage

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
