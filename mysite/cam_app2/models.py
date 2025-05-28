from django.db import models
from django.shortcuts import render
from django.conf import settings
from django import forms

from modelcluster.fields import ParentalKey

from wagtail.admin.panels import (
    FieldPanel,
    MultiFieldPanel,
)
from wagtail.models import Page
from wagtail.fields import RichTextField
from django.core.files.storage import default_storage

from pathlib import Path
from django.conf import settings # Ensure settings is imported for BASE_DIR

import torch
import numpy as np
import os, uuid, glob, cv2
from ultralytics import YOLO # Added for custom YOLO model

# Placeholder if 'from ultralytics import YOLO' fails (as per instructions, not used initially)
# class YOLO:
#     def __init__(self, model_path): self.model_path = model_path; print(f"Placeholder YOLO loaded: {model_path}")
#     def __call__(self, tensor_input): print("Placeholder YOLO called"); return [type('obj', (object,), {'boxes': type('obj', (object,), {'data': torch.empty(0,6)})})] # Return dummy results matching expected structure
#     def to(self, device): print(f"Placeholder YOLO to {device}"); self._device = device; return self
#     @property
#     def device(self): return getattr(self, '_device', torch.device('cpu'))
#     # Placeholder for model.parameters() if needed by model_device logic and YOLO doesn't have it
#     def parameters(self): yield torch.nn.Parameter(torch.empty(1)) # Dummy parameter for device check

str_uuid = uuid.uuid4()  # The UUID for image uploading

def reset():
    files_result = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/Result/*.*')), recursive=True)
    files_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/*.*')), recursive=True)
    files = []
    if len(files_result) != 0:
        files.extend(files_result)
    if len(files_upload) != 0:
        files.extend(files_upload)
    if len(files) != 0:
        for f in files:
            try:
                if (not (f.endswith(".txt"))):
                    os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        file_li = [Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'),
                   Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'),
                   Path(f'{settings.MEDIA_ROOT}/Result/stats.txt')]
        for p in file_li:
            file = open(Path(p), "r+")
            file.truncate(0)
            file.close()

# Create your models here.
class ImagePage(Page):
    """Image Page."""

    template = "cam_app2/image.html"

    max_count = 2

    name_title = models.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("name_subtitle"),

            ],
            heading="Page Options",
        ),
    ]


    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"]= []
        context["my_result_file_names"]=[]
        context["my_staticSet_names"]= []
        context["my_lines"]= []
        return context

    def serve(self, request):
        emptyButtonFlag = False
        if request.POST.get('start')=="":
            context = self.reset_context(request)
            print(request.POST.get('start'))
            print("Start selected")
            fileroot = os.path.join(settings.MEDIA_ROOT, 'uploadedPics')
            res_f_root = os.path.join(settings.MEDIA_ROOT, 'Result')

            CLASS_NAMES = ['Cracks', 'flaking', 'joints', 'squats']
            model_path_abs = Path(settings.BASE_DIR) / 'yolo11' / 'best.pt'

            try:
                if not model_path_abs.exists():
                    print(f"Error: Model file not found at {str(model_path_abs)}")
                    context["error_message"] = "Object detection model file not found."
                    return render(request, "cam_app2/image.html", context)

                model = YOLO(model_path_abs)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                print(f"YOLO Model loaded successfully on {device}.")

            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                context["error_message"] = f"Error loading object detection model: {e}"
                return render(request, "cam_app2/image.html", context)

            with open(Path(settings.MEDIA_ROOT) / 'uploadedPics' / 'img_list.txt', 'r') as f:
                image_files_lines = f.readlines()

            if len(image_files_lines) > 0:
                for file_line in image_files_lines:
                    try:
                        # Get filename from the line (might be full path or just name)
                        # and ensure it's just the name.
                        img_filename_from_list = file_line.strip().split('/')[-1]
                        
                        filepath = os.path.join(fileroot, img_filename_from_list)
                        
                        if not os.path.exists(filepath):
                            print(f"Error: Image file not found at {filepath} (from list: {file_line.strip()})")
                            context["my_uploaded_file_names"].append(str(Path(settings.MEDIA_URL) / 'uploadedPics' / img_filename_from_list) + " (File not found)")
                            continue 

                        original_image = cv2.imread(filepath)
                        if original_image is None:
                            print(f"Error: Could not read image {filepath}")
                            context["my_uploaded_file_names"].append(str(Path(settings.MEDIA_URL) / 'uploadedPics' / img_filename_from_list) + " (Could not read)")
                            continue
                        
                        original_height, original_width = original_image.shape[:2]

                        # Preprocessing
                        image_for_model = cv2.resize(original_image, (640, 640))
                        image_rgb = cv2.cvtColor(image_for_model, cv2.COLOR_BGR2RGB)
                        image_chw = np.transpose(image_rgb, (2, 0, 1)) # HWC to CHW
                        image_chw = np.ascontiguousarray(image_chw)
                        image_norm = image_chw.astype(np.float32) / 255.0
                        input_tensor = torch.from_numpy(image_norm).unsqueeze(0)
                        
                        # Ensure tensor is on the same device as the model
                        # model_device = next(model.parameters()).device # Common PyTorch way
                        # For Ultralytics YOLO, model.device is usually available
                        model_device = model.device 
                        input_tensor = input_tensor.to(model_device)

                        # Perform inference
                        results = model(input_tensor)

                        # Postprocessing
                        output_image_to_draw_on = original_image.copy()
                        
                        # Assuming results[0].boxes.data gives [x1, y1, x2, y2, conf, class_id]
                        # This structure is common for Ultralytics YOLO models.
                        for detection in results[0].boxes.data:
                            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()
                            
                            x_scale = original_width / 640.0
                            y_scale = original_height / 640.0
                            
                            x1_orig = int(x1 * x_scale)
                            y1_orig = int(y1 * y_scale)
                            x2_orig = int(x2 * x_scale)
                            y2_orig = int(y2 * y_scale)
                            
                            cls_id_int = int(class_id)
                            if 0 <= cls_id_int < len(CLASS_NAMES):
                                label = CLASS_NAMES[cls_id_int]
                            else:
                                label = "Unknown" # Fallback for unexpected class_id
                            
                            text_to_display = f'{label} {float(confidence):.2f}'
                            
                            cv2.rectangle(output_image_to_draw_on, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
                            cv2.putText(output_image_to_draw_on, text_to_display, (x1_orig, y1_orig - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Saving and Context Update
                        fn_without_ext = img_filename_from_list.split('.')[0]
                        # Ensure consistent extension, e.g., jpeg
                        r_filename = f'result_{fn_without_ext}.jpeg' 
                        
                        cv2.imwrite(str(os.path.join(res_f_root, r_filename)), output_image_to_draw_on)
                        
                        r_media_filepath = Path(settings.MEDIA_URL) / 'Result' / r_filename
                        
                        print(f"Processed: {img_filename_from_list}, Result saved as: {r_filename}")

                        with open(Path(settings.MEDIA_ROOT) / 'Result' / 'Result.txt', 'a') as f_results_txt:
                            f_results_txt.write(str(r_media_filepath))
                            f_results_txt.write("\n")
                        
                        # Use web-accessible path for uploaded images in context
                        uploaded_media_filepath = Path(settings.MEDIA_URL) / 'uploadedPics' / img_filename_from_list
                        context["my_uploaded_file_names"].append(str(uploaded_media_filepath))
                        context["my_result_file_names"].append(str(r_media_filepath))

                    except Exception as e:
                        print(f"Error processing file {file_line.strip()}: {e}")
                        error_img_path = str(Path(settings.MEDIA_URL) / 'uploadedPics' / file_line.strip().split('/')[-1])
                        context["my_uploaded_file_names"].append(error_img_path + " (Processing Error)")
            
            return render(request, "cam_app2/image.html", context)

        if (request.FILES and emptyButtonFlag == False):
            print("reached here files")
            reset()
            context = self.reset_context(request)
            context["my_uploaded_file_names"] = []
            for file_obj in request.FILES.getlist("file_data"):
                uuidStr = uuid.uuid4()
                filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                with default_storage.open(Path(f"uploadedPics/{filename}"), 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
                filename = Path(f"{settings.MEDIA_URL}uploadedPics/{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}")
                with open(Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'), 'a') as f:
                    f.write(str(filename))
                    f.write("\n")

                context["my_uploaded_file_names"].append(str(f'{str(filename)}'))
            return render(request, "cam_app2/image.html", context)
        context = self.reset_context(request)
        reset()
        return render(request, "cam_app2/image.html", {'page': self})
