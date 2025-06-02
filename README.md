# InSpectRail 🚆🔍

**AI-Powered Railway Track Defect Detection System**

InSpectRail is a Django + Wagtail-based web application that leverages real-time computer vision to detect railway track defects using YOLOv11s. It allows users to upload images or stream video input and receive immediate defect classification.

---

## 🚀 Features

- 🧠 Integrated YOLOv11s object detection model
- 📷 Real-time video inference and image upload support
- 🖼️ Bounding box visualization for detected defects
- 🕵️ Identifies cracks, flakings, squats, and joints
- 🛠️ Built with Django, Wagtail, OpenCV, and PyTorch
- 💻 Clean, interactive UI for users to inspect results

---

## 📂 Folder Structure

InSpectRail/
├── model/ # YOLO model weights and configurations
├── static/ # Static CSS, JS, images
├── templates/ # HTML templates
├── mysite/ # Main Django project
│ ├── settings.py
│ ├── urls.py
│ └── views.py
├── manage.py # Django entrypoint
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🧪 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/quantrana/InSpectRail.git
cd InSpectRail

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
bash```

### 2. Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
