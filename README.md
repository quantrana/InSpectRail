# InSpectRail 🚆🔍

**AI-Powered Railway Track Defect Detection System**

InSpectRail is a Django + Wagtail-based web application that leverages real-time computer vision model to detect railway track defects. It allows users to upload images or stream video input and receive immediate defect classification.

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
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies 
```bash
pip install -r requirements.txt
```

### 4. Run migrations
```bash
cd mysite
python manage.py migrate
```
### 5. Start the development server
```bash
python manage.py runserver
```




🙏 Acknowledgements

This project builds upon the foundational work provided in the GitarthVaishnav/Basic_Web_App repository, which was made available as part of the Deep Learning workshop. It served as a valuable starting point for developing our custom YOLOv11-based railway defect detection application.


🧑‍💻 Contributors

Anusha Pariti
Anh Quan Tran
Nicole Hutomo


📜 License

This project, "InSpectRail – Railway Track Defect Detection Using YOLOv11s", was developed as part of the academic coursework for subject **42028: Deep Learning and Convolutional Neural Networks**.

Authors: Anusha Pariti, Anh Quan Tran, Nicole Hutomo  
Project Number: 64 | Team Name: Deep Visionaries

Permission is hereby granted to UTS faculty, students, and academic reviewers to view, evaluate, and reproduce this work for educational or assessment purposes only.

Any commercial use, redistribution, or derivative works outside of academic review is strictly prohibited without explicit written consent from the authors and the University of Technology Sydney.

This codebase and accompanying materials are provided **as-is**, with no warranties expressed or implied.

For inquiries, contact: [wuang.work@gmail.com] [anusha.pariti@student.uts.edu.au] [nicole.f.hutomo@student.uts.edu.au]

