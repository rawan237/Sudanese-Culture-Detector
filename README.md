\# Sudanese Culture Detector



A unified web application for detecting Sudanese food items and clothing items using YOLO object detection models, built with FastAPI.

## 🚀 Live Demo
Try the app here: http://16.192.170.210

\## Features



\- User authentication (signup/login) with hashed passwords

\- Two detection models in one app:

&#x20; - \*\*Food Detector\*\*: detects zalabia, cay, mol5iya

&#x20; - \*\*Cloth Detector\*\*: detects traditional Sudanese clothing items

\- Bounding box visualization with confidence scores

\- Automatic logging of detection results (with coordinates) to JSON



\## Tech stack



\- \*\*Backend\*\*: FastAPI + Uvicorn

\- \*\*ML Models\*\*: YOLOv11 (Ultralytics), trained on custom datasets

\- \*\*Image processing\*\*: OpenCV

\- \*\*Authentication\*\*: Passlib (bcrypt hashing)



\## Project structure
├── app.py # Main FastAPI application

├── Sudanese-food-detection.pt # Food detection model

├── best.pt # Cloth detection model

├── requirements.txt # Python dependencies

├── uploaded\_images/ # Stores uploaded + annotated images

├── detection\_results.json # Log of all detections (auto-generated)

└── users.json # User accounts (auto-generated, not tracked in git)

## Setup and installation



1\. Clone the repository:

```bash

git clone https://github.com/rawan237/Sudanese-Culture-Detector.git

cd Sudanese-Culture-Detector

```



2\. Install dependencies:

```bash

pip install -r requirements.txt

```



3\. Run the server:

```bash

uvicorn app:app --reload

```



\## Usage



1\. Create an account on the signup page, or log in if you already have one

2\. Choose a detector (Food or Cloth) from the menu

3\. Upload an image

4\. View the detected items with bounding boxes and confidence scores



\## Model training



Both models were trained using YOLOv11 on custom-labeled datasets:

\- Images collected and annotated manually via MakeSense.ai

\- Data split: 70% train / 20% validation / 10% test

\- Training performed on Google Colab with GPU acceleration



\## Authors



\- Rawan237 — Food detection model

\- Afraalkhider — Cloth detection model

