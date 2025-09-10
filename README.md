# Automated-Brain-tumor-detection-using-Deep-Learning
# 🧠 Brain Tumor Detection using Deep Learning

A **deep learning-based web application** for detecting brain tumors from MRI images using **transfer learning** with **ResNet18**, built with **PyTorch** and deployed via **Streamlit**. This project aims to assist medical professionals and patients by providing a quick, interpretable, and reliable tool for identifying brain tumors from medical scans.

---

## 📖 Project Overview

Brain tumors are a critical medical condition requiring early diagnosis for better treatment outcomes. Manual diagnosis from MRI images can be time-consuming and prone to human error. This project leverages deep learning techniques to automate tumor detection, providing an efficient and accurate solution.

The model is trained on labeled MRI images to classify whether a scan shows the presence of a brain tumor or not. It also incorporates **Grad-CAM** visualizations to explain the model’s predictions, making it more transparent and interpretable.

---

## 🚀 Features

- ✅ **Binary Classification** – Classifies MRI images into "Tumor" and "No Tumor"
- ✅ **Transfer Learning** – Utilizes **ResNet18** pretrained on ImageNet for better performance
- ✅ **Streamlit Interface** – Simple and interactive web-based UI for users to upload images and view results
- ✅ **Grad-CAM Visualization** – Highlights areas of the image that influence the model’s decision
- ✅ **PDF Report Generation** – Generates a detailed report with patient information, prediction results, and Grad-CAM heatmaps
- ✅ **Real-time Predictions** – Fast and efficient inference for immediate results

---

## 📂 Dataset

- The dataset consists of labeled brain MRI images categorized into "Tumor" and "No Tumor".
- Data preprocessing includes resizing, normalization, and augmentation techniques to improve model generalization.

---

## 🛠 Tech Stack

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Model Architecture:** ResNet18 (Transfer Learning)
- **Web Framework:** Streamlit
- **Visualization:** Matplotlib, OpenCV (for Grad-CAM heatmaps)
- **PDF Report:** ReportLab / Python libraries
- **Deployment:** Local or cloud-based Streamlit sharing

---

## 📦 Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py

brain-tumor-detection/
├── data/               # MRI dataset
├── models/             # Trained models
├── outputs/            # Grad-CAM and reports
├── app.py              # Streamlit app
├── utils.py            # Utility functions
├── requirements.txt    # Python dependencies
├── README.md           # Project description
