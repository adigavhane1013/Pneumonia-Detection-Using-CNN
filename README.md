# 🩺 Pneumonia Detection Using CNN

A complete machine learning pipeline for automatic pneumonia detection from chest X-ray images using **Convolutional Neural Networks (CNN)**. This project combines robust image preprocessing, a highly accurate CNN model, and an interactive GUI for practical medical screening.

---

## 🚀 Features

- End-to-end pneumonia classification from chest X-rays  
- User-friendly GUI for image upload and instant diagnosis  
- CNN with two convolutional layers, max-pooling, and fully connected layers  
- Data augmentation for better generalization  
- Model training and evaluation with saved models for inference  

---

## 📂 Project Structure

Pneumonia-Detection-Using-CNN/
├── train_model.py # CNN model training and evaluation script
├── gui_app.py # GUI application for image upload and inference
├── README.md # This documentation file
├── data/ # Dataset folder
│ ├── PNEUMONIA/ # Pneumonia-positive X-ray images
│ └── NORMAL/ # Healthy lung X-ray images
└── saved_model.h5 # Trained CNN model (generated after training)

text

---

## 🧐 Project Overview

Pneumonia diagnosis from chest X-rays is critical yet challenging and time-consuming for healthcare professionals.  
This project accelerates screening and increases consistency by training a CNN on labeled X-ray images.  
The GUI enables fast image upload and diagnosis, providing instant feedback on whether an X-ray is classified as **PNEUMONIA** or **NORMAL**.

---

## 📈 Dataset

- Public Kaggle chest X-ray dataset:  
  [Kaggle - Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- Dataset contains:
  - `PNEUMONIA/` - Pneumonia-positive X-rays  
  - `NORMAL/` - Healthy X-rays  

---

## ⚙️ Technologies Used

- Python 3.x  
- TensorFlow and Keras for deep learning  
- OpenCV for image preprocessing  
- Pillow (PIL) for image handling in GUI  
- Tkinter for GUI development  
- NumPy for numerical operations  

---

## 🏗️ Model Architecture

- Input images resized to 150x150 pixels and normalized to grayscale  
- Conv2D Layers with 32 and 64 filters; ReLU activations  
- MaxPooling to reduce spatial dimensions  
- Fully connected dense layers with sigmoid output for binary classification  
- Binary crossentropy loss, Adam optimizer, accuracy metric  
- Optional data augmentation: rotation, zoom, shear, flipping  

---

## 🛠️ Setup Instructions

1. Clone the repository and navigate to project folder:  
git clone https://github.com/<your-username>/Pneumonia-Detection-Using-CNN.git
cd Pneumonia-Detection-Using-CNN

text

2. Install the required packages:  
pip install tensorflow opencv-python pillow numpy

text

3. Download the Kaggle chest X-ray dataset and organize under `data/`:  
data/
├── PNEUMONIA/
└── NORMAL/

text

4. Train the model:  
python train_model.py

text

5. Run the GUI app for pneumonia detection:  
python gui_app.py

text

---

## 📊 Results & Evaluation

- Model achieves reliable pneumonia detection after training for 10 epochs  
- Validation accuracy and loss monitored to avoid overfitting  
- GUI displays classification results immediately upon image upload  

---

## 💡 Usage

- Medical professionals can use the GUI to screen chest X-rays quickly  
- Researchers can extend the model with other datasets or architectures  
- Adaptable for other medical image classification tasks with minor modifications  

---

## 👤 Author

Aditya Gavhane

---

## 📄 License

This project is licensed under the MIT License.
