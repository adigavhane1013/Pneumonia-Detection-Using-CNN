Pneumonia Detection from Chest X-rays using Convolutional Neural Networks (CNN)
This project implements a complete pipeline for automatic pneumonia detection from chest X-ray images. It employs an effective CNN architecture combined with image preprocessing and data augmentation to accurately classify X-rays as Pneumonia or Normal. An interactive GUI allows intuitive image uploading and visualization of diagnosis results.

Project Overview
Pneumonia diagnosis from chest X-rays is critical in healthcare and requires expert evaluation. This project automates this process by training a CNN on labeled X-ray images, facilitating fast and reliable pneumonia screening. The trained model is integrated with a user-friendly GUI for real-time image classification.

Dataset
The project uses the publicly available Kaggle chest X-ray dataset, structured with:

PNEUMONIA/ folder containing pneumonia-positive X-ray images.

NORMAL/ folder containing healthy lung X-ray images.

Dataset link:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Key Features
Data preprocessing: Resizes to 150x150 pixels and normalizes grayscale X-ray images.

CNN model: Two convolutional layers with increasing filters, max-pooling, ReLU activations, followed by fully connected dense layers, optimized for binary classification.

Data Augmentation (optional enhancement): Techniques like rotation, zoom, shear, and horizontal flipping to improve model generalization.

Training: Uses an 80-20 train-validation split for performance evaluation.

Model saving: Saves the trained model (.h5 format) for later inference.

GUI Interface: Enables image upload, preview, and instantaneous inference results display.

Technologies and Libraries
Python 3.x

TensorFlow and Keras for CNN modeling

OpenCV for image processing

Pillow (PIL) for GUI image handling

Tkinter for GUI development

NumPy for numerical operations

Model Architecture
Input: 150x150 grayscale X-ray images.

Conv2D Layers: 32 and 64 filters respectively, with ReLU activations.

MaxPooling: Downsamples feature maps to reduce spatial dimensions.

Dense Layers: Fully connected layers, final layer with sigmoid activation for binary output.

Loss Function: Binary crossentropy.

Optimizer: Adam.

Evaluation Metric: Accuracy.

Setup Instructions
Clone or download the project repository.

Install dependencies:

bash
pip install tensorflow opencv-python pillow numpy
Download and organize the dataset in the data folder with PNEUMONIA and NORMAL subfolders.

Train the model by running:

bash
python train_model.py
Run the GUI interface for pneumonia detection:

bash
python gui_app.py
Results and Evaluation
The model trains for 10 epochs with validation split.

Accuracy and loss monitored during training for overfitting control.

Final saved model can predict pneumonia with reliable accuracy on chest X-rays.

GUI displays classification as PNEUMONIA or NORMAL with corresponding uploaded X-ray preview.

Usage
Medical practitioners can use the GUI tool to quickly screen chest X-rays.

Researchers can extend the model with larger datasets or advanced CNN architectures.

Adaptable to other medical image classification tasks with modifications.
