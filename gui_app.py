import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('saved_model/pneumonia_detection_model_best.h5')

IMG_SIZE = 150

# Prediction function returns label and probability
def predict_pneumonia(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        return None, None
    resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_reshaped = np.reshape(resized_img, (1, IMG_SIZE, IMG_SIZE, 1))
    prediction = model.predict(img_reshaped)[0][0]
    label = 'PNEUMONIA' if prediction > 0.5 else 'NORMAL'
    prob = prediction if prediction > 0.5 else 1 - prediction
    return label, prob

# Upload function for GUI
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[('Image Files', '*.png *.jpg *.jpeg *.bmp *.tiff')])
    if file_path:
        try:
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            label_image.config(image=img_tk)
            label_image.image = img_tk
            label_result.config(text='Predicting...', fg='black')
            root.update_idletasks()

            # Predict
            label, prob = predict_pneumonia(file_path)
            if label is None:
                label_result.config(text='Error loading image.', fg='red')
            else:
                color = 'red' if label == 'PNEUMONIA' else 'green'
                label_result.config(text=f'Result: {label} (Confidence: {prob:.2f})', font=('Arial', 16, 'bold'), fg=color)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to process image: {e}')

# Set up the main GUI window
root = tk.Tk()
root.title('Pneumonia Detection System')
root.geometry('450x600')
root.configure(bg='#f0f0f0')

# Frames
frame_top = tk.Frame(root, bg='#f0f0f0')
frame_top.pack(pady=20)

label_instruction = tk.Label(frame_top, text='Upload an X-ray image to check for pneumonia', font=('Arial', 16), bg='#f0f0f0')
label_instruction.pack()

button_upload = tk.Button(frame_top, text='Upload X-ray Image', command=upload_image, font=('Arial', 14), bg='#4CAF50', fg='white', padx=10, pady=5)
button_upload.pack(pady=15)

frame_image = tk.Frame(root, bg='#f0f0f0', bd=2, relief='sunken')
frame_image.pack(pady=10)

label_image = tk.Label(frame_image, bg='#f0f0f0')
label_image.pack()

label_result = tk.Label(root, text='', font=('Arial', 14), bg='#f0f0f0')
label_result.pack(pady=20)

root.mainloop()
