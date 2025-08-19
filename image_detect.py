import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load pretrained model
model = load_model("/home/robin/Projects/Face-Mask-Detection/model/mask_detector.h5")
classes = ["Mask", "No Mask"]

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# GUI window
root = tk.Tk()
root.title("Face Mask Detection")
root.geometry("800x600")

label = Label(root, text="Upload an image for mask detection", font=("Arial", 16))
label.pack(pady=10)

panel = Label(root)
panel.pack()

def detect_mask(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    target_size = model.input_shape[1:3]

    for (x, y, w, h) in faces:
        face = cv2.resize(img[y:y+h, x:x+w], target_size)
        face = np.expand_dims(face, axis=0) / 255.0

        prediction = model.predict(face)[0]
        label = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(img, f"{label} {confidence:.1f}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return

    result_img = detect_mask(file_path)

    # Convert to Tkinter-compatible image
    img = Image.fromarray(result_img)
    img = img.resize((600, 400))  # Resize for display
    imgtk = ImageTk.PhotoImage(image=img)

    panel.config(image=imgtk)
    panel.image = imgtk

btn = Button(root, text="Upload Image", command=upload_image, font=("Arial", 14), bg="blue", fg="white")
btn.pack(pady=20)

root.mainloop()
