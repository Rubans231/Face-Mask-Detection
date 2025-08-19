# gui_detect.py
import os
import cv2
import numpy as np
from tkinter import Tk, Label, Frame, filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES  # drag-and-drop support
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import tkinter as tk

# Configuration
IMG_SIZE = 100
MODEL_PATH = 'model/mask_detector.h5'
LABELS = ['Without Mask', 'With Mask']
COLORS = [(0, 0, 255), (0, 255, 0)]  # BGR: Red for no mask, Green for mask

# Load model
print("Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

# Load model
model = load_model(MODEL_PATH)

# Re-compile the model to build metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------ GUI Application ------------------
class MaskDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Detector - Drag & Drop")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Instructions
        self.label = Label(
            root,
            text="🔽 Drag and drop an image here 🔽\n\nSupports: .png, .jpg, .jpeg",
            bg="#e0e0e0",
            fg="#333",
            font=("Helvetica", 14),
            relief="solid",
            width=50,
            height=6
        )
        self.label.pack(padx=20, pady=20)

        # Image display area
        self.image_frame = Frame(root, bg="white", width=700, height=400)
        self.image_frame.pack(padx=20, pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = Label(self.image_frame, bg="white")
        self.image_label.pack()

        # Register drag and drop
        self.label.drop_target_register(DND_FILES)
        self.label.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, event):
        file_path = event.data

        # Clean up path: remove braces if present
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]  # Remove { and }

        # Trim whitespace
        file_path = file_path.strip()

        # Validate image file
        if not os.path.isfile(file_path):
            self.show_message("File not found!")
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg']:
            self.show_message("Not an image file!\nPlease drop a .png, .jpg, or .jpeg")
            return

        self.process_image(file_path)

    def process_image(self, image_path):
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            self.show_message("Could not read image.")
            return

        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Predict mask for each face
        for (x, y, w_face, h_face) in faces:
            face_crop = img_rgb[y:y+h_face, x:x+w_face]
            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            pred = model.predict(face_input)[0]
            class_id = np.argmax(pred)
            label = LABELS[class_id]
            color = COLORS[class_id]
            confidence = f"{pred[class_id]*100:.1f}%"

            # Draw bounding box and label on image
            cv2.rectangle(img_rgb, (x, y), (x + w_face, y + h_face), color[::-1], 2)  # OpenCV uses BGR
            cv2.putText(img_rgb, f"{label} ({confidence})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[::-1], 2)

        # Resize image for display if too large
        max_size = (700, 400)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Display image
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img  # Keep reference

        # Update status
        num_faces = len(faces)
        status = f"Detected {num_faces} face(s): "
        mask_count = sum(1 for f in faces if np.argmax(model.predict(np.expand_dims(np.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[f[1]:f[1]+f[3], f[0]:f[0]+f[2]], (IMG_SIZE, IMG_SIZE)) / 255.0, (1, IMG_SIZE, IMG_SIZE, 3)))) == 1)
        status += f"{mask_count} with mask, {num_faces - mask_count} without"
        self.label.config(text=status)

    def show_message(self, msg):
        self.label.config(text=msg)


# ------------------ Main ------------------
if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please run train.py first.")
        exit(1)

    # Use TkinterDnD (enhanced Tkinter with drag-and-drop)
    root = TkinterDnD.Tk()
    app = MaskDetectionApp(root)
    root.mainloop()