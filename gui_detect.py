# gui_detect.py
import os
import cv2
import numpy as np
from tkinter import Label, Frame
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

model = load_model(MODEL_PATH)

# Compile to ensure metrics available (optional for inference, but keeps metrics defined)
try:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception:
    # Some saved models may not recompile cleanly; ignore since we're only predicting
    pass

# Load OpenCV's Haar Cascade for face detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(face_cascade_path):
    raise FileNotFoundError(f"Haar cascade not found at: {face_cascade_path}")
face_cascade = cv2.CascadeClassifier(face_cascade_path)

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
        # event.data can be a Tcl list: use splitlist to handle spaces/braces/multiple files
        paths = self.root.tk.splitlist(event.data)
        if not paths:
            self.show_message("No file received.")
            return

        file_path = paths[0].strip()  # Use first file only

        if not os.path.isfile(file_path):
            self.show_message("File not found!")
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg']:
            self.show_message("Not an image file!\nPlease drop a .png, .jpg, or .jpeg")
            return

        self.process_image(file_path)

    def process_image(self, image_path):
        # Read image with OpenCV (BGR)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            self.show_message("Could not read image.")
            return

        # Convert BGR to RGB for display/drawing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Detect faces (on grayscale)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            # Show the image as-is and message
            self._display_image(img_rgb)
            self.label.config(text="No faces detected.")
            return

        # Prepare batch for model
        face_inputs = []
        face_boxes = []
        for (x, y, w_face, h_face) in faces:
            # Clip to bounds (robustness)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(img_rgb.shape[1], x + w_face), min(img_rgb.shape[0], y + h_face)
            face_crop = img_rgb[y0:y1, x0:x1]
            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_inputs.append(face_normalized)
            face_boxes.append((x, y, w_face, h_face))

        if not face_inputs:
            self._display_image(img_rgb)
            self.label.config(text="No valid face crops found.")
            return

        batch = np.stack(face_inputs, axis=0)  # shape: (N, IMG_SIZE, IMG_SIZE, 3)

        # Predict in one go (fast)
        preds = model.predict(batch, verbose=0)  # shape: (N, num_classes)

        # Draw results
        mask_count = 0
        for (x, y, w_face, h_face), pred in zip(face_boxes, preds):
            class_id = int(np.argmax(pred))
            label = LABELS[class_id]
            color_bgr = COLORS[class_id]
            confidence = f"{pred[class_id] * 100:.1f}%"

            # We are drawing on an RGB image; cv2 expects BGR colors.
            # Convert BGR -> RGB for correct color on RGB image by reversing.
            color_rgb = color_bgr[::-1]

            cv2.rectangle(img_rgb, (x, y), (x + w_face, y + h_face), color_rgb, 2)
            cv2.putText(img_rgb, f"{label} ({confidence})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)

            if class_id == 1:
                mask_count += 1

        # Show image
        self._display_image(img_rgb)

        # Update status (reuse computed results)
        num_faces = len(face_boxes)
        status = f"Detected {num_faces} face(s): {mask_count} with mask, {num_faces - mask_count} without"
        self.label.config(text=status)

    def _display_image(self, img_rgb):
        # Resize image for display if too large
        max_size = (700, 400)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img  # Keep reference

    def show_message(self, msg):
        self.label.config(text=msg)


# ------------------ Main ------------------
if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please run train.py first.")
        raise SystemExit(1)

    # Use TkinterDnD (enhanced Tkinter with drag-and-drop)
    root = TkinterDnD.Tk()
    app = MaskDetectionApp(root)
    root.mainloop()
