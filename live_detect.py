import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pretrained model
model = load_model("/home/robin/Projects/Face-Mask-Detection/model/mask_detector.h5")
classes = ["Mask", "No Mask"]

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Get model input size dynamically
target_size = model.input_shape[1:3]  # (height, width)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract and preprocess face ROI
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        face = np.expand_dims(face, axis=0) / 255.0

        prediction = model.predict(face, verbose=0)[0]
        label = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Draw bounding box & label
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(frame, f"{label} {confidence:.1f}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Show video
    cv2.imshow("Live Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
