import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pretrained face mask detection model
# (Make sure you have a trained "mask_detector.model" file or download one)
model = load_model("/home/robin/Projects/Face-Mask-Detection/model/mask_detector.h5")
print("Model input shape:", model.input_shape)


# Load class labels
classes = ["Mask", "No Mask"]

# Load Haar Cascade for face detection (pretrained by OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to detect masks in an image
def detect_mask(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = img[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (100,100))  # Resize to model input size
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize

        # Predict mask / no mask
        prediction = model.predict(face)[0]
        label = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Draw results on the image
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(img, f"{label}: {confidence:.2f}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Show the output
    cv2.imshow("Face Mask Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_mask("/home/robin/Downloads/man.jpg")
