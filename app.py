import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import os

# Define L1Dist layer for the Siamese network
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)

# Load the Siamese model
model = load_model('siamese_model.h5',
                   custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

# Preprocessing functions
def preprocess_file(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

def preprocess_img(img):
    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    return img.astype('float32')

# Verification function
def verification(model, detection_threshold, verification_threshold, input_img):
    results = []
    for image in os.listdir(os.path.join("application data", "verification_images")):
        validation_img = preprocess_file(os.path.join('application data', 'verification_images', image))
        result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold
    return results, verified

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit app configuration
st.title("Real-Time Face Verification with Capture")
st.text("Click 'Start' to begin the webcam feed, 'Capture' to save a snapshot, and 'Stop' to end the session.")

# Session state to manage the webcam feed
if 'running' not in st.session_state:
    st.session_state.running = False

# Placeholder for the webcam feed
FRAME_WINDOW = st.image([])
captured_msg = st.empty()

# Button functionality
if st.button("Start"):
    st.session_state.running = True

if st.button("Stop"):
    st.session_state.running = False

if 'capture' not in st.session_state:
    st.session_state.capture = False

if st.button("Capture"):
    st.session_state.capture = True

# Real-time webcam feed
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop and preprocess the face
            face = frame[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            input_img = preprocess_img(face_rgb)

            # Perform verification
            results, verified = verification(model, 0.9, 0.7, input_img)

            # Display verification result
            if verified:
                cv2.putText(frame, 'Verified', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if st.session_state.capture:
                    # Save the verified face to a file
                    os.makedirs("captured_faces", exist_ok=True)
                    save_path = os.path.join("captured_faces", "captured_face.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    captured_msg.success(f"Captured and saved at {save_path}")
                    st.session_state.capture = False
            else:
                cv2.putText(frame, 'Not Verified', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert the frame to RGB and display it in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    # Release the webcam once stopped
    cap.release()
    st.session_state.running = False
else:
    st.write("Webcam feed is not running.")