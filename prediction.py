import cv2
import os
import numpy as np
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.models import load_model

class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)
    
    
model = load_model('siamese_model.h5',
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
def preprocess_file(file_path):
    byte_img = tf.io.read_file(file_path)#read in image from file path
    img = tf.io.decode_jpeg(byte_img)# load in img
    img = tf.image.resize(img, (100, 100))
    img = img /255.0
    return img

def preprocess_img(img):
    img = tf.image.resize(img, (100, 100))
    img = img/255.0
    return img

def verification(model , detection_threshold, verification_threshold, input_img):
    results=[]
    for image in os.listdir(os.path.join("application data","verification_images")):
        validation_img = preprocess_file(os.path.join('application data', 'verification_images', image))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
        
    detection = np.sum(np.array(results)>detection_threshold)
    verification = detection/len(os.path.join("application data", 'verification_images'))
    
    verified  =verification >verification_threshold
    return results, verified

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face
        face = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        input_img = preprocess_img(face_rgb)

        # Trigger verification when 'v' is pressed
        if cv2.waitKey(10) & 0xFF == ord('v'):
            results, verified = verification(model, 0.9, 0.7, input_img)

            # Display verification result on the rectangle
            if verified:
                cv2.putText(frame, 'Verified', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Not Verified', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Verification', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()