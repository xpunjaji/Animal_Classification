import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("animal_classification_model.h5")  # Update with your model path

# Define class labels (Update based on your dataset)
class_labels = ['antelope', 'bear', 'beaver', 'bee', 'bison', 'blackbird', 'buffalo', 'butterfly', 
                'camel', 'cat', 'cheetah', 'chimpanzee', 'chinchilla', 'cow', 'crab', 'crocodile', 
                'deer', 'dog', 'dolphin', 'donkey', 'duck', 'eagle', 'elephant',
                'falcon', 'ferret', 'flamingo', 'fox', 'frog', 'giraffe', 'goat', 'goose', 'gorilla', 
                'grasshopper', 'hawk', 'hedgehog', 'hippopotamus', 'hyena', 'iguana', 'jaguar', 'kangaroo',
                'koala', 'lemur', 'leopard', 'lizard', 'lynx', 'mole', 
                'mongoose', 'ostrich', 'otter', 'owl', 'panda', 'peacock', 'penguin', 'porcupine', 
                'raccoon', 'seal', 'sheep', 'snail', 'snake', 'spider', 'squid', 'walrus', 'whale', 'wolf']

def predict_image(image):
    # Convert image to OpenCV format
    img = np.array(image)
    img_resized = cv2.resize(img, (150, 150)) / 255.0  # Resize & normalize
    img_expanded = np.expand_dims(img_resized, axis=0)
    
    # Make prediction
    prediction = model.predict(img_expanded)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return class_labels[predicted_class], confidence

# Streamlit UI
st.title("Animal Classification App")
st.write("Upload an image of an animal to get its classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    label, confidence = predict_image(image)
    
    # Display result
    st.success(f"Prediction: {label} ({confidence*100:.2f}% confidence)")