import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the saved model
model = load_model('ann_model.keras')

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((8, 8))  # Resize to 8x8 pixels as in the dataset
    image = np.array(image) / 16.0  # Normalize (as the dataset uses 16 grayscale values)
    image = image.flatten().reshape(1, -1)  # Flatten and reshape
    return image

# Streamlit app interface
st.title("Digit Recognition App")

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    st.write(f'Predicted Digit: {predicted_digit}')