import streamlit as st
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from PIL import Image
import cv2

# Load the digits dataset
digits = load_digits()
X = digits.data / 16.0  # Normalize input
y = digits.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the neural network
model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500)
model.fit(X_train, y_train)

# Function to preprocess the image
def preprocess_image(image):
    try:
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((8, 8))  # Resize to 8x8 pixels as in the dataset
        image = np.array(image) / 16.0  # Normalize (as the dataset uses 16 grayscale values)
        image = image.flatten().reshape(1, -1)  # Flatten and reshape
        return image
    except Exception as e:
        st.error("Error processing image.")
        return None

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
    
    if processed_image is not None:
        prediction = model.predict(processed_image)
        predicted_digit = prediction[0]
        st.write(f'Predicted Digit: {predicted_digit}')
    else:
        st.error("Failed to preprocess the image.")

