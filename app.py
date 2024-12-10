import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("CNN_model.h5")

# Helper function for preprocessing the image
def preprocess_image(image, target_size=(255, 255)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit app
st.title("Pneumonia Detection App")

# Input section
st.header("Upload an X-ray Image")
uploaded_file = st.file_uploader("Choose an X-ray image file", type=["jpg", "jpeg", "png"])

# Output section
st.header("Prediction Output")

if uploaded_file:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        prediction_label = np.argmax(prediction, axis=1)
        confidence = np.max(prediction) * 100

        # Display prediction
        if prediction_label[0] == 1:
            st.error(f"Signs of pneumonia detected (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"No signs of pneumonia detected (Confidence: {confidence:.2f}%)")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
