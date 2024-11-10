import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('/kaggle/input/testing-model/skin_disease_model_mobilenet.h5')

class_labels = [
    'Eczema',
    'Warts Molluscum and other Viral Infections',
    'Melanoma',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma (BCC)',
    'Melanocytic Nevi (NV)',
    'Benign Keratosis-like Lesions (BKL)',
    'Psoriasis',
    'Seborrheic Keratoses',
    'Tinea Ringworm Candidiasis'
]

# Define a function to preprocess a new image
def preprocess_image(img_path, target_size=(224, 224)):
    """
    Loads an image, resizes it to the required input size, and normalizes pixel values.
    """
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Define a function to make predictions on a new image
def predict_skin_disease(img_path, class_labels, threshold=30):
    """
    Predicts the class of the skin disease in the provided image with a confidence threshold.
    """
    img_array = preprocess_image(img_path)
    if img_array is None:
        return None, None  # Return early if image preprocessing failed

    predictions = model.predict(img_array)  # Get model predictions
    class_index = np.argmax(predictions)  # Find index of the highest probability
    confidence = predictions[0][class_index] * 100  # Convert to percentage

    # Check if confidence meets the threshold
    if confidence < threshold:
        return "Uncertain or Normal", confidence
    else:
        predicted_class = class_labels[class_index]
        return predicted_class, confidence

# Path to the test image
img_path = '/kaggle/input/testjnj/IMG_7438.jpg'

# Make a prediction and print the result
predicted_class, confidence = predict_skin_disease(img_path, class_labels)
if predicted_class:
    print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}%")
else:
    print("Prediction failed. Please check the image path and preprocessing.")
