import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        text-align: center;
        line-height: 20px;
        color: white;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classifier</h1>
    <p>Upload a satellite image to classify it into one of four categories: Cloudy, Desert, Green Area, or Water</p>
</div>
""", unsafe_allow_html=True)

# Class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Class descriptions
class_descriptions = {
    'Cloudy': '☁️ Cloud formations and overcast areas',
    'Desert': '🏜️ Arid and sandy terrain',
    'Green_Area': '🌿 Vegetation, forests, and green landscapes',
    'Water': '🌊 Water bodies, rivers, lakes, and oceans'
}

# Function to load model
@st.cache_resource
def load_trained_model():
    """Load the trained model. For deployment, you'll need to have the model file."""
    try:
        # First, try to load from the same directory
        if os.path.exists('Modelenv.v1.h5'):
            model = load_model('Modelenv.v1.h5')
            return model
        else:
            st.error("Model file not found. Please ensure 'Modelenv.v1.h5' is in the same directory as this script.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess image
def preprocess_image(uploaded_image):
    """Preprocess the uploaded image for model prediction."""
    try:
        # Open image
        img = Image.open(uploaded_image)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((255, 255))
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array, img
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

# Function to make prediction
def predict_image(model, processed_image):
    """Make prediction on the processed image."""
    try:
        prediction = model.predict(processed_image)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
        
        return predicted_class, confidence, class_probabilities
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# Function to display confidence bars
def display_confidence_bars(probabilities):
    """Display confidence bars for all classes."""
    st.subheader("Confidence Scores for All Classes:")
    
    for class_name, probability in probabilities.items():
        percentage = probability * 100
        st.write(f"**{class_name}**: {percentage:.2f}%")
        
        # Create a colored bar
        bar_html = f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {percentage}%;">
                {percentage:.1f}%
            </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

# Main app logic
def main():
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a satellite image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a satellite image to classify it"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.subheader("Uploaded Image:")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            processed_image, display_image = preprocess_image(uploaded_file)
            
            if processed_image is not None:
                # Make prediction
                predicted_class, confidence, probabilities = predict_image(model, processed_image)
                
                if predicted_class is not None:
                    with col2:
                        st.header("Prediction Results")
                        
                        # Display main prediction
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Predicted Class: {predicted_class}</h3>
                            <p>{class_descriptions[predicted_class]}</p>
                            <h4>Confidence: {confidence*100:.2f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display confidence bars
                        display_confidence_bars(probabilities)
    
    # Add information section
    with st.expander("ℹ️ About this classifier"):
        st.write("""
        This satellite image classifier uses a Convolutional Neural Network (CNN) to categorize satellite images into four classes:
        
        - **Cloudy**: Images showing cloud formations and overcast areas
        - **Desert**: Images of arid and sandy terrain
        - **Green Area**: Images showing vegetation, forests, and green landscapes
        - **Water**: Images of water bodies including rivers, lakes, and oceans
        
        The model was trained on a dataset of satellite images and uses deep learning techniques to achieve accurate classification.
        """)
    
    # Add sample images section
    with st.expander("📸 Sample Images"):
        st.write("Here are some examples of the types of images the classifier can identify:")
        
        sample_cols = st.columns(4)
        sample_classes = ['Cloudy', 'Desert', 'Green_Area', 'Water']
        
        for i, class_name in enumerate(sample_classes):
            with sample_cols[i]:
                st.write(f"**{class_name}**")
                st.write(class_descriptions[class_name])

# Run the app
if __name__ == "__main__":
    main()