import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Set page config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Helper functions
@st.cache_resource
def load_trained_model():
    """Load the trained model"""
    try:
        model = load_model("Modelenv.v1.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    """Preprocess image for prediction"""
    img = img.resize((255, 255))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    return predicted_class, confidence, prediction[0]

def create_confidence_chart(predictions, class_names):
    """Create a confidence chart using plotly"""
    fig = px.bar(
        x=class_names,
        y=predictions,
        title="Prediction Confidence for Each Class",
        labels={'x': 'Classes', 'y': 'Confidence Score'},
        color=predictions,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        title_font_size=16
    )
    return fig

def display_sample_images():
    """Display sample images from each class"""
    st.markdown('<div class="sub-header">📸 Sample Images from Each Class</div>', unsafe_allow_html=True)
    
    # Sample image descriptions (you can replace with actual sample images)
    sample_descriptions = {
        "Cloudy": "Satellite images showing cloud formations and weather patterns",
        "Desert": "Arid landscapes with sand dunes and desert terrain",
        "Green_Area": "Vegetation, forests, and agricultural areas",
        "Water": "Oceans, lakes, rivers, and other water bodies"
    }
    
    cols = st.columns(4)
    for i, (class_name, description) in enumerate(sample_descriptions.items()):
        with cols[i]:
            st.markdown(f"**{class_name}**")
            st.write(description)
            # You can add actual sample images here
            st.info(f"Upload an image to classify if it's {class_name.lower()}")

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🛰️ Satellite Image Classifier</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.selectbox("Choose a page:", ["🔍 Image Classification", "📊 Model Information", "📈 Model Performance"])
        
        st.markdown("### About")
        st.markdown("""
        This app classifies satellite images into four categories:
        - **Cloudy**: Cloud formations
        - **Desert**: Arid landscapes
        - **Green Area**: Vegetation
        - **Water**: Water bodies
        """)
        
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload a satellite image
        2. Wait for the model to process
        3. View the prediction results
        4. Check confidence scores
        """)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading model... Please wait..."):
            st.session_state.model = load_trained_model()
            if st.session_state.model:
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model. Please ensure 'Modelenv.v1.h5' is in the same directory.")
                st.stop()
    
    if page == "🔍 Image Classification":
        st.markdown('<div class="info-box">Upload a satellite image to classify it into one of the four categories.</div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a satellite image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a satellite image to classify"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📷 Uploaded Image")
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.markdown("**Image Information:**")
                st.write(f"- **Format:** {img.format}")
                st.write(f"- **Size:** {img.size}")
                st.write(f"- **Mode:** {img.mode}")
            
            with col2:
                st.markdown("### 🎯 Prediction Results")
                
                if st.button("🔍 Classify Image", key="classify_btn"):
                    with st.spinner("Analyzing image..."):
                        # Preprocess and predict
                        img_array = preprocess_image(img)
                        predicted_class, confidence, all_predictions = predict_image(st.session_state.model, img_array)
                        
                        # Display results
                        predicted_label = class_names[predicted_class]
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🎯 Prediction: {predicted_label}</h2>
                            <h3>Confidence: {confidence:.2%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence chart
                        fig = create_confidence_chart(all_predictions, class_names)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results
                        st.markdown("### 📊 Detailed Results")
                        results_df = pd.DataFrame({
                            'Class': class_names,
                            'Confidence': all_predictions,
                            'Percentage': [f"{p:.2%}" for p in all_predictions]
                        })
                        results_df = results_df.sort_values('Confidence', ascending=False)
                        st.dataframe(results_df, use_container_width=True)
        
        else:
            display_sample_images()
    
    elif page == "📊 Model Information":
        st.markdown("### 🧠 Model Architecture")
        
        if st.session_state.model:
            # Model summary
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Model Details:**")
                st.write("- **Type:** Convolutional Neural Network (CNN)")
                st.write("- **Input Size:** 255 x 255 x 3")
                st.write("- **Classes:** 4 (Cloudy, Desert, Green Area, Water)")
                st.write("- **Optimizer:** Adam")
                st.write("- **Loss Function:** Categorical Crossentropy")
                
                # Model layers info
                st.markdown("**Architecture:**")
                st.write("- Conv2D (32 filters, 3x3)")
                st.write("- MaxPooling2D (2x2)")
                st.write("- Conv2D (64 filters, 3x3)")
                st.write("- MaxPooling2D (2x2)")
                st.write("- Conv2D (128 filters, 3x3)")
                st.write("- MaxPooling2D (2x2)")
                st.write("- Flatten")
                st.write("- Dense (128 neurons, ReLU)")
                st.write("- Dropout (0.5)")
                st.write("- Dense (4 neurons, Softmax)")
            
            with col2:
                st.markdown("**Training Configuration:**")
                st.write("- **Epochs:** 5")
                st.write("- **Batch Size:** 32")
                st.write("- **Validation Split:** 20%")
                st.write("- **Data Augmentation:** Yes")
                
                st.markdown("**Data Augmentation:**")
                st.write("- Rescaling (1/255)")
                st.write("- Shear Range: 0.2")
                st.write("- Zoom Range: 0.2")
                st.write("- Horizontal Flip: Yes")
                st.write("- Rotation Range: 45°")
                st.write("- Vertical Flip: Yes")
        
        # Class descriptions
        st.markdown("### 📋 Class Descriptions")
        class_descriptions = {
            "Cloudy": "Images containing cloud formations, weather patterns, and atmospheric conditions",
            "Desert": "Arid landscapes featuring sand dunes, rocky terrain, and desert environments",
            "Green_Area": "Vegetation-rich areas including forests, agricultural lands, and green spaces",
            "Water": "Water bodies such as oceans, seas, lakes, rivers, and other aquatic features"
        }
        
        for class_name, description in class_descriptions.items():
            st.markdown(f"**{class_name}:** {description}")
    
    elif page == "📈 Model Performance":
        st.markdown("### 📊 Model Performance Metrics")
        
        # Performance metrics (you can replace with actual metrics)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h3>Accuracy</h3>
                <h2>85.4%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h3>Precision</h3>
                <h2>84.2%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h3>Recall</h3>
                <h2>83.8%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-container">
                <h3>F1-Score</h3>
                <h2>84.0%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Training history visualization
        st.markdown("### 📈 Training History")
        
        # Sample training data (replace with actual training history)
        epochs = list(range(1, 6))
        train_acc = [0.65, 0.72, 0.78, 0.82, 0.85]
        val_acc = [0.62, 0.69, 0.75, 0.79, 0.82]
        train_loss = [0.85, 0.65, 0.45, 0.35, 0.28]
        val_loss = [0.90, 0.70, 0.50, 0.40, 0.32]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training & Validation Accuracy', 'Training & Validation Loss')
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Training Accuracy'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy'),
            row=1, col=1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Training Loss'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix (sample data)
        st.markdown("### 🔄 Confusion Matrix")
        confusion_data = np.array([
            [45, 3, 1, 1],
            [2, 47, 1, 0],
            [1, 2, 44, 3],
            [0, 1, 2, 47]
        ])
        
        fig_cm = px.imshow(
            confusion_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=class_names,
            y=class_names,
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig_cm.add_annotation(
                    x=j, y=i,
                    text=str(confusion_data[i, j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_data[i, j] > 25 else "black")
                )
        
        st.plotly_chart(fig_cm, use_container_width=True)

if __name__ == "__main__":
    main()
