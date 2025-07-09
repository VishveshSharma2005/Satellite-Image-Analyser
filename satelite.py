import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set Streamlit page config
st.set_page_config(page_title="Satellite Image Classifier", layout="wide")

st.title("🌍 Satellite Image Classification with CNN")

# Define the folder labels (edit these paths to your real dataset location)
labels = {
    "dataset/cloudy": "Cloudy",
    "dataset/desert": "Desert",
    "dataset/green_area": "Green_Area",
    "dataset/water": "Water",
}

# Sidebar: Model parameters
st.sidebar.header("⚙️ Model Settings")
epochs = st.sidebar.slider("Number of Epochs", 1, 20, 5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
img_size = (128, 128)

# Load dataset
st.header("📊 Loading Dataset")
data = pd.DataFrame(columns=['image_path', 'label'])
for folder, label in labels.items():
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                data = pd.concat([data, pd.DataFrame({'image_path': [file_path], 'label': [label]})], ignore_index=True)
    else:
        st.warning(f"Folder not found: {folder}")

st.write(f"Total images found: {len(data)}")

# Show sample images
st.subheader("🖼️ Sample Images")
fig, axes = plt.subplots(len(labels), 10, figsize=(15, len(labels)*1.5))

for i, (folder, label_name) in enumerate(labels.items()):
    if os.path.exists(folder):
        image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        sample_images = np.random.choice(image_files, min(10, len(image_files)), replace=False)
        for j in range(10):
            axes[i, j].axis('off')
            if j < len(sample_images):
                img = Image.open(os.path.join(folder, sample_images[j]))
                axes[i, j].imshow(img)
                axes[i, j].set_title(label_name, fontsize=8)
plt.tight_layout()
st.pyplot(fig)

# Split data
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=45,
                                   vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="label",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="label",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Build model
st.subheader("🧠 Building CNN Model")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary(print_fn=lambda x: st.text(x))

# Train model
if st.button("🚀 Train Model"):
    with st.spinner("Training..."):
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator
        )
        model.save('model_satellite.h5')
    st.success("✅ Model trained and saved as 'model_satellite.h5'")

    # Plot accuracy
    st.subheader("📈 Training Metrics")
    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(history.history['accuracy'], label='Training Accuracy')
    ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax_acc.set_title('Accuracy')
    ax_acc.legend()
    st.pyplot(fig_acc)

    # Plot loss
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(history.history['loss'], label='Training Loss')
    ax_loss.plot(history.history['val_loss'], label='Validation Loss')
    ax_loss.set_title('Loss')
    ax_loss.legend()
    st.pyplot(fig_loss)

    # Confusion matrix
    st.subheader("🔍 Confusion Matrix")
    class_names = list(labels.values())
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cax = ax_cm.matshow(cm, cmap=plt.cm.Blues)
    fig_cm.colorbar(cax)
    ax_cm.set_xticklabels([''] + class_names, rotation=45)
    ax_cm.set_yticklabels([''] + class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax_cm.text(j, i, cm[i, j], va='center', ha='center', color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig_cm)

# Inference section
st.subheader("🧪 Try Prediction on New Image")
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    img = Image.open(uploaded_file).resize(img_size)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    model = load_model('model_satellite.h5')
    pred = model.predict(img_array)
    pred_label = class_names[np.argmax(pred)]
    st.success(f"✅ Predicted Class: **{pred_label}**")

st.caption("Made with ❤️ using Streamlit")
