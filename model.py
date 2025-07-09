import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_model():
    """Create and compile the CNN model."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_dataset():
    """Prepare the dataset from folders."""
    # Create an empty dataframe
    data = pd.DataFrame(columns=['image_path', 'label'])
    
    # Define the labels/classes
    labels = {
        r"/content/dataset/Satellite Image data/cloudy": "Cloudy",
        r"/content/dataset/Satellite Image data/desert": "Desert",
        r"/content/dataset/Satellite Image data/green_area": "Green_Area",
        r"/content/dataset/Satellite Image data/water": "Water",
    }
    
    # Validate folder paths and process images
    for folder in labels:
        if not os.path.exists(folder):
            print(f"Warning: The folder {folder} does not exist.")
            continue
        
        # Process each image in the folder
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            if os.path.isfile(image_path):  # Only process files
                label = labels[folder]
                new_row = pd.DataFrame({'image_path': [image_path], 'label': [label]})
                data = pd.concat([data, new_row], ignore_index=True)
    
    return data

def train_model():
    """Complete training pipeline."""
    print("Preparing dataset...")
    data = prepare_dataset()
    
    if data.empty:
        print("No data found. Please check your folder paths.")
        return
    
    print(f"Dataset contains {len(data)} images")
    print(data['label'].value_counts())
    
    # Save dataset info
    data.to_csv('image_dataset.csv', index=False)
    
    # Split data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=45,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="label",
        target_size=(255, 255),
        batch_size=32,
        class_mode="categorical"
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_path",
        y_col="label",
        target_size=(255, 255),
        batch_size=32,
        class_mode="categorical"
    )
    
    # Create and train model
    print("Creating model...")
    model = create_model()
    model.summary()
    
    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=5,
        validation_data=test_generator,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    num_samples = test_df.shape[0]
    score = model.evaluate(test_generator, steps=num_samples//32+1)
    print(f"Test accuracy: {score[1]:.4f}")
    
    # Save model
    model.save('Modelenv.v1.h5')
    print("Model saved as 'Modelenv.v1.h5'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model

if __name__ == "__main__":
    model = train_model()