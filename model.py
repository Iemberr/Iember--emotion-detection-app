import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os

# Paths to dataset folders
train_dir = "dataset/train"
test_dir = "dataset/test"
MODEL_PATH = "emotion_model.h5"

# --- Train model if not already saved ---
if not os.path.exists(MODEL_PATH):
    print("⚙️ Training new CNN model (this will take a few minutes)...")

    # Image data generators for preprocessing
    train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Load and preprocess images
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode="categorical"
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode="categorical"
    )

    # Define CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_data, validation_data=test_data, epochs=20)

    # Save trained model
    model.save(MODEL_PATH)
    print("✅ Model training complete and saved as emotion_model.h5")
else:
    print("✅ Pre-trained model found, skipping training.")

# --- Load model for predictions ---
model = load_model(MODEL_PATH)

# Get emotion class labels (for consistent predictions)
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=1,
    color_mode="grayscale",
    class_mode="categorical"
)
emotion_classes = list(train_data.class_indices.keys())

# --- Function to analyze emotion from a single image ---
def analyze_emotion(image_path):
    img = load_img(image_path, target_size=(48, 48), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    emotion_classes = list(train_data.class_indices.keys())
    dominant_emotion = emotion_classes[np.argmax(prediction)]

    return f"Detected Emotion: {dominant_emotion}"

