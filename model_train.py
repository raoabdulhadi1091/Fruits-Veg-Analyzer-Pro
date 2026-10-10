import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np

# Image dimensions
img_height = 180
img_width = 180

# Local dataset folder path (Aapke project folder ke andar 'Fruits_Vegetables' folder hona chahiye)
dataset_dir = "Fruits_Vegetables"

# Training dataset
train_images = tf.keras.utils.image_dataset_from_directory(
    f"{dataset_dir}/train",
    image_size=(img_height, img_width),
    batch_size=32,
    shuffle=True,
    seed=42
)

data_labels = train_images.class_names
print("Detected Class Labels:", data_labels)

# Test dataset
test_images = tf.keras.utils.image_dataset_from_directory(
    f"{dataset_dir}/test",
    image_size=(img_height, img_width),
    batch_size=32,
    shuffle=True,
    seed=42
)

# Validation dataset
val_images = tf.keras.utils.image_dataset_from_directory(
    f"{dataset_dir}/validation",
    image_size=(img_height, img_width),
    batch_size=32,
    shuffle=True,
    seed=42
)

# Model Architecture
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(data_labels))
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Model Training
print("Starting model training...")
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=5
)

# Model save karein taake app.py isay use kar sake
model.save("Fruits_Vegetables_model.keras")
print("Model successfully trained and saved as Fruits_Vegetables_model.keras!")