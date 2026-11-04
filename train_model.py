
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Dataset folders
BASE_DIR = Path("data/chest_xray")

train_dir = BASE_DIR / "train"
val_dir = BASE_DIR / "val"
test_dir = BASE_DIR / "test"

# Image settings
IMG_SIZE = (150, 150)
BATCH_SIZE = 16

# Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"]
)

# Load validation images
val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"]
)

# Load testing images
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"],
    shuffle=False
)

# Improve data loading
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# Create CNN model
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print("Test Accuracy:", accuracy)

# Save trained model
Path("models").mkdir(exist_ok=True)
model.save("models/pneumonia_model.keras")

print("Model training completed successfully!")
