
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "data/chest_xray/test",
    image_size=(150, 150),
    batch_size=16,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"],
    shuffle=False
)

# Load trained model
model = tf.keras.models.load_model(
    "models/pneumonia_model.keras"
)

# Actual labels
y_true = np.concatenate([
    labels.numpy().ravel()
    for images, labels in test_data
]).astype(int)

# Predictions
y_prob = model.predict(test_data).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# Evaluation results
print("\nAccuracy:", accuracy_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
))
