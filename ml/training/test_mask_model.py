import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "ml/models/mask_detector.keras"

# Change this to a real image from your validation folder
IMAGE_PATH = "data/processed/val/masked/with_mask_5.jpg"

model = tf.keras.models.load_model(MODEL_PATH)

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise Exception(f"Could not load image: {IMAGE_PATH}")

# Resize image to match model input
img = cv2.resize(img, (160, 160))

# OpenCV uses BGR, TensorFlow expects RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Add batch dimension:
# from (224, 224, 3) to (1, 224, 224, 3)
img = np.expand_dims(img, axis=0)

prediction = model.predict(img, verbose=0)[0][0]

if prediction >= 0.5:
    label = "unmasked"
    confidence = prediction
else:
    label = "masked"
    confidence = 1 - prediction

print("Prediction:", label)
print("Confidence:", round(float(confidence), 4))
print("Raw output:", round(float(prediction), 4))