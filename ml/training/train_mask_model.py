import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# 1. Dataset paths
# -----------------------------
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

# -----------------------------
# 2. Model save paths
# -----------------------------
MODEL_DIR = Path("ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "mask_detector.keras"
FINAL_MODEL_PATH = MODEL_DIR / "mask_detector_final.keras"

# -----------------------------
# 3. Training settings
# -----------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 12

# -----------------------------
# 4. Load training dataset
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# -----------------------------
# 5. Load validation dataset
# -----------------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Class names:", train_ds.class_names)
print("Expected: ['masked', 'unmasked']")
print("masked = 0, unmasked = 1")

# -----------------------------
# 6. Improve dataset loading speed
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# 7. Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
])

# -----------------------------
# 8. Load MobileNetV2 pretrained model
# -----------------------------
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# 9. Build model
# -----------------------------
inputs = layers.Input(shape=(160, 160, 3))

# Randomly augment training images
x = data_augmentation(inputs)

# MobileNetV2-specific preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Use MobileNetV2 as feature extractor
x = base_model(x, training=False)

# Convert feature maps into one vector
x = layers.GlobalAveragePooling2D()(x)

# Dropout helps prevent overfitting
x = layers.Dropout(0.35)(x)

# Binary output:
# close to 0 = masked
# close to 1 = unmasked
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# -----------------------------
# 10. Compile model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 11. Save best model during training
# -----------------------------
checkpoint = ModelCheckpoint(
    filepath=str(BEST_MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Stop training if validation loss stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# -----------------------------
# 12. Train model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# 13. Save final model
# -----------------------------
model.save(FINAL_MODEL_PATH)

print("Training complete.")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Final model saved to: {FINAL_MODEL_PATH}")

# -----------------------------
# 14. Plot accuracy
# -----------------------------
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Mask Classifier Accuracy")
plt.legend()
plt.show()

# -----------------------------
# 15. Plot loss
# -----------------------------
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Mask Classifier Loss")
plt.legend()
plt.show()