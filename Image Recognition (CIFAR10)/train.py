
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from keras.utils import to_categorical




BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cifar10_model.keras"

EPOCHS = 10
BATCH_SIZE = 64
NUM_CLASSES = 10

CLASS_NAMES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]




print("\nLoading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)




print("\nPreprocessing dataset...")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)


print("\nBuilding CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),

    MaxPooling2D((2, 2)),

    Conv2D(64,(3, 3), activation="relu"),

    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.5),

    Dense(NUM_CLASSES, activation="softmax")
])



model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


print("\nStarting model training...")

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)



print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n========== MODEL RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



MODEL_DIR.mkdir(parents=True, exist_ok=True)

model.save(MODEL_PATH)

print("\nModel saved successfully!")
print(f"Location: {MODEL_PATH}")



print("\nTesting a sample prediction...")

sample_image = np.expand_dims(x_test[0], axis=0)

prediction = model.predict(sample_image, verbose=0)

predicted_class = np.argmax(prediction[0])
confidence = np.max(prediction[0]) * 100

actual_class = np.argmax(y_test[0])

print(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
print(f"Confidence: {confidence:.2f}%")
print(f"Actual Class: {CLASS_NAMES[actual_class]}")

print("\nTraining process completed!")
