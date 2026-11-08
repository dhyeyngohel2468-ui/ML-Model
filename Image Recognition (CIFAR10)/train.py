import os
import cv2
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = np.reshape(image, (1, 32, 32, 3))
    return image

def train_model():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test Accuracy:", test_acc)

    model.save('cifar10_model.keras')

    return model

def classify_image(image_path, model):
    input_image = preprocess_image(image_path)
    prediction = model.predict(input_image)
    predicted_label = np.argmax(prediction)

    classes = {
        0: 'Airplane',
        1: 'Automobile',
        2: 'Bird',
        3: 'Cat',
        4: 'Deer',
        5: 'Dog',
        6: 'Frog',
        7: 'Horse',
        8: 'Ship',
        9: 'Truck'
    }

    predicted_class = classes[predicted_label]

    return predicted_class

model_path = 'cifar10_model.keras'
if not os.path.exists(model_path):
    print("Model not found. Training a new model.")
    model = train_model()
else:
    print("Loading the existing trained model.")
    model = load_model(model_path)

image_path = '/content/frog.jpeg'

if os.path.exists(image_path):
    predicted_class = classify_image(image_path, model)
    print(f'The input image is predicted to be a {predicted_class}.')

    user_feedback = input("Was the prediction correct? (yes/no): ").lower()

    if user_feedback == 'no':
        print("Retraining the model with the updated data.")
        model = train_model()

        predicted_class = classify_image(image_path, model)
        print(f'After retraining, the input image is predicted to be a {predicted_class}.')
else:
    print(f"Error: Image file not found at {image_path}.")