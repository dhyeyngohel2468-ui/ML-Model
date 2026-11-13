# CIFAR-10 Image Classification using CNN

A Deep Learning project that classifies images into 10 different categories using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The project uses TensorFlow and Keras for model development and supports local model training, image prediction, and API-based inference.

## Project Overview

This project implements a CNN-based image classification system using the CIFAR-10 dataset. The model learns visual patterns from RGB images and predicts which of the ten predefined classes an image belongs to.

The project includes:
- CNN model development using TensorFlow and Keras.
- Local model training using a Python script.
- Image preprocessing using OpenCV.
- Model evaluation on the CIFAR-10 test dataset.
- Saving the trained model in `.keras` format.
- Image prediction through a standalone Python script.
- FastAPI integration for image-upload-based predictions.

## Dataset

**CIFAR-10** is a publicly available image classification dataset containing 60,000 color images.

| Dataset property | Description |
|---|---|
| Total images | 60,000 |
| Training images | 50,000 |
| Testing images | 10,000 |
| Image dimensions | 32 × 32 |
| Channels | 3 (RGB) |
| Number of classes | 10 |

### Classification Categories

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## CNN Architecture

The model uses the following architecture:

| Layer | Configuration |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| Flatten | Converts feature maps into a vector |
| Dense | 128 neurons, ReLU |
| Dropout | 0.5 |
| Output Dense | 10 neurons, Softmax |

### Model Configuration

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 10
- Batch Size: 64
- Validation Split: 20%

## Data Preprocessing

The dataset undergoes the following preprocessing steps:

1. Load CIFAR-10 training and testing datasets.
2. Convert image pixel values from the range 0–255 to 0–1.
3. Convert class labels into one-hot encoded vectors.
4. Pass the processed images to the CNN for training and evaluation.

For external image prediction, OpenCV is used to load images, convert BGR to RGB, resize them to 32×32 pixels, and normalize pixel values.

## Project Structure

```text
Cifar-10/
│
├── models/
│   └── cifar10_model.keras
│
├── train.py
├── main.py
├── requirements.txt
├── README.md
│
└── venv/
```

| File / Folder | Purpose |
|---|---|
| `train.py` | Loads the dataset, builds and trains the CNN, evaluates it, and saves the model. |
| `main.py` | Implements the FastAPI endpoint for image-upload-based predictions. |
| `models/` | Stores the trained Keras model. |
| `requirements.txt` | Contains the project dependencies. |

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <https://github.com/dhyeyngohel2468-ui/ML-Model/tree/main/Image%20Recognition%20(CIFAR10)>
cd Cifar-10
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Model Training

Run the following command from the project root:

```bash
python train.py
```

The training script automatically:

1. Downloads or loads the CIFAR-10 dataset.
2. Preprocesses the training and testing images.
3. Builds the CNN architecture.
4. Trains the model using the configured parameters.
5. Evaluates the trained model on the test dataset.
6. Saves the trained model to:

```text
models/cifar10_model.keras
```

**Note:** Running `train.py` trains the model from scratch and overwrites the existing saved model. The first dataset download requires an internet connection.


## FastAPI Integration

The project also provides a FastAPI-based prediction interface.

Start the API:

```bash
uvicorn main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint to upload an image and receive its predicted class and confidence score.

## Technologies Used

- **Python** – Core programming language.
- **TensorFlow & Keras** – CNN development and model training.
- **NumPy** – Numerical operations and array processing.
- **OpenCV** – Image loading, color conversion, and resizing.
- **FastAPI** – REST API for image prediction.
- **Uvicorn** – ASGI server for running the API.
- **CIFAR-10** – Image classification dataset.

## Model Performance

The initial Colab-trained model achieved the following results:

| Metric | Result |
|---|---:|
| Training Accuracy | 70.35% |
| Validation Accuracy | 69.20% |
| Test Accuracy | 68.91% |
| Test Loss | 0.9026 |

These metrics represent the previously trained Colab model. Results from a new local training run may differ.

## Limitations

- The model is trained on low-resolution 32×32 CIFAR-10 images.
- Prediction accuracy on arbitrary external images may vary due to differences in image quality, backgrounds, lighting, and image composition.
- Softmax confidence represents the model's highest output score and is not necessarily a calibrated probability.
- The current system uses a predefined dataset and does not automatically learn from user corrections.

## Future Improvements

- Improve classification accuracy using data augmentation.
- Experiment with deeper CNN architectures and Batch Normalization.
- Generate confusion matrices and detailed classification reports.
- Improve prediction performance on external images.
- Explore deployment through a cloud-hosted API.

## Author

**Dhyey Gohel**

B.Tech – Computer Engineering
