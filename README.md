# ML Models — Image Recognition

Two Convolutional Neural Network (CNN) models built with TensorFlow/Keras for image classification tasks, trained and tested on Google Colab.

---

## 🧠 Models

### 1. CIFAR-10 Image Recognition (`/cifar10`)
Classifies real-world images into 10 categories using a CNN trained on the CIFAR-10 dataset.

**Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

**Results:**
| Metric | Value |
|---|---|
| Training Accuracy | ~71.96% |
| Test Accuracy | ~70.54% |
| Epochs | 10 |

---

### 2. MNIST Handwritten Digit Recognition (`/mnist`)
Classifies handwritten digit images (0–9) using a CNN trained on the MNIST dataset.

**Classes:** Digits 0 through 9

**Model trained on:** 60,000 handwritten digit images

---

## 🔧 Tech Stack

| | |
|---|---|
| Language | Python 3 |
| Framework | TensorFlow / Keras |
| Libraries | NumPy, OpenCV (cv2) |
| Platform | Google Colab |
| Model Architecture | CNN (Conv2D + MaxPooling + Dropout + Dense) |

---

## 🏗️ Model Architecture (Both Models)

```
Conv2D(32) → MaxPooling
Conv2D(64) → MaxPooling
Conv2D(64)
Flatten
Dense(128) → Dropout(0.5)
Dense(10, softmax)
```

---

## ✨ Features

- Auto-trains the model if no saved model is found
- Loads existing saved model to avoid retraining
- Takes a custom image as input and predicts the class
- User feedback loop — if prediction is wrong, model retrains automatically

---

## ▶️ How to Run

1. Open the `.ipynb` file in **Google Colab**
2. Upload your test image to `/content/`
3. Update the `image_path` variable with your image filename
4. Run all cells

---

## 👨‍💻 Developed By

**Dhyey Gohel**
