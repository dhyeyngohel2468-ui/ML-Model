import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File
from keras.models import load_model


app = FastAPI()
model = load_model('models/cifar10_model.keras')

classes = [
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

def preprocess_image(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid Image")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (32,32))
    image = image.astype('float32')/255.0
    
    image = np.expand_dims(image, axis=0)
    
    return image

@app.get('/')
def home():
    return {'Message': 'CIFAR-10 Image Recognition Model Home Page'}

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image)
        
        predicted_index = int(np.argmax(predictions[0]))
        
        confidence = float(predictions[0][predicted_index])
        
        return {
            'file_name': file.filename,
            'predicted_class': classes[predicted_index],
            'confidence': round(confidence*100, 2)    
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Something Went Wrong {str(e)}'
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )