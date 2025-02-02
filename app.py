from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io
import numpy as np

# Load your pre-trained model (for example, model.h5)
model = load_model('asl_model.h5')

# Initialize FastAPI app
app = FastAPI()

# Define the route for image prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image data from the file upload
    image_data = await file.read()
    
    # Convert byte data to image
    image = Image.open(io.BytesIO(image_data))
    
    # Preprocess the image to match model input requirements
    image = image.resize((64, 64))  # Resize to model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Run inference (prediction) on the image using the model
    predictions = model.predict(image_array)
    
    # Get the predicted class index (you can modify this based on your use case)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]

    # Return the prediction result
    return JSONResponse(content={"prediction": str(predicted_class)})

