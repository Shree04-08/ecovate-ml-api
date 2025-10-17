# api_app.py
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import uvicorn

# ---- Initialize FastAPI app ----
app = FastAPI()

# ---- Allow Wix Studio (CORS setup) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your wix domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model ----
MODEL_PATH = "resnet50_plastic.keras"
model = load_model(MODEL_PATH, compile=False)

# ---- Plastic type labels ----
categories = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Others"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    label = categories[class_index]
    confidence = float(prediction[0][class_index]) * 100

    return {"plastic_type": label, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
